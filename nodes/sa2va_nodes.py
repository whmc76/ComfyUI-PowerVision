"""
PowerVision Sa2VA 节点
Segment Anything 2 Video Assistant 节点实现
基于 ByteDance/Sa2VA 模型，结合 SAM2 和 VLLM
"""

import torch
import numpy as np
import os
import gc
import threading
import time
from contextlib import nullcontext
from PIL import Image
import sys

# 导入全局配置
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from config import be_quiet
except ImportError:
    be_quiet = False


class PowerVisionSa2VASegmentation:
    """PowerVision Sa2VA 分割节点"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None  # 跟踪当前加载的模型
        self._download_cancelled = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [
                        "ByteDance/Sa2VA-InternVL3-2B",
                        "ByteDance/Sa2VA-InternVL3-8B",
                        "ByteDance/Sa2VA-InternVL3-14B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-3B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-7B",
                        "ByteDance/Sa2VA-Qwen3-VL-4B",
                    ],
                    {"default": "ByteDance/Sa2VA-Qwen3-VL-4B"},
                ),
                "image": ("IMAGE",),  # ComfyUI 标准图像输入
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0},
                ),  # 二值化阈值
                "use_8bit_quantization": (
                    "BOOLEAN",
                    {"default": False},
                ),  # 启用 8 位量化
                "use_flash_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),  # 使用 Flash Attention 提高效率
                "segmentation_prompt": (
                    "STRING",
                    {
                        "default": "Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIST", "MASK", "IMAGE")
    RETURN_NAMES = ("text_outputs", "masks", "mask_images")
    FUNCTION = "process_with_sa2va"
    CATEGORY = "PowerVision/Sa2VA"

    def check_transformers_version(self):
        """检查 transformers 版本是否支持 Sa2VA 模型"""
        try:
            from transformers import __version__ as transformers_version

            version_parts = transformers_version.split(".")
            major, minor = int(version_parts[0]), int(version_parts[1])

            # Sa2VA 模型需要 transformers >= 4.57.0
            if major < 4 or (major == 4 and minor < 57):
                return (
                    False,
                    f"Sa2VA 需要 transformers >= 4.57.0，当前版本: {transformers_version}",
                )
            return True, transformers_version
        except Exception as e:
            return False, f"检查 transformers 版本时出错: {e}"

    def install_transformers_upgrade(self):
        """尝试自动升级 transformers"""
        try:
            import subprocess
            import sys

            print("🔄 正在尝试升级 transformers...")

            # 尝试升级 transformers
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "transformers>=4.57.0",
                    "--upgrade",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Transformers 升级成功")
                print("🔄 请重启 ComfyUI 以使用升级后的版本")
                return True
            else:
                print(f"❌ 升级 transformers 失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ 升级 transformers 时出错: {e}")
            return False

    def load_model(
        self,
        model_name,
        use_flash_attn=True,
        dtype="auto",
        cache_dir="",
        use_8bit_quantization=False,
    ):
        """加载指定的 Sa2VA 模型，只加载一次并缓存"""
        if (
            self.model is None
            or self.processor is None
            or self.current_model_name != model_name
        ):
            # 清理任何现有的模型状态
            if self.model is not None:
                try:
                    del self.model
                    self.model = None
                except:
                    pass
            if self.processor is not None:
                try:
                    del self.processor
                    self.processor = None
                except:
                    pass
            self.current_model_name = None

            # 在加载新模型前清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            if not be_quiet:
                print(f"🔄 正在加载 Sa2VA 模型: {model_name}")

            # 检查 transformers 版本
            version_ok, version_info = self.check_transformers_version()
            if not version_ok:
                print(f"❌ {version_info}")
                print("💡 正在尝试自动升级...")

                if self.install_transformers_upgrade():
                    print("⚠️  需要重启 ComfyUI 才能使升级生效")
                    return False
                else:
                    print(
                        "💡 需要手动升级: pip install transformers>=4.57.0 --upgrade"
                    )
                    return False

            # 确定缓存目录
            effective_cache_dir = None
            if cache_dir and cache_dir.strip():
                effective_cache_dir = cache_dir.strip()
                if not be_quiet:
                    print(f"   使用自定义缓存目录: {effective_cache_dir}")
            else:
                # 使用 ComfyUI-PowerVision 文件夹中的本地缓存
                # 获取当前文件所在目录（PowerVision nodes 文件夹）
                current_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                effective_cache_dir = os.path.join(
                    current_dir, ".cache", "huggingface", "hub"
                )

                # 如果缓存目录不存在则创建
                os.makedirs(effective_cache_dir, exist_ok=True)

                if not be_quiet:
                    print(f"   使用本地 ComfyUI-PowerVision 缓存: {effective_cache_dir}")

            # 处理 dtype 转换并给出适当警告
            # 稳健地解析目标 dtype 以减少内存同时保持兼容性
            auto_selected = False
            if dtype == "auto":
                auto_selected = True
                if torch.cuda.is_available():
                    # 如果支持则优先使用 bf16，否则使用 fp16；在 CPU 上坚持使用 fp32
                    if (
                        hasattr(torch.cuda, "is_bf16_supported")
                        and torch.cuda.is_bf16_supported()
                    ):
                        resolved_dtype = torch.bfloat16
                    else:
                        resolved_dtype = torch.float16
                else:
                    resolved_dtype = torch.float32
                if not be_quiet:
                    print(
                        f"   自动选择 dtype: {resolved_dtype} (基于设备能力)"
                    )
            else:
                # 映射显式 dtype 请求
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                resolved_dtype = dtype_map.get(str(dtype), torch.float32)
                if not be_quiet:
                    print(f"   模型目标 dtype: {resolved_dtype}")

            try:
                # 在此处导入以捕获缺失的依赖
                from transformers import AutoProcessor, AutoModel

                # 构建模型加载参数
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }

                # 如果指定了缓存目录则添加
                if effective_cache_dir:
                    model_kwargs["cache_dir"] = effective_cache_dir

                # 如果请求则添加 8 位量化
                if use_8bit_quantization:
                    try:
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig

                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        if not be_quiet:
                            print("   使用 bitsandbytes 进行 8 位量化")
                    except ImportError:
                        if not be_quiet:
                            print(
                                "   警告: bitsandbytes 不可用，跳过 8 位量化"
                            )
                            print("   安装方法: pip install bitsandbytes")

                # 如果可用且请求则添加 flash attention
                if use_flash_attn:
                    try:
                        import flash_attn

                        model_kwargs["use_flash_attn"] = True
                        if not be_quiet:
                            print("   使用 Flash Attention")
                    except ImportError:
                        if not be_quiet:
                            print(
                                "   Flash Attention 不可用，继续不使用它"
                            )
                            print("   安装方法: pip install flash-attn")
                        # 如果不可用则不添加 flash_attn 到 model_kwargs
                else:
                    if not be_quiet:
                        print("   用户已禁用 Flash Attention")

                # 使用解析的 dtype 加载以减少内存（如果使用量化则跳过）
                if resolved_dtype is not None and not use_8bit_quantization:
                    model_kwargs["torch_dtype"] = resolved_dtype

                # 使用增强的进度和取消支持加载模型
                print("🔄 开始模型下载/加载...")
                print("   注意: 大型模型可能需要几分钟才能下载")

                # 检查 ComfyUI 取消是否可用
                def is_cancelled():
                    try:
                        # 尝试访问 ComfyUI 的执行状态
                        import execution

                        return (
                            execution.current_task is not None
                            and execution.current_task.cancelled
                        )
                    except:
                        try:
                            # 替代的 ComfyUI 取消检查
                            import model_management

                            return model_management.processing_interrupted()
                        except:
                            return False

                # 增强的下载，支持可取消的 snapshot_download 和仓库大小摘要
                try:
                    from huggingface_hub import HfApi, snapshot_download
                    from huggingface_hub.utils import tqdm as hub_tqdm

                    # 打印仓库大小摘要以设置期望
                    try:
                        api = HfApi()
                        info = api.repo_info(
                            model_name, repo_type="model", files_metadata=True
                        )
                        sizes = []
                        file_entries = []
                        for s in getattr(info, "siblings", []):
                            sz = getattr(s, "size", None)
                            if sz is None:
                                lfs = getattr(s, "lfs", None)
                                sz = (
                                    getattr(lfs, "size", None)
                                    if lfs is not None
                                    else None
                                )
                            if isinstance(sz, int) and sz > 0:
                                sizes.append(sz)
                                file_entries.append(
                                    (
                                        getattr(
                                            s, "rfilename", getattr(s, "path", "file")
                                        ),
                                        sz,
                                    )
                                )
                        total_bytes = sum(sizes)
                        if total_bytes > 0:
                            gb = total_bytes / (1024**3)
                            print(
                                f"   估计总下载大小: {gb:.2f} GB，共 {len(sizes)} 个文件"
                            )
                            largest = sorted(
                                file_entries, key=lambda x: x[1], reverse=True
                            )[:5]
                            if largest:
                                print("   最大文件:")
                                for name, sz in largest:
                                    print(f"     • {name}: {sz / (1024**2):.1f} MB")
                    except Exception as e:
                        if not be_quiet:
                            print(f"   无法确定仓库大小: {e}")

                    class CancellableTqdm(hub_tqdm):
                        def update(self, n=1):
                            if is_cancelled():
                                raise KeyboardInterrupt("用户取消了下载")
                            return super().update(n)

                    local_dir = snapshot_download(
                        repo_id=model_name,
                        cache_dir=effective_cache_dir if effective_cache_dir else None,
                        resume_download=True,
                        local_files_only=False,
                        tqdm_class=CancellableTqdm,
                    )

                    # 从本地目录加载模型以避免额外的网络调用
                    model_kwargs_local = dict(model_kwargs)
                    model_kwargs_local["local_files_only"] = True
                    model_kwargs_local.pop("cache_dir", None)
                    self.model = AutoModel.from_pretrained(
                        local_dir, **model_kwargs_local
                    ).eval()
                    print("✅ 模型文件已从缓存下载并加载")

                except KeyboardInterrupt:
                    print("\n⚠️ 模型下载已取消")
                    return False
                except Exception as e:
                    if not be_quiet:
                        print(f"   增强下载失败: {e}")
                        print("   使用标准下载...")
                    self.model = AutoModel.from_pretrained(
                        model_name, **model_kwargs
                    ).eval()

                # 将模型放置在适当的设备上，并使用 dtype 以减少内存
                target_device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )

                # 如果使用 8 位量化则跳过设备/dtype 转换（已处理）
                if not use_8bit_quantization:
                    try:
                        # 先移动到设备，然后根据需要处理 dtype
                        self.model = self.model.to(device=target_device)
                        # 仅当它与当前不同且受支持时才转换 dtype
                        if (
                            hasattr(self.model, "dtype")
                            and self.model.dtype != resolved_dtype
                        ):
                            try:
                                self.model = self.model.to(dtype=resolved_dtype)
                            except Exception as e:
                                if not be_quiet:
                                    print(
                                        f"   注意: 无法转换为 {resolved_dtype}，保持原始 dtype: {e}"
                                    )
                    except Exception as e:
                        if not be_quiet:
                            print(f"   警告: 模型放置失败: {e}")

                if not be_quiet:
                    if use_8bit_quantization:
                        print(
                            f"   模型已使用 8 位量化加载到 {target_device}"
                        )
                    else:
                        actual_dtype = (
                            getattr(self.model, "dtype", "unknown")
                            if hasattr(self.model, "dtype")
                            else "unknown"
                        )
                        print(
                            f"   模型已移动到 {target_device}，dtype: {actual_dtype}"
                        )

                # 加载处理器（如果可用则从 local_dir 加载以避免重新获取）
                processor_kwargs = {"trust_remote_code": True, "use_fast": False}
                if effective_cache_dir:
                    processor_kwargs["cache_dir"] = effective_cache_dir

                processor_source = local_dir if "local_dir" in locals() else model_name
                self.processor = AutoProcessor.from_pretrained(
                    processor_source, **processor_kwargs
                )

                self.current_model_name = model_name

                if not be_quiet:
                    print(f"✅ Sa2VA 模型成功加载: {model_name}")

            except ImportError as e:
                error_str = str(e)
                if "flash_attn" in error_str:
                    print(f"❌ Flash Attention 依赖缺失: {e}")
                    print("💡 正在重试模型加载而不使用 Flash Attention...")
                    # 移除 flash_attn 要求并重试
                    model_kwargs.pop("use_flash_attn", None)
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_name, **model_kwargs
                        ).eval()
                        print("✅ 模型已成功加载，未使用 Flash Attention")
                    except Exception as retry_e:
                        print(
                            f"❌ 即使不使用 Flash Attention，模型加载也失败: {retry_e}"
                        )
                        return False
                else:
                    print(f"❌ Sa2VA 模型缺少依赖: {e}")
                    print("💡 尝试安装: pip install transformers>=4.57.0")
                    return False
            except Exception as e:
                print(f"❌ 加载 Sa2VA 模型 {model_name} 时出错: {e}")
                if "qwen_vl_utils" in str(e).lower():
                    print("💡 缺少 qwen_vl_utils 依赖")
                    print("   安装方法: pip install qwen_vl_utils")
                elif "qwen3_vl" in str(e).lower():
                    print(
                        "💡 此错误表明您的 transformers 版本不支持 Qwen3-VL"
                    )
                    print("   尝试升级: pip install transformers>=4.57.0")
                elif "trust_remote_code" in str(e).lower():
                    print(
                        "💡 此模型需要 trust_remote_code=True（默认已启用）"
                    )
                return False

        return True

    def process_single_image(
        self, image, text_prompt, segmentation_mode=False, segmentation_prompt=""
    ):
        """使用 Sa2VA 模型处理单个图像"""
        try:
            # 如果启用了分割模式则使用分割提示
            prompt = (
                segmentation_prompt
                if segmentation_mode and segmentation_prompt
                else text_prompt
            )

            # 确保图像是 PIL Image
            if isinstance(image, str) and os.path.exists(image):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                # 尝试将 tensor/array 转换为 PIL Image
                if hasattr(image, "numpy"):
                    image_np = image.numpy()
                elif isinstance(image, np.ndarray):
                    image_np = image
                else:
                    print(f"⚠️ 不支持的图像类型: {type(image)}")
                    return "错误: 不支持的图像格式", []

                # 将 numpy 数组转换为 PIL Image
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)
                if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3]:
                    image_np = np.transpose(image_np, (1, 2, 0))
                image = Image.fromarray(image_np)

            # 为 Sa2VA 准备输入字典
            input_dict = {
                "image": image,
                "text": f"<image>{prompt}",
                "past_text": "",
                "mask_prompts": None,
                "processor": self.processor,
            }

            # 通过 Sa2VA 模型进行前向传播
            with torch.no_grad():
                return_dict = self.model.predict_forward(**input_dict)

            # 提取文本输出
            text_output = return_dict.get("prediction", "")

            # 提取分割掩码（如果可用）
            masks = return_dict.get("prediction_masks", [])

            return text_output, masks

        except Exception as e:
            error_msg = f"处理图像时出错: {e}"
            print(f"❌ {error_msg}")
            return error_msg, []

    def convert_masks_to_comfyui(
        self,
        masks,
        input_height,
        input_width,
        output_format="both",
        normalize=True,
        threshold=0.5,
        apply_threshold=False,
        batchify_mask=True,
    ):
        """
        将 Sa2VA numpy 掩码转换为 ComfyUI 格式
        """
        try:
            # 优雅地处理 None 输入
            if masks is None or len(masks) == 0:
                if not be_quiet:
                    print("⚠️ 没有要转换的掩码，创建空白掩码")
                # 返回大小与输入匹配的空白掩码
                empty_mask = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )
                empty_image = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )
                return empty_mask, empty_image

            comfyui_masks = []
            image_tensors = []

            for i, mask in enumerate(masks):
                if mask is None:
                    continue

                try:
                    # 如果还不是 numpy 数组则转换为 numpy 数组
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.detach().cpu().numpy()
                    elif isinstance(mask, np.ndarray):
                        mask_np = mask.copy()
                    elif isinstance(mask, (list, tuple)):
                        mask_np = np.array(mask)
                    else:
                        continue

                    # 处理不同的掩码维度
                    if len(mask_np.shape) == 4:  # (batch, channel, height, width)
                        mask_np = mask_np[0, 0]
                    elif len(mask_np.shape) == 3:
                        if mask_np.shape[0] == 1:  # (1, height, width)
                            mask_np = mask_np[0]
                        elif mask_np.shape[2] == 1:  # (height, width, 1)
                            mask_np = mask_np[:, :, 0]
                        elif (
                            mask_np.shape[0] < mask_np.shape[1]
                            and mask_np.shape[0] < mask_np.shape[2]
                        ):
                            mask_np = mask_np[0]
                        else:
                            mask_np = mask_np[:, :, 0]

                    # 确保我们有 2D 掩码
                    if len(mask_np.shape) != 2:
                        continue

                    # 处理空或无效的掩码
                    if mask_np.size == 0:
                        continue

                    # 转换为 float 进行处理
                    if mask_np.dtype == bool:
                        mask_np = mask_np.astype(np.float32)
                    elif not np.issubdtype(mask_np.dtype, np.floating):
                        mask_np = mask_np.astype(np.float32)

                    # 处理 NaN 和无限值
                    if np.any(np.isnan(mask_np)) or np.any(np.isinf(mask_np)):
                        mask_np = np.nan_to_num(
                            mask_np, nan=0.0, posinf=1.0, neginf=0.0
                        )

                    # 如果请求则归一化到 0-1 范围
                    if normalize:
                        mask_min, mask_max = mask_np.min(), mask_np.max()
                        if mask_max > mask_min:
                            mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                        else:
                            mask_np = (
                                np.ones_like(mask_np)
                                if mask_min > 0
                                else np.zeros_like(mask_np)
                            )

                    # 如果请求则应用阈值
                    if apply_threshold:
                        mask_np = (mask_np > threshold).astype(np.float32)

                    # 转换为 ComfyUI 掩码格式（torch tensor）
                    if output_format in ["comfyui_mask", "both"]:
                        comfyui_mask = torch.from_numpy(mask_np).float()
                        while comfyui_mask.ndim > 2:
                            comfyui_mask = comfyui_mask.squeeze(0)
                        if comfyui_mask.ndim == 2:
                            comfyui_masks.append(comfyui_mask)

                    # 转换为 ComfyUI IMAGE tensor [H, W, 3] 每个掩码（归一化 0-1）
                    rgb_np = np.stack([mask_np, mask_np, mask_np], axis=-1)
                    rgb_np = np.clip(rgb_np, 0.0, 1.0).astype(np.float32)
                    image_tensors.append(torch.from_numpy(rgb_np))

                except Exception as e:
                    if not be_quiet:
                        print(f"❌ 处理掩码 {i} 时出错: {e}")
                    continue

            # 处理没有成功处理掩码的情况
            if not comfyui_masks:
                empty_mask = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )
                empty_image = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )
                return empty_mask, empty_image

            # 构建最终的 MASK tensor: [B, H, W] 如果 batchify_mask 否则 [H, W]
            final_comfyui_masks = None
            if comfyui_masks:
                try:
                    masks_2d = []
                    for m in comfyui_masks:
                        while m.ndim > 2:
                            m = m.squeeze(0)
                        if m.ndim == 2:
                            masks_2d.append(m.float())

                    if not masks_2d:
                        final_comfyui_masks = (
                            torch.zeros(
                                (1, input_height, input_width), dtype=torch.float32
                            )
                            if batchify_mask
                            else torch.zeros(
                                (input_height, input_width), dtype=torch.float32
                            )
                        )
                    else:
                        if batchify_mask:
                            first_hw = masks_2d[0].shape
                            aligned = [t for t in masks_2d if t.shape == first_hw]
                            final_comfyui_masks = (
                                torch.stack(aligned, dim=0).float()
                                if aligned
                                else torch.zeros(
                                    (1, input_height, input_width), dtype=torch.float32
                                )
                            )
                        else:
                            final_comfyui_masks = masks_2d[0]
                except Exception as e:
                    if not be_quiet:
                        print(f"⚠️ 处理 ComfyUI 掩码时出错: {e}")
                    final_comfyui_masks = (
                        torch.zeros((1, input_height, input_width), dtype=torch.float32)
                        if batchify_mask
                        else torch.zeros(
                            (input_height, input_width), dtype=torch.float32
                        )
                    )
            else:
                final_comfyui_masks = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )

            # 构建 IMAGE batch tensor [B, H, W, 3]
            if image_tensors:
                try:
                    first_hw = image_tensors[0].shape[:2]
                    aligned = [t for t in image_tensors if t.shape[:2] == first_hw]
                    if not aligned:
                        final_image_tensor = torch.zeros(
                            (1, input_height, input_width, 3), dtype=torch.float32
                        )
                    else:
                        final_image_tensor = torch.stack(aligned, dim=0).float()
                except Exception as e:
                    if not be_quiet:
                        print(f"⚠️ 堆叠 IMAGE tensor 时出错: {e}")
                    final_image_tensor = torch.zeros(
                        (1, input_height, input_width, 3), dtype=torch.float32
                    )
            else:
                final_image_tensor = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )

            return final_comfyui_masks, final_image_tensor

        except Exception as e:
            if not be_quiet:
                print(f"❌ 转换掩码时出错: {e}")
            empty_mask = (
                torch.zeros((1, input_height, input_width), dtype=torch.float32)
                if batchify_mask
                else torch.zeros((input_height, input_width), dtype=torch.float32)
            )
            empty_image = torch.zeros(
                (1, input_height, input_width, 3), dtype=torch.float32
            )
            return empty_mask, empty_image

    def process_with_sa2va(
        self,
        model_name,
        image,
        mask_threshold,
        segmentation_prompt,
        use_8bit_quantization,
        use_flash_attn,
    ):
        """Sa2VA 模型的主处理函数"""
        # 为隐藏参数设置默认值
        text_prompt = "请描述这张图片。"
        segmentation_mode = True
        video_mode = False
        max_frames = 5
        dtype = "auto"
        use_inference_mode = True
        use_autocast = True
        autocast_dtype = "bfloat16"
        free_gpu_after = True
        unload_model_after = False
        offload_to_cpu = False
        offload_input_to_cpu = True
        cache_dir = ""
        output_mask_format = "both"
        normalize_masks = True
        apply_mask_threshold = False
        batchify_mask = True

        try:
            # 如果尚未加载则加载模型
            model_loaded = self.load_model(
                model_name, use_flash_attn, dtype, cache_dir, use_8bit_quantization
            )
            if not model_loaded:
                error_msg = f"加载 Sa2VA 模型失败: {model_name}。请检查控制台以获取详细信息。"
                print(f"❌ {error_msg}")
                # 返回有效结构以防止下游错误
                return ([error_msg], torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64, 3), dtype=torch.float32))

            # 验证输入
            if image is None:
                error_msg = "未提供图像"
                print(f"⚠️ {error_msg}")
                return ([error_msg], torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64, 3), dtype=torch.float32))

            if not be_quiet:
                print(f"🔄 正在处理图像 | 分割模式: {segmentation_mode}")

            # 将 ComfyUI 图像 tensor 转换为 PIL Image
            if hasattr(image, "shape") and len(image.shape) == 4:
                # ComfyUI 图像格式: (batch, height, width, channels)
                img_t = image[0]
            elif hasattr(image, "shape") and len(image.shape) == 3:
                # 单张图像: (height, width, channels)
                img_t = image
            else:
                error_msg = f"不支持的图像格式: {type(image)}"
                print(f"❌ {error_msg}")
                return ([error_msg], torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64, 3), dtype=torch.float32))

            # 将图像 tensor 卸载到 CPU 并立即释放 GPU 内存
            if isinstance(img_t, torch.Tensor):
                try:
                    if offload_input_to_cpu and img_t.is_cuda:
                        img_cpu = img_t.detach().to("cpu")
                        del img_t
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                        img_t = img_cpu
                    else:
                        img_t = img_t.detach().cpu()
                except Exception:
                    # 回退到普通的 .cpu()
                    img_t = img_t.cpu()
                image_np = img_t.numpy()
                # 帮助 GC 立即执行
                del img_t
            else:
                error_msg = f"不支持的图像 tensor 类型: {type(image)}"
                print(f"❌ {error_msg}")
                return ([error_msg], torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64, 3), dtype=torch.float32))

            # 转换为 PIL Image
            if image_np.dtype != "uint8":
                image_np = (image_np * 255).astype("uint8")

            pil_image = Image.fromarray(image_np)

            # 使用内存友好的上下文处理单张图像
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if use_autocast and device == "cuda":
                if autocast_dtype == "float16":
                    _amp_dtype = torch.float16
                elif autocast_dtype == "bfloat16" or autocast_dtype == "auto":
                    _amp_dtype = torch.bfloat16
                else:
                    _amp_dtype = torch.bfloat16
                autocast_ctx = torch.cuda.amp.autocast(dtype=_amp_dtype)
            else:
                autocast_ctx = nullcontext()

            inference_ctx = (
                torch.inference_mode() if use_inference_mode else nullcontext()
            )

            with inference_ctx:
                with autocast_ctx:
                    text_output, masks = self.process_single_image(
                        pil_image, text_prompt, segmentation_mode, segmentation_prompt
                    )

            text_outputs = [text_output]
            all_masks = masks if masks else []

            # 获取用于掩码大小的输入维度
            h, w = int(image_np.shape[0]), int(image_np.shape[1])

            # 在分割模式下始终确保我们有掩码
            if segmentation_mode and len(all_masks) == 0:
                blank_mask = np.zeros((h, w), dtype=np.float32)
                all_masks = [blank_mask]

            # 将掩码转换为 ComfyUI 格式
            comfyui_masks, mask_images = self.convert_masks_to_comfyui(
                all_masks,
                h,
                w,
                output_mask_format,
                normalize_masks,
                mask_threshold,
                apply_mask_threshold,
                batchify_mask,
            )

            if not be_quiet:
                print(
                    f"✅ Sa2VA 处理完成: {len(text_outputs)} 个文本输出，{len(all_masks)} 个掩码"
                )
                if dtype != "auto":
                    print(f"   注意: 模型已从原生精度转换为 {dtype}")
                if comfyui_masks is not None:
                    print(f"   ComfyUI 掩码形状: {comfyui_masks.shape}")
                if mask_images is not None:
                    print(f"   IMAGE tensor 形状: {mask_images.shape}")

            # 确保我们始终返回有效列表，永远不返回可能导致索引错误的空列表
            if not text_outputs:
                text_outputs = ["处理完成但未生成文本"]

            # 确保 text_outputs 永远不为空以防止下游索引错误
            if len(text_outputs) == 0:
                text_outputs = ["错误: 未生成输出"]

            # 运行后内存管理
            try:
                if torch.cuda.is_available():
                    if free_gpu_after:
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                    if unload_model_after:
                        if offload_to_cpu and self.model is not None:
                            try:
                                # 确保模型及其所有参数正确移动到 CPU
                                self.model = self.model.cpu()
                                # 强制所有参数到 CPU
                                for param in self.model.parameters():
                                    param.data = param.data.cpu()
                                for buffer in self.model.buffers():
                                    buffer.data = buffer.data.cpu()
                                if not be_quiet:
                                    print("   模型已卸载到 CPU")
                            except Exception as _e:
                                if not be_quiet:
                                    print(f"   卸载到 CPU 失败: {_e}")
                                # 如果 CPU 卸载失败则回退到完全卸载
                                try:
                                    del self.model
                                except:
                                    pass
                                self.model = None
                                self.processor = None
                                self.current_model_name = None
                        else:
                            # 完全卸载模型
                            try:
                                del self.model
                            except:
                                pass
                            try:
                                del self.processor
                            except:
                                pass
                            self.model = None
                            self.processor = None
                            self.current_model_name = None
                            if not be_quiet:
                                print("   模型已卸载")
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                # 始终收集 Python GC
                gc.collect()
            except Exception as _e:
                if not be_quiet:
                    print(f"⚠️ 内存管理步骤遇到问题: {_e}")

            return (text_outputs, comfyui_masks, mask_images)

        except Exception as e:
            error_msg = f"Sa2VA 处理失败: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()

            # 始终返回有效结构以防止下游崩溃
            # 获取回退维度
            try:
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    if len(image.shape) == 4:
                        fb_h, fb_w = image.shape[1], image.shape[2]
                    elif len(image.shape) == 3:
                        fb_h, fb_w = image.shape[0], image.shape[1]
                    else:
                        fb_h, fb_w = 64, 64
                else:
                    fb_h, fb_w = 64, 64
            except:
                fb_h, fb_w = 64, 64

            empty_mask = (
                torch.zeros((1, fb_h, fb_w), dtype=torch.float32)
                if batchify_mask
                else torch.zeros((fb_h, fb_w), dtype=torch.float32)
            )
            empty_image = torch.zeros((1, fb_h, fb_w, 3), dtype=torch.float32)
            return ([f"错误: {error_msg}"], empty_mask, empty_image)

