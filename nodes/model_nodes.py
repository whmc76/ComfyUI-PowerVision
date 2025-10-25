"""
PowerVision 模型相关节点

包含模型加载、管理相关的节点
"""

import os
import torch
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# 尝试导入 Qwen3VLForConditionalGeneration，如果不可用则使用占位符
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    print("PowerVision: Qwen3VLForConditionalGeneration 不可用，将使用 Qwen2.5-VL 模型")
    QWEN3_AVAILABLE = False
    # 创建占位符类
    class Qwen3VLForConditionalGeneration:
        pass

import folder_paths
import comfy.model_management


@dataclass
class QwenModel:
    """Qwen模型包装类"""
    model: Any
    processor: Any
    device: str
    model_type: str = "qwen3"  # "qwen3" 或 "qwen2.5"


class PowerVisionQwen3VQA:
    """PowerVision Qwen3-VL 视觉问答节点"""
    
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                    ],
                    {"default": "Qwen3-VL-4B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                ),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "inference"
    CATEGORY = "PowerVision/Load Model"

    def inference(
        self,
        text: str,
        model: str,
        keep_model_loaded: bool,
        temperature: float,
        max_new_tokens: int,
        min_pixels: int,
        max_pixels: int,
        seed: int,
        quantization: str,
        source_path: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
        attention: str = "eager",
    ) -> Tuple[str]:
        """执行视觉问答推理"""
        if seed != -1:
            torch.manual_seed(seed)
        
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "Qwen", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )

        if self.model is None:
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

        temp_path = None
        if image is not None:
            from torchvision.transforms import ToPILImage
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = os.path.join(folder_paths.temp_directory, f"temp_image_{seed}.png")
            pil_image.save(temp_path)

        with torch.no_grad():
            if source_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 尝试导入 qwen_vl_utils，如果不存在则提供备用实现
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                def process_vision_info(messages):
                    image_inputs = []
                    video_inputs = []
                    for message in messages:
                        if isinstance(message.get("content"), list):
                            for content in message["content"]:
                                if content.get("type") == "image":
                                    image_inputs.append(content.get("image"))
                                elif content.get("type") == "video":
                                    video_inputs.append(content.get("video"))
                    return image_inputs, video_inputs
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            return (result,)


class PowerVisionQwen3VQAWithModel:
    """PowerVision Qwen3-VL 视觉问答节点（使用预加载模型）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 256000, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "inference"
    CATEGORY = "PowerVision/Image Caption"

    def inference(
        self,
        qwen_model: QwenModel,
        text: str,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        source_path: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        """执行视觉问答推理（使用预加载模型）"""
        if seed != -1:
            torch.manual_seed(seed)
        
        model = qwen_model.model
        processor = qwen_model.processor
        device = qwen_model.device
        
        # 处理设备字符串，将 "auto" 转换为具体设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

        temp_path = None
        if image is not None:
            from torchvision.transforms import ToPILImage
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = os.path.join(folder_paths.temp_directory, f"temp_image_{seed}.png")
            pil_image.save(temp_path)

        with torch.no_grad():
            # 优先使用直接传入的图片，然后是 source_path，最后是纯文本
            if temp_path:
                # 使用直接传入的图片
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif source_path:
                # 使用 source_path 作为备选
                if isinstance(source_path, list):
                    source_path = source_path[0] if source_path else None
                
                if source_path:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{source_path}"},
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                else:
                    # 如果 source_path 为空，回退到纯文本模式
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                            ],
                        }
                    ]
            else:
                # 纯文本模式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 尝试导入 qwen_vl_utils，如果不存在则提供备用实现
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                def process_vision_info(messages):
                    image_inputs = []
                    video_inputs = []
                    for message in messages:
                        if isinstance(message.get("content"), list):
                            for content in message["content"]:
                                if content.get("type") == "image":
                                    image_path = content.get("image")
                                    # 移除 file:// 协议前缀，直接使用文件路径
                                    if image_path and image_path.startswith("file://"):
                                        image_path = image_path[7:]  # 移除 "file://" 前缀
                                    image_inputs.append(image_path)
                                elif content.get("type") == "video":
                                    video_inputs.append(content.get("video"))
                    return image_inputs, video_inputs
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 根据模型类型选择不同的调用方式
            if qwen_model.model_type == "qwen3":
                # Qwen3-VL 支持 images 和 videos 参数
                processor_kwargs = {
                    "text": [text],
                    "padding": True,
                    "return_tensors": "pt",
                }
                
                # 只有当有图片时才添加 images 参数
                if image_inputs:
                    processor_kwargs["images"] = image_inputs
                
                # 只有当有视频时才添加 videos 参数
                if video_inputs:
                    processor_kwargs["videos"] = video_inputs
                
                inputs = processor(**processor_kwargs)
            else:
                # Qwen2.5-VL 只支持 images 参数，不支持 videos
                processor_kwargs = {
                    "text": [text],
                    "padding": True,
                    "return_tensors": "pt",
                }
                
                # 只有当有图片时才添加 images 参数
                if image_inputs:
                    processor_kwargs["images"] = image_inputs
                
                inputs = processor(**processor_kwargs)
            inputs = inputs.to(device)
            
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            return (result,)


class PowerVisionQwenModelLoader:
    """PowerVision Qwen模型加载器节点"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    # Qwen3-VL 模型
                    "Qwen/Qwen3-VL-2B-Instruct",
                    "Qwen/Qwen3-VL-2B-Thinking",
                    "Qwen/Qwen3-VL-4B-Instruct",
                    "Qwen/Qwen3-VL-4B-Thinking",
                    "Qwen/Qwen3-VL-8B-Instruct",
                    "Qwen/Qwen3-VL-8B-Thinking",
                    # Qwen2.5-VL 模型
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                ], {"default": "Qwen/Qwen3-VL-4B-Instruct"}),
                "device": ([
                    "auto",
                    "cuda:0",
                    "cuda:1",
                    "cpu",
                ], ),
                "precision": ([
                    "INT4",
                    "INT8",
                    "FP8",
                    "BF16",
                    "FP16",
                    "FP32",
                ], ),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], ),
                "download_source": ([
                    "auto",
                    "huggingface",
                    "modelscope",
                ], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "PowerVision/Load Model"

    def load(self, model_name: str, device: str, precision: str, attention: str, download_source: str = "auto") -> Tuple[QwenModel]:
        """加载Qwen模型"""
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        
        def is_model_complete(model_path):
            """检查模型文件夹是否包含必要的模型文件"""
            if not os.path.exists(model_path):
                return False
            
            # 检查是否有 .safetensors 文件（模型权重文件）
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            if not safetensors_files:
                return False
            
            # 检查是否有 config.json（配置文件）
            config_file = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_file):
                return False
                
            return True
        
        def find_alternative_model_folder(model_name):
            """查找替代模型文件夹（用于回退）"""
            # 尝试不同的文件夹命名格式
            variants = [
                model_name.split("/")[-1],     # Qwen3-VL-4B-Instruct
                model_name.replace("/", "_"),  # Qwen_Qwen3-VL-4B-Instruct
            ]
            
            for variant in variants:
                test_path = os.path.join(folder_paths.models_dir, "Qwen", variant)
                if is_model_complete(test_path):
                    print(f"PowerVision: 找到替代模型文件夹: {test_path}")
                    return test_path
            return None
        
        # 优先使用不带前缀的模型文件夹（标准 Hugging Face 目录结构）
        clean_model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.split("/")[-1])
        if is_model_complete(clean_model_dir):
            print(f"PowerVision: 使用现有模型文件夹: {clean_model_dir}")
            model_dir = clean_model_dir
        else:
            # 尝试使用带前缀的模型文件夹（兼容旧版本）
            if is_model_complete(model_dir):
                print(f"PowerVision: 使用现有模型: {model_dir}")
            else:
                # 尝试查找替代模型文件夹
                alt_dir = find_alternative_model_folder(model_name)
                if alt_dir:
                    print(f"PowerVision: 使用替代模型文件夹: {alt_dir}")
                    model_dir = alt_dir
                else:
                    print(f"PowerVision: 模型 {model_name} 不存在或不完整，开始下载...")
                    print(f"PowerVision: 下载源设置: {download_source}")
                    
                    download_success = False
                    
                    if download_source == "huggingface":
                        # 仅使用 Hugging Face
                        try:
                            print(f"PowerVision: 从 Hugging Face 下载 {model_name}...")
                            print(f"PowerVision: 开始下载模型文件，请稍候...")
                            snapshot_download(
                                repo_id=model_name,
                                local_dir=model_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True,
                            )
                            download_success = True
                            print(f"PowerVision: 从 Hugging Face 下载成功")
                        except Exception as e:
                            print(f"PowerVision: Hugging Face 下载失败: {e}")
                            
                    elif download_source == "modelscope":
                        # 仅使用 ModelScope
                        try:
                            print(f"PowerVision: 从 ModelScope 下载 {model_name}...")
                            
                            # 方法1: 尝试使用 ModelScope 库
                            try:
                                from modelscope import snapshot_download as ms_snapshot_download
                                print(f"PowerVision: 开始下载模型文件，请稍候...")
                                ms_snapshot_download(
                                    model_id=model_name,
                                    cache_dir=model_dir,
                                    local_dir=model_dir,
                                    local_dir_use_symlinks=False,
                                )
                                download_success = True
                                print(f"PowerVision: 使用 ModelScope 库下载成功")
                            except ImportError:
                                print(f"PowerVision: ModelScope 库未安装，尝试使用 git clone...")
                                
                                # 方法2: 使用 git clone 从 ModelScope
                                import subprocess
                                import shutil
                                import time
                                
                                # 构建 ModelScope git URL
                                modelscope_url = f"https://www.modelscope.cn/{model_name}.git"
                                temp_dir = model_dir + "_temp"
                                
                                # 清理可能存在的临时目录
                                if os.path.exists(temp_dir):
                                    print(f"PowerVision: 清理临时目录: {temp_dir}")
                                    shutil.rmtree(temp_dir, ignore_errors=True)
                                
                                print(f"PowerVision: 使用 git clone 从 ModelScope: {modelscope_url}")
                                
                                # 执行 git clone 并显示进度
                                print(f"PowerVision: 开始下载模型文件...")
                                result = subprocess.run([
                                    "git", "clone", "--progress", modelscope_url, temp_dir
                                ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
                                
                                if result.returncode == 0:
                                    # 移动文件到目标目录
                                    if os.path.exists(temp_dir):
                                        if os.path.exists(model_dir):
                                            shutil.rmtree(model_dir)
                                        shutil.move(temp_dir, model_dir)
                                        download_success = True
                                        print(f"PowerVision: 使用 git clone 从 ModelScope 下载成功")
                                    else:
                                        print(f"PowerVision: git clone 完成但目录不存在")
                                else:
                                    print(f"PowerVision: git clone 失败: {result.stderr}")
                                    # 清理失败的临时目录
                                    if os.path.exists(temp_dir):
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                    
                        except Exception as ms_e:
                            print(f"PowerVision: ModelScope 下载失败: {ms_e}")
                            
                    else:  # download_source == "auto"
                        # 自动选择：先尝试 Hugging Face，失败后尝试 ModelScope
                        try:
                            print(f"PowerVision: 尝试从 Hugging Face 下载 {model_name}...")
                            print(f"PowerVision: 开始下载模型文件，请稍候...")
                            snapshot_download(
                                repo_id=model_name,
                                local_dir=model_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True,
                            )
                            download_success = True
                            print(f"PowerVision: 从 Hugging Face 下载成功")
                        except Exception as e:
                            print(f"PowerVision: Hugging Face 下载失败: {e}")
                            
                            # 尝试从 ModelScope 下载
                            try:
                                print(f"PowerVision: 尝试从 ModelScope 下载 {model_name}...")
                                
                                # 方法1: 尝试使用 ModelScope 库
                                try:
                                    from modelscope import snapshot_download as ms_snapshot_download
                                    print(f"PowerVision: 开始下载模型文件，请稍候...")
                                    ms_snapshot_download(
                                        model_id=model_name,
                                        cache_dir=model_dir,
                                        local_dir=model_dir,
                                        local_dir_use_symlinks=False,
                                    )
                                    download_success = True
                                    print(f"PowerVision: 使用 ModelScope 库下载成功")
                                except ImportError:
                                    print(f"PowerVision: ModelScope 库未安装，尝试使用 git clone...")
                                    
                                    # 方法2: 使用 git clone 从 ModelScope
                                    import subprocess
                                    import shutil
                                    import time
                                    
                                    # 构建 ModelScope git URL
                                    modelscope_url = f"https://www.modelscope.cn/{model_name}.git"
                                    temp_dir = model_dir + "_temp"
                                    
                                    # 清理可能存在的临时目录
                                    if os.path.exists(temp_dir):
                                        print(f"PowerVision: 清理临时目录: {temp_dir}")
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                    
                                    print(f"PowerVision: 使用 git clone 从 ModelScope: {modelscope_url}")
                                    
                                    # 执行 git clone 并显示进度
                                    print(f"PowerVision: 开始下载模型文件...")
                                    result = subprocess.run([
                                        "git", "clone", "--progress", modelscope_url, temp_dir
                                    ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
                                    
                                    if result.returncode == 0:
                                        # 移动文件到目标目录
                                        if os.path.exists(temp_dir):
                                            if os.path.exists(model_dir):
                                                shutil.rmtree(model_dir)
                                            shutil.move(temp_dir, model_dir)
                                            download_success = True
                                            print(f"PowerVision: 使用 git clone 从 ModelScope 下载成功")
                                        else:
                                            print(f"PowerVision: git clone 完成但目录不存在")
                                    else:
                                        print(f"PowerVision: git clone 失败: {result.stderr}")
                                        # 清理失败的临时目录
                                        if os.path.exists(temp_dir):
                                            shutil.rmtree(temp_dir, ignore_errors=True)
                                        
                            except Exception as ms_e:
                                print(f"PowerVision: ModelScope 下载失败: {ms_e}")
                    
                    if not download_success:
                        raise Exception(f"PowerVision: 下载失败，无法获取模型 {model_name}")
        
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        # 根据模型名称选择合适的模型类
        # 检查是否为 FP8 模型
        is_fp8_model = "FP8" in model_name
        
        precision = precision.upper()
        dtype_map = {
            "BF16": torch.bfloat16,
            "FP16": torch.float16,
            "FP32": torch.float32,
            "FP8": torch.float16,  # FP8 使用 FP16 作为基础类型
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)
        quant_config = None
        
        # FP8 模型不支持 INT4/INT8 量化配置，因为模型已经预量化
        if not is_fp8_model:
            if precision == "INT4":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_use_double_quant=True
                )
            elif precision == "INT8":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif is_fp8_model and precision in ["INT4", "INT8"]:
            print(f"PowerVision: FP8 模型不支持 INT4/INT8 精度设置，将使用 FP8 模型的默认精度")

        attn_impl = attention
        if precision == "FP32" and attn_impl == "flash_attention_2":
            attn_impl = "sdpa"
        
        if "Qwen3-VL" in model_name:
            if not QWEN3_AVAILABLE:
                raise Exception(f"PowerVision: Qwen3-VL 模型 {model_name} 不可用。请升级 transformers 库到 4.51.0 或更高版本，或手动选择 Qwen2.5-VL 模型。")
            model_class = Qwen3VLForConditionalGeneration
            model_type = "qwen3"
            if is_fp8_model:
                print(f"PowerVision: 使用 Qwen3-VL FP8 模型: {model_name}")
        else:
            model_class = Qwen2_5_VLForConditionalGeneration
            model_type = "qwen2.5"

        try:
            model = model_class.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except OSError:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=True,
            )
            model = model_class.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception:
            raise
        
        # 对于 Qwen2.5-VL，使用 slow processor 以避免检测结果差异
        if model_type == "qwen2.5":
            processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
            print("PowerVision: Qwen2.5-VL 使用 slow processor (use_fast=False)")
        else:
            processor = AutoProcessor.from_pretrained(model_dir)
        
        return (QwenModel(model=model, processor=processor, device=device, model_type=model_type),)

