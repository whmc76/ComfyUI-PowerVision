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

# 直接导入Qwen3VL，如果失败则使用Qwen2.5-VL作为后备
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("PowerVision: Qwen3-VL 支持已启用")
except ImportError:
    print("PowerVision: Qwen3-VL 不可用，将使用 Qwen2.5-VL 模型")
    # 创建兼容类，自动使用Qwen2.5-VL
    class Qwen3VLForConditionalGeneration:
        def __init__(self, *args, **kwargs):
            # 直接使用Qwen2.5-VL作为后备
            self._model = Qwen2_5_VLForConditionalGeneration(*args, **kwargs)
        
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            # 自动降级到Qwen2.5-VL
            print("PowerVision: Qwen3-VL 不可用，自动使用 Qwen2.5-VL 模型")
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(*args, **kwargs)
        
        def __getattr__(self, name):
            # 将所有方法调用转发到实际的Qwen2.5-VL模型
            return getattr(self._model, name)

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
                "min_pixels": ("INT", {
                    "default": 256 * 28 * 28,
                    "min": 4 * 28 * 28,
                    "max": 16384 * 28 * 28,
                    "step": 28 * 28,
                    "tooltip": "最小像素数，用于控制视觉令牌数量。视频处理时建议使用较小值以减少内存使用。"
                }),
                "max_pixels": ("INT", {
                    "default": 1280 * 28 * 28,
                    "min": 4 * 28 * 28,
                    "max": 16384 * 28 * 28,
                    "step": 28 * 28,
                    "tooltip": "最大像素数，用于控制视觉令牌数量。视频处理时建议使用较小值以减少内存使用。"
                }),
            },
            "optional": {
                "source_path": ("PATH",), 
                "image": ("IMAGE",),
                "video": ("VIDEO",)
            },
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
        min_pixels: int,
        max_pixels: int,
        source_path: Optional[str] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[str] = None,
    ) -> Tuple[str]:
        """执行视觉问答推理（使用预加载模型）"""
        if seed != -1:
            torch.manual_seed(seed)
        
        # 检测视频处理并提供内存优化建议
        if video is not None or (source_path and source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))):
            print("PowerVision: 检测到视频处理，应用内存优化措施")
            # 对于视频处理，建议使用更保守的像素设置
            if max_pixels > 640 * 28 * 28:  # 如果超过建议值
                print(f"PowerVision: 建议将 max_pixels 降低到 640*28*28 以下以减少内存使用")
            if min_pixels > 256 * 28 * 28:  # 如果超过建议值
                print(f"PowerVision: 建议将 min_pixels 降低到 256*28*28 以下以减少内存使用")
        
        model = qwen_model.model
        device = qwen_model.device
        
        # 动态创建处理器，使用像素控制参数来优化内存使用
        # 这对于视频处理特别重要，可以减少内存消耗
        try:
            # 尝试使用像素控制参数创建处理器
            processor = AutoProcessor.from_pretrained(
                qwen_model.model.config._name_or_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            print(f"PowerVision: 使用像素控制参数创建处理器 (min_pixels={min_pixels}, max_pixels={max_pixels})")
        except Exception as e:
            print(f"PowerVision: 像素控制参数创建失败，使用默认处理器: {e}")
            # 如果失败，使用预加载的处理器
            processor = qwen_model.processor
        
        # 处理设备字符串，将 "auto" 转换为具体设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

        # 处理图像输入
        temp_image_path = None
        if image is not None:
            from torchvision.transforms import ToPILImage
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_image_path = os.path.join(folder_paths.temp_directory, f"temp_image_{seed}.png")
            pil_image.save(temp_image_path)

        # 处理视频输入
        temp_video_path = None
        if video is not None:
            # 如果 video 是字符串路径，直接使用
            if isinstance(video, str):
                temp_video_path = video
            # 如果 video 是 VideoFromFile 对象，提取路径
            elif hasattr(video, 'get_stream_source'):
                try:
                    temp_video_path = video.get_stream_source()
                except Exception:
                    temp_video_path = None
            elif hasattr(video, 'file_path'):
                temp_video_path = video.file_path
            elif hasattr(video, 'path'):
                temp_video_path = video.path
            elif hasattr(video, 'filename'):
                temp_video_path = video.filename
            elif hasattr(video, 'name'):
                temp_video_path = video.name
            else:
                # 尝试获取所有属性值
                for attr in dir(video):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(video, attr)
                            if isinstance(value, str) and ('.mp4' in value or '.avi' in value or '.mov' in value):
                                temp_video_path = value
                                break
                        except:
                            pass
                
                # 如果仍然没有找到路径，尝试转换为字符串
                if not temp_video_path:
                    temp_video_path = str(video)

        with torch.no_grad():
            # 构建消息，参考项目的实现方式
            if source_path:
                # 使用 source_path，让 process_vision_info 自动识别文件类型
                if isinstance(source_path, list):
                    # source_path 已经是列表，直接使用
                    messages = [
                        {
                            "role": "system",
                            "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                        },
                        {
                            "role": "user",
                            "content": source_path + [
                                {"type": "text", "text": text},
                            ],
                        },
                    ]
                else:
                    # source_path 是字符串，需要根据文件类型处理
                    if source_path:
                        # 检查文件类型
                        file_ext = os.path.splitext(source_path)[1].lower()
                        
                        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                            # 视频文件
                            messages = [
                                {
                                    "role": "system",
                                    "content": "You are QwenVL, you are a helpful assistant expert in turning videos into words.",
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "video", "video": source_path},  # 直接使用路径，不加 file:// 前缀
                                        {"type": "text", "text": text},
                                    ],
                                },
                            ]
                        else:
                            # 图像文件或其他类型
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
            elif temp_image_path:
                # 使用直接传入的图像
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_image_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_video_path:
                # 使用直接传入的视频
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning videos into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": temp_video_path},  # 直接使用路径，不加 file:// 前缀
                            {"type": "text", "text": text},
                        ],
                    },
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
            
            # 使用 qwen_vl_utils 的 process_vision_info 函数
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                # 如果 qwen_vl_utils 不可用，提供备用实现
                def process_vision_info(messages):
                    image_inputs = []
                    video_inputs = []
                    
                    for message in messages:
                        content = message.get("content")
                        
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "image":
                                        image_path = item.get("image")
                                        # 移除 file:// 协议前缀，直接使用文件路径
                                        if image_path and image_path.startswith("file://"):
                                            image_path = image_path[7:]  # 移除 "file://" 前缀
                                        image_inputs.append(image_path)
                                    elif item.get("type") == "video":
                                        video_obj = item.get("video")
                                        
                                        # 处理 VideoFromFile 对象
                                        if video_obj:
                                            # 检查是否是 VideoFromFile 对象
                                            if hasattr(video_obj, 'get_stream_source'):
                                                try:
                                                    video_path = video_obj.get_stream_source()
                                                    video_inputs.append(video_path)
                                                except Exception:
                                                    pass
                                            elif hasattr(video_obj, 'file_path'):
                                                video_path = video_obj.file_path
                                                video_inputs.append(video_path)
                                            elif hasattr(video_obj, 'path'):
                                                video_path = video_obj.path
                                                video_inputs.append(video_path)
                                            elif hasattr(video_obj, 'filename'):
                                                video_path = video_obj.filename
                                                video_inputs.append(video_path)
                                            elif hasattr(video_obj, 'name'):
                                                video_path = video_obj.name
                                                video_inputs.append(video_path)
                                            elif isinstance(video_obj, str):
                                                # 如果已经是字符串路径
                                                video_inputs.append(video_obj)
                                            else:
                                                # 尝试获取所有属性值
                                                for attr in dir(video_obj):
                                                    if not attr.startswith('_'):
                                                        try:
                                                            value = getattr(video_obj, attr)
                                                            if isinstance(value, str) and ('.mp4' in value or '.avi' in value or '.mov' in value):
                                                                video_inputs.append(value)
                                                                break
                                                        except:
                                                            pass
                                                
                                                # 如果仍然没有找到路径，尝试转换为字符串
                                                if not video_inputs:
                                                    video_path = str(video_obj)
                                                    video_inputs.append(video_path)
                        elif isinstance(content, str):
                            # 处理字符串格式的 content（可能是文件路径）
                            if content and content.startswith("file://"):
                                content = content[7:]  # 移除 "file://" 前缀
                            
                            # 根据文件扩展名判断类型
                            if content:
                                file_ext = os.path.splitext(content)[1].lower()
                                if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                                    video_inputs.append(content)
                                elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']:
                                    image_inputs.append(content)
                    
                    return image_inputs, video_inputs
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 调用处理器，参考项目的实现方式
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
            inputs = inputs.to(device)
            
            # 添加内存优化措施
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 使用更保守的生成参数来减少内存使用
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True if temperature > 0 else False,
                "pad_token_id": processor.tokenizer.eos_token_id,
            }
            
            # 对于视频处理，使用更保守的参数
            if video_inputs:
                generation_kwargs.update({
                    "max_new_tokens": min(max_new_tokens, 1024),  # 限制生成长度
                    "num_beams": 1,  # 使用贪心搜索而不是束搜索
                })
                print("PowerVision: 视频处理模式，使用保守的生成参数")
            
            try:
                generated_ids = model.generate(**inputs, **generation_kwargs)
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
            except torch.cuda.OutOfMemoryError as e:
                error_msg = f"GPU内存不足: {str(e)}\n\n建议解决方案:\n"
                error_msg += "1. 降低 max_pixels 参数 (建议: 640*28*28 或更小)\n"
                error_msg += "2. 降低 min_pixels 参数 (建议: 256*28*28 或更小)\n"
                error_msg += "3. 使用量化模型 (INT4/INT8)\n"
                error_msg += "4. 减少 max_new_tokens 参数\n"
                error_msg += "5. 关闭其他占用GPU内存的程序\n"
                print(f"PowerVision: {error_msg}")
                return (error_msg,)
            except Exception as e:
                error_msg = f"推理过程中发生错误: {str(e)}"
                print(f"PowerVision: {error_msg}")
                return (error_msg,)


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
            # 直接使用Qwen3VL类，如果不可用会自动降级到Qwen2.5-VL
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

