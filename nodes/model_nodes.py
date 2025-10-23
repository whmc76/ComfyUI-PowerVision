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
                        "Qwen3-VL-4B-Instruct-FP8",
                        "Qwen3-VL-4B-Thinking-FP8",
                        "Qwen3-VL-8B-Instruct-FP8",
                        "Qwen3-VL-8B-Thinking-FP8",
                        "Qwen3-VL-4B-Instruct",
                        "Qwen3-VL-4B-Thinking",
                        "Qwen3-VL-8B-Instruct",
                        "Qwen3-VL-8B-Thinking",
                    ],
                    {"default": "Qwen3-VL-4B-Instruct-FP8"},
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
            
            # 只传递非空的输入
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
                    "BF16",
                    "FP16",
                    "FP32",
                ], ),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], ),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "PowerVision/Load Model"

    def load(self, model_name: str, device: str, precision: str, attention: str) -> Tuple[QwenModel]:
        """加载Qwen模型"""
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        
        # 首先检查带前缀的模型文件夹
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"PowerVision: 使用现有模型: {model_dir}")
        else:
            # 尝试使用不带前缀的模型文件夹
            alternative_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.split("/")[-1])
            if os.path.exists(alternative_dir) and os.listdir(alternative_dir):
                print(f"PowerVision: 使用现有模型文件夹: {alternative_dir}")
                model_dir = alternative_dir
            else:
                print(f"PowerVision: 模型 {model_name} 不存在，开始下载...")
                try:
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                except Exception as e:
                    print(f"PowerVision: 下载失败: {e}")
                    raise e
        
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        precision = precision.upper()
        dtype_map = {
            "BF16": torch.bfloat16,
            "FP16": torch.float16,
            "FP32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)
        quant_config = None
        
        if precision == "INT4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_use_double_quant=True
            )
        elif precision == "INT8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = attention
        if precision == "FP32" and attn_impl == "flash_attention_2":
            attn_impl = "sdpa"

        # 根据模型名称选择合适的模型类
        if "Qwen3-VL" in model_name and QWEN3_AVAILABLE:
            model_class = Qwen3VLForConditionalGeneration
            model_type = "qwen3"
        elif "Qwen3-VL" in model_name and not QWEN3_AVAILABLE:
            print(f"PowerVision: Qwen3-VL 模型 {model_name} 不可用，回退到 Qwen2.5-VL")
            # 将 Qwen3-VL 模型名称映射到对应的 Qwen2.5-VL 模型
            model_mapping = {
                "Qwen/Qwen3-VL-4B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen3-VL-4B-Thinking": "Qwen/Qwen2.5-VL-3B-Instruct",  # 没有对应的 Thinking 版本
                "Qwen/Qwen3-VL-8B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen3-VL-8B-Thinking": "Qwen/Qwen2.5-VL-7B-Instruct",  # 没有对应的 Thinking 版本
            }
            model_name = model_mapping.get(model_name, "Qwen/Qwen2.5-VL-3B-Instruct")
            model_class = Qwen2_5_VLForConditionalGeneration
            model_type = "qwen2.5"
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
        
        processor = AutoProcessor.from_pretrained(model_dir)
        return (QwenModel(model=model, processor=processor, device=device, model_type=model_type),)

