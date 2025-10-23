"""
PowerVision 工具函数和辅助类

提供图像处理、模型管理、设备管理等辅助功能
"""

import os
import json
import hashlib
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageOps, ImageSequence
from pathlib import Path
import folder_paths
import comfy.model_management


class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:  # 批次维度
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3:
                # 确保通道在最后
                if tensor.shape[0] == 3 or tensor.shape[0] == 4:
                    tensor = tensor.permute(1, 2, 0)
            
            # 确保值在0-1范围内
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            # 转换为numpy数组
            array = (tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            return Image.fromarray(array)
        else:
            raise ValueError("Input must be a torch.Tensor")
    
    @staticmethod
    def pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量"""
        if isinstance(image, Image.Image):
            array = np.array(image).astype(np.float32) / 255.0
            return torch.from_numpy(array)[None,]
        else:
            raise ValueError("Input must be a PIL.Image")
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """调整图像大小，保持宽高比"""
        w, h = image.size
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def normalize_image(image: torch.Tensor) -> torch.Tensor:
        """标准化图像张量"""
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image.clamp(0, 1)


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.device = comfy.model_management.get_torch_device()
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_capability": torch.cuda.get_device_capability(self.device),
                "cuda_memory_allocated": torch.cuda.memory_allocated(self.device),
                "cuda_memory_reserved": torch.cuda.memory_reserved(self.device),
            })
        
        return info
    
    def clear_cache(self):
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def move_to_device(self, model: Any, device: str) -> Any:
        """将模型移动到指定设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        
        try:
            model.to(device)
            return model
        except Exception as e:
            print(f"Failed to move model to {device}: {e}")
            return model


class BBoxProcessor:
    """边界框处理器"""
    
    @staticmethod
    def parse_bboxes_from_json(json_str: str) -> List[Dict[str, Any]]:
        """从JSON字符串解析边界框"""
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "bboxes" in data:
                return data["bboxes"]
            else:
                return []
        except Exception:
            return []
    
    @staticmethod
    def filter_bboxes_by_score(bboxes: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """根据分数过滤边界框"""
        return [bbox for bbox in bboxes if bbox.get("score", 0) >= threshold]
    
    @staticmethod
    def merge_bboxes(bboxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个边界框"""
        if not bboxes:
            return {"bbox": [0, 0, 0, 0], "score": 0, "label": ""}
        
        x1 = min(bbox["bbox"][0] for bbox in bboxes)
        y1 = min(bbox["bbox"][1] for bbox in bboxes)
        x2 = max(bbox["bbox"][2] for bbox in bboxes)
        y2 = max(bbox["bbox"][3] for bbox in bboxes)
        score = max(bbox.get("score", 0) for bbox in bboxes)
        label = bboxes[0].get("label", "")
        
        return {
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "label": label
        }
    
    @staticmethod
    def validate_bbox(bbox: List[int], img_width: int, img_height: int) -> List[int]:
        """验证和修正边界框坐标"""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(x1, min(x2, img_width))
        y2 = max(y1, min(y2, img_height))
        return [x1, y1, x2, y2]


class TextProcessor:
    """文本处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        return text.strip()
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """从文本中提取JSON"""
        import re
        
        # 查找JSON模式
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        
        return None
    
    @staticmethod
    def format_detection_result(result: str) -> str:
        """格式化检测结果"""
        try:
            data = json.loads(result)
            if isinstance(data, list):
                formatted = []
                for i, item in enumerate(data):
                    bbox = item.get("bbox_2d", item.get("bbox", []))
                    label = item.get("label", "object")
                    score = item.get("score", 0)
                    formatted.append(f"Object {i+1}: {label} (confidence: {score:.2f}) at {bbox}")
                return "\n".join(formatted)
            else:
                return result
        except:
            return result


class FileManager:
    """文件管理器"""
    
    @staticmethod
    def get_temp_path(filename: str) -> str:
        """获取临时文件路径"""
        temp_dir = folder_paths.temp_directory
        return os.path.join(temp_dir, filename)
    
    @staticmethod
    def save_image(image: Image.Image, path: str) -> str:
        """保存图像"""
        image.save(path)
        return path
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """获取文件哈希值"""
        m = hashlib.sha256()
        with open(filepath, "rb") as f:
            m.update(f.read())
        return m.digest().hex()
    
    @staticmethod
    def ensure_dir(path: str):
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = {
            "default_model": "Qwen3-VL-4B-Instruct-FP8",
            "default_device": "auto",
            "default_precision": "BF16",
            "max_image_size": 1024,
            "default_temperature": 0.7,
            "default_max_tokens": 2048,
        }
    
    def get_config(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
    
    def load_from_file(self, filepath: str):
        """从文件加载配置"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"Failed to load config: {e}")
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save config: {e}")


class ErrorHandler:
    """错误处理器"""
    
    @staticmethod
    def handle_model_error(error: Exception) -> str:
        """处理模型相关错误"""
        error_msg = str(error)
        if "CUDA" in error_msg:
            return "CUDA错误，请检查GPU驱动和CUDA安装"
        elif "memory" in error_msg.lower():
            return "内存不足，请尝试使用更小的模型或降低精度"
        elif "model" in error_msg.lower():
            return "模型加载失败，请检查模型文件"
        else:
            return f"模型错误: {error_msg}"
    
    @staticmethod
    def handle_image_error(error: Exception) -> str:
        """处理图像相关错误"""
        error_msg = str(error)
        if "format" in error_msg.lower():
            return "不支持的图像格式"
        elif "size" in error_msg.lower():
            return "图像尺寸过大或过小"
        else:
            return f"图像处理错误: {error_msg}"
    
    @staticmethod
    def handle_bbox_error(error: Exception) -> str:
        """处理边界框相关错误"""
        error_msg = str(error)
        if "coordinate" in error_msg.lower():
            return "边界框坐标无效"
        elif "format" in error_msg.lower():
            return "边界框格式错误"
        else:
            return f"边界框处理错误: {error_msg}"


# 全局实例
image_processor = ImageProcessor()
model_manager = ModelManager()
bbox_processor = BBoxProcessor()
text_processor = TextProcessor()
file_manager = FileManager()
config_manager = ConfigManager()
error_handler = ErrorHandler()
