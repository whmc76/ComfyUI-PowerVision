"""
PowerVision 工具节点

包含各种实用工具节点
"""

import json
import torch
from typing import List, Dict, Any, Tuple, Union
from PIL import Image


class PowerVisionTextProcessor:
    """PowerVision 文本处理器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "operation": (["clean", "extract_json", "format_result"], {"default": "clean"}),
                "max_length": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process"
    CATEGORY = "PowerVision/Utility"

    def process(self, text: str, operation: str, max_length: int) -> Tuple[str]:
        """处理文本"""
        if operation == "clean":
            # 清理文本
            processed = text.strip()
        elif operation == "extract_json":
            # 提取JSON
            processed = self.extract_json_from_text(text)
        elif operation == "format_result":
            # 格式化结果
            processed = self.format_detection_result(text)
        else:
            processed = text
        
        # 限制长度
        if len(processed) > max_length:
            processed = processed[:max_length] + "..."
        
        return (processed,)
    
    def extract_json_from_text(self, text: str) -> str:
        """从文本中提取JSON"""
        import re
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        
        return text
    
    def format_detection_result(self, result: str) -> str:
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


class PowerVisionResultAnalyzer:
    """PowerVision 结果分析器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection_result": ("JSON",),
                "analysis_type": (["summary", "statistics", "details"], {"default": "summary"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis",)
    FUNCTION = "analyze"
    CATEGORY = "PowerVision/Utility"

    def analyze(self, detection_result: str, analysis_type: str) -> Tuple[str]:
        """分析检测结果"""
        try:
            data = json.loads(detection_result)
            if not isinstance(data, list):
                return ("Invalid detection result format",)
            
            if analysis_type == "summary":
                return (self.generate_summary(data),)
            elif analysis_type == "statistics":
                return (self.generate_statistics(data),)
            elif analysis_type == "details":
                return (self.generate_details(data),)
            else:
                return ("Unknown analysis type",)
                
        except Exception as e:
            return (f"Analysis error: {str(e)}",)
    
    def generate_summary(self, data: List[Dict]) -> str:
        """生成摘要"""
        if not data:
            return "No objects detected."
        
        total_objects = len(data)
        labels = [item.get("label", "unknown") for item in data]
        unique_labels = list(set(labels))
        
        summary = f"Detected {total_objects} objects:\n"
        for label in unique_labels:
            count = labels.count(label)
            summary += f"- {label}: {count}\n"
        
        return summary
    
    def generate_statistics(self, data: List[Dict]) -> str:
        """生成统计信息"""
        if not data:
            return "No data to analyze."
        
        scores = [item.get("score", 0) for item in data]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        stats = f"Detection Statistics:\n"
        stats += f"Total objects: {len(data)}\n"
        stats += f"Average confidence: {avg_score:.3f}\n"
        stats += f"Highest confidence: {max_score:.3f}\n"
        stats += f"Lowest confidence: {min_score:.3f}\n"
        
        return stats
    
    def generate_details(self, data: List[Dict]) -> str:
        """生成详细信息"""
        if not data:
            return "No objects detected."
        
        details = "Detailed Detection Results:\n"
        for i, item in enumerate(data, 1):
            bbox = item.get("bbox_2d", item.get("bbox", []))
            label = item.get("label", "unknown")
            score = item.get("score", 0)
            details += f"{i}. {label} (confidence: {score:.3f}) at {bbox}\n"
        
        return details


class PowerVisionBatchProcessor:
    """PowerVision 批量处理器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "process_type": (["detection", "recognition"], {"default": "detection"}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("batch_results",)
    FUNCTION = "process_batch"
    CATEGORY = "PowerVision/Utility"

    def process_batch(
        self, 
        images: torch.Tensor, 
        batch_size: int, 
        process_type: str
    ) -> Tuple[str]:
        """批量处理图像"""
        try:
            results = []
            num_images = images.shape[0]
            
            for i in range(0, num_images, batch_size):
                batch = images[i:i+batch_size]
                batch_results = self.process_single_batch(batch, process_type)
                results.extend(batch_results)
            
            return (json.dumps(results, ensure_ascii=False),)
            
        except Exception as e:
            return (json.dumps([{"error": str(e)}], ensure_ascii=False),)
    
    def process_single_batch(self, batch: torch.Tensor, process_type: str) -> List[Dict]:
        """处理单个批次"""
        # 这里是批量处理的占位符实现
        # 实际实现需要根据具体的处理类型来调整
        results = []
        
        for i in range(batch.shape[0]):
            image = batch[i]
            if process_type == "detection":
                result = {
                    "image_index": i,
                    "objects": [],
                    "status": "processed"
                }
            else:  # recognition
                result = {
                    "image_index": i,
                    "description": "Image content description",
                    "status": "processed"
                }
            results.append(result)
        
        return results


class PowerVisionConfigManager:
    """PowerVision 配置管理器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_name": ("STRING", {"default": "default"}),
                "action": (["load", "save", "reset"], {"default": "load"}),
            },
            "optional": {
                "config_data": ("JSON",),
            }
        }

    RETURN_TYPES = ("JSON", "STRING")
    RETURN_NAMES = ("config", "status")
    FUNCTION = "manage_config"
    CATEGORY = "PowerVision/Utility"

    def manage_config(
        self, 
        config_name: str, 
        action: str, 
        config_data: str = None
    ) -> Tuple[str, str]:
        """管理配置"""
        try:
            if action == "load":
                config = self.load_config(config_name)
                return (json.dumps(config, ensure_ascii=False), "Config loaded")
            elif action == "save":
                if config_data:
                    data = json.loads(config_data)
                    self.save_config(config_name, data)
                    return (config_data, "Config saved")
                else:
                    return ("{}", "No config data provided")
            elif action == "reset":
                default_config = self.get_default_config()
                self.save_config(config_name, default_config)
                return (json.dumps(default_config, ensure_ascii=False), "Config reset")
            else:
                return ("{}", "Unknown action")
                
        except Exception as e:
            return ("{}", f"Error: {str(e)}")
    
    def load_config(self, config_name: str) -> Dict:
        """加载配置"""
        # 这里应该从文件或数据库加载配置
        # 现在返回默认配置
        return self.get_default_config()
    
    def save_config(self, config_name: str, config: Dict) -> None:
        """保存配置"""
        # 这里应该保存到文件或数据库
        pass
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model": "Qwen3-VL-4B-Instruct-FP8",
            "device": "auto",
            "precision": "BF16",
            "max_image_size": 1024,
            "temperature": 0.7,
            "max_tokens": 2048,
            "confidence_threshold": 0.5
        }

