"""
PowerVision 节点模块

统一导入所有节点类
"""

# 导入图像相关节点
from .image_nodes import (
    PowerVisionImageLoader,
    PowerVisionVideoLoader,
    PowerVisionImageProcessor,
)

# 导入模型相关节点
from .model_nodes import (
    PowerVisionQwen3VQA,
    PowerVisionQwen3VQAWithModel,
    PowerVisionQwenModelLoader,
    QwenModel,
)

# 导入检测相关节点
from .detection_nodes import (
    PowerVisionObjectDetection,
    PowerVisionBBoxProcessor,
    PowerVisionDetectionFilter,
    DetectedBox,
)

# 导入工具节点
from .utility_nodes import (
    PowerVisionTextProcessor,
    PowerVisionResultAnalyzer,
    PowerVisionBatchProcessor,
    PowerVisionConfigManager,
)

# 节点映射
NODE_CLASS_MAPPINGS = {
    # 图像节点
    "PowerVisionImageLoader": PowerVisionImageLoader,
    "PowerVisionVideoLoader": PowerVisionVideoLoader,
    "PowerVisionImageProcessor": PowerVisionImageProcessor,
    
    # 模型节点
    "PowerVisionQwen3VQA": PowerVisionQwen3VQA,
    "PowerVisionQwen3VQAWithModel": PowerVisionQwen3VQAWithModel,
    "PowerVisionQwenModelLoader": PowerVisionQwenModelLoader,
    
    # 检测节点
    "PowerVisionObjectDetection": PowerVisionObjectDetection,
    "PowerVisionBBoxProcessor": PowerVisionBBoxProcessor,
    "PowerVisionDetectionFilter": PowerVisionDetectionFilter,
    
    # 工具节点
    "PowerVisionTextProcessor": PowerVisionTextProcessor,
    "PowerVisionResultAnalyzer": PowerVisionResultAnalyzer,
    "PowerVisionBatchProcessor": PowerVisionBatchProcessor,
    "PowerVisionConfigManager": PowerVisionConfigManager,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    # 图像描述节点
    "PowerVisionImageLoader": "PowerVision Image Loader",
    "PowerVisionVideoLoader": "PowerVision Video Loader",
    "PowerVisionImageProcessor": "PowerVision Image Processor",
    
    # 模型节点
    "PowerVisionQwen3VQA": "PowerVision Qwen3-VL VQA",
    "PowerVisionQwen3VQAWithModel": "PowerVision Qwen3-VL VQA (With Model)",
    "PowerVisionQwenModelLoader": "PowerVision Qwen Model Loader",
    
    # 目标检测节点
    "PowerVisionObjectDetection": "PowerVision Object Detection",
    "PowerVisionBBoxProcessor": "PowerVision BBox Processor",
    "PowerVisionDetectionFilter": "PowerVision Detection Filter",
    
    # 工具节点
    "PowerVisionTextProcessor": "PowerVision Text Processor",
    "PowerVisionResultAnalyzer": "PowerVision Result Analyzer",
    "PowerVisionBatchProcessor": "PowerVision Batch Processor",
    "PowerVisionConfigManager": "PowerVision Config Manager",
}

# 导出所有内容
__all__ = [
    # 节点类
    "PowerVisionImageLoader",
    "PowerVisionVideoLoader", 
    "PowerVisionImageProcessor",
    "PowerVisionQwen3VQA",
    "PowerVisionQwen3VQAWithModel",
    "PowerVisionQwenModelLoader",
    "PowerVisionObjectDetection",
    "PowerVisionBBoxProcessor",
    "PowerVisionDetectionFilter",
    "PowerVisionTextProcessor",
    "PowerVisionResultAnalyzer",
    "PowerVisionBatchProcessor",
    "PowerVisionConfigManager",
    # 数据类
    "QwenModel",
    "DetectedBox",
    # 映射
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

