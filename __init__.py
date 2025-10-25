"""
ComfyUI-PowerVision 插件

这是一个基于 ComfyUI 的 PowerVision 插件，提供强大的视觉处理功能。

作者: PowerVision Team
版本: 1.2.2
许可证: MIT

开源项目来源声明:
- 基于 ComfyUI_Qwen3-VL-Instruct 和 Comfyui_Object_Detect_QWen_VL 项目进行开发
- 详细来源信息请查看 ATTRIBUTION.md 文件
- 本项目严格遵循开源精神，明确标注所有代码来源
"""

import os
import sys
from typing import Dict, Any, Optional

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入节点模块
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("PowerVision: 节点模块导入成功")
    print(f"PowerVision: 加载了 {len(NODE_CLASS_MAPPINGS)} 个节点")
except ImportError as e:
    print(f"PowerVision 插件导入错误: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# 插件信息
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# ComfyUI 插件标准接口
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY"
]

# 插件元数据
PLUGIN_INFO = {
    "name": "ComfyUI-PowerVision",
    "version": "1.2.2",
    "description": "强大的视觉处理插件",
    "author": "PowerVision Team",
    "license": "MIT",
    "repository": "https://github.com/whmc76/ComfyUI_PowerVision",
    "documentation": "https://github.com/whmc76/ComfyUI_PowerVision/wiki"
}

def get_plugin_info() -> Dict[str, Any]:
    """
    获取插件信息
    
    Returns:
        Dict[str, Any]: 插件信息字典
    """
    return PLUGIN_INFO.copy()

def check_dependencies() -> bool:
    """
    检查插件依赖是否满足
    
    Returns:
        bool: 依赖是否满足
    """
    try:
        # 检查必要的依赖包
        import torch
        import numpy as np
        import PIL
        return True
    except ImportError as e:
        print(f"PowerVision 插件依赖检查失败: {e}")
        return False

def initialize_plugin() -> bool:
    """
    初始化插件
    
    Returns:
        bool: 初始化是否成功
    """
    try:
        # 检查依赖
        if not check_dependencies():
            return False
            
        # 初始化插件特定资源
        print("PowerVision 插件初始化成功")
        return True
    except Exception as e:
        print(f"PowerVision 插件初始化失败: {e}")
        return False

# 自动初始化
if __name__ != "__main__":
    initialize_plugin()
