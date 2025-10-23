#!/usr/bin/env python3
"""
PowerVision 插件集成测试脚本

用于测试插件的各个组件是否正常工作
"""

import os
import sys
import torch
import json
from pathlib import Path

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试基础导入
        import nodes
        print("✅ nodes 模块导入成功")
        
        import utils
        print("✅ utils 模块导入成功")
        
        # 测试节点映射
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("✅ 节点映射导入成功")
        
        # 检查节点数量
        node_count = len(NODE_CLASS_MAPPINGS)
        print(f"📊 发现 {node_count} 个节点")
        
        for node_name in NODE_CLASS_MAPPINGS.keys():
            print(f"  - {node_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_node_classes():
    """测试节点类"""
    print("\n🔍 测试节点类...")
    
    try:
        from nodes import (
            PowerVisionImageLoader,
            PowerVisionVideoLoader,
            PowerVisionQwen3VQA,
            PowerVisionQwenModelLoader,
            PowerVisionObjectDetection,
            PowerVisionBBoxProcessor
        )
        
        # 测试节点类实例化
        nodes_to_test = [
            ("PowerVisionImageLoader", PowerVisionImageLoader),
            ("PowerVisionVideoLoader", PowerVisionVideoLoader),
            ("PowerVisionQwen3VQA", PowerVisionQwen3VQA),
            ("PowerVisionQwenModelLoader", PowerVisionQwenModelLoader),
            ("PowerVisionObjectDetection", PowerVisionObjectDetection),
            ("PowerVisionBBoxProcessor", PowerVisionBBoxProcessor),
        ]
        
        for name, node_class in nodes_to_test:
            try:
                # 测试 INPUT_TYPES
                input_types = node_class.INPUT_TYPES()
                print(f"✅ {name} INPUT_TYPES 正常")
                
                # 测试 RETURN_TYPES
                if hasattr(node_class, 'RETURN_TYPES'):
                    return_types = node_class.RETURN_TYPES
                    print(f"✅ {name} RETURN_TYPES 正常")
                
                # 测试 FUNCTION
                if hasattr(node_class, 'FUNCTION'):
                    function_name = node_class.FUNCTION
                    print(f"✅ {name} FUNCTION: {function_name}")
                
            except Exception as e:
                print(f"❌ {name} 测试失败: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 节点类测试失败: {e}")
        return False

def test_utils():
    """测试工具函数"""
    print("\n🔍 测试工具函数...")
    
    try:
        from utils import (
            ImageProcessor,
            ModelManager,
            BBoxProcessor,
            TextProcessor,
            FileManager,
            ConfigManager,
            ErrorHandler
        )
        
        # 测试图像处理器
        processor = ImageProcessor()
        print("✅ ImageProcessor 创建成功")
        
        # 测试模型管理器
        model_manager = ModelManager()
        device_info = model_manager.get_device_info()
        print(f"✅ ModelManager 设备信息: {device_info['device']}")
        
        # 测试边界框处理器
        bbox_processor = BBoxProcessor()
        test_bbox = [100, 100, 200, 200]
        validated_bbox = bbox_processor.validate_bbox(test_bbox, 500, 500)
        print(f"✅ BBoxProcessor 验证成功: {validated_bbox}")
        
        # 测试文本处理器
        text_processor = TextProcessor()
        cleaned_text = text_processor.clean_text("  Hello World  ")
        print(f"✅ TextProcessor 清理成功: '{cleaned_text}'")
        
        # 测试文件管理器
        file_manager = FileManager()
        temp_path = file_manager.get_temp_path("test.txt")
        print(f"✅ FileManager 临时路径: {temp_path}")
        
        # 测试配置管理器
        config_manager = ConfigManager()
        default_model = config_manager.get_config("default_model")
        print(f"✅ ConfigManager 默认模型: {default_model}")
        
        # 测试错误处理器
        error_handler = ErrorHandler()
        test_error = Exception("CUDA out of memory")
        error_msg = error_handler.handle_model_error(test_error)
        print(f"✅ ErrorHandler 错误处理: {error_msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("\n🔍 测试依赖包...")
    
    required_packages = [
        "torch",
        "numpy", 
        "PIL",
        "transformers",
        "huggingface_hub"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def test_torch_functionality():
    """测试 PyTorch 功能"""
    print("\n🔍 测试 PyTorch 功能...")
    
    try:
        # 测试基本张量操作
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        print(f"✅ 张量运算正常: {z.shape}")
        
        # 测试 CUDA 可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.device_count()} 个设备")
            print(f"   当前设备: {torch.cuda.current_device()}")
            print(f"   设备名称: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA 不可用，将使用 CPU")
        
        # 测试内存管理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ CUDA 缓存清理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch 测试失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n🔍 测试文件结构...")
    
    required_files = [
        "__init__.py",
        "nodes.py", 
        "utils.py",
        "requirements.txt",
        "README.md",
        "rules.md",
        "ATTRIBUTION.md",
        ".gitignore"
    ]
    
    required_dirs = [
        "web",
        "web/js",
        "web/css", 
        "examples"
    ]
    
    missing_files = []
    missing_dirs = []
    
    # 检查文件
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file} 存在")
    
    # 检查目录
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/ 存在")
    
    if missing_files:
        print(f"\n❌ 缺少文件: {', '.join(missing_files)}")
    
    if missing_dirs:
        print(f"\n❌ 缺少目录: {', '.join(missing_dirs)}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def main():
    """主测试函数"""
    print("🚀 PowerVision 插件集成测试开始\n")
    
    tests = [
        ("文件结构", test_file_structure),
        ("依赖包", test_dependencies),
        ("PyTorch 功能", test_torch_functionality),
        ("模块导入", test_imports),
        ("节点类", test_node_classes),
        ("工具函数", test_utils),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过！PowerVision 插件已准备就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
