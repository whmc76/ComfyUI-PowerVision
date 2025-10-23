#!/usr/bin/env python3
"""
PowerVision 基础测试脚本

测试插件的基本功能，不依赖 ComfyUI 环境
"""

import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """测试文件结构"""
    print("🔍 测试文件结构...")
    
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

def test_code_syntax():
    """测试代码语法"""
    print("\n🔍 测试代码语法...")
    
    python_files = ["__init__.py", "nodes.py", "utils.py"]
    
    for file in python_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # 简单的语法检查
                compile(code, file, 'exec')
                print(f"✅ {file} 语法正确")
                
            except SyntaxError as e:
                print(f"❌ {file} 语法错误: {e}")
                return False
            except Exception as e:
                print(f"⚠️  {file} 检查异常: {e}")
    
    return True

def test_json_files():
    """测试JSON文件"""
    print("\n🔍 测试JSON文件...")
    
    json_files = [
        "examples/basic_image_recognition.json",
        "examples/advanced_object_detection.json", 
        "examples/video_analysis.json"
    ]
    
    for file in json_files:
        if os.path.exists(file):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查必要字段
                required_fields = ["nodes", "links"]
                for field in required_fields:
                    if field not in data:
                        print(f"❌ {file} 缺少字段: {field}")
                        return False
                
                print(f"✅ {file} JSON格式正确")
                
            except json.JSONDecodeError as e:
                print(f"❌ {file} JSON格式错误: {e}")
                return False
            except Exception as e:
                print(f"⚠️  {file} 检查异常: {e}")
    
    return True

def test_documentation():
    """测试文档完整性"""
    print("\n🔍 测试文档完整性...")
    
    doc_files = ["README.md", "rules.md", "ATTRIBUTION.md"]
    
    for file in doc_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查文档长度
            if len(content) < 100:
                print(f"⚠️  {file} 内容过短")
            else:
                print(f"✅ {file} 内容完整 ({len(content)} 字符)")
        else:
            print(f"❌ {file} 不存在")
            return False
    
    return True

def test_web_assets():
    """测试前端资源"""
    print("\n🔍 测试前端资源...")
    
    web_files = [
        "web/js/powervision.js",
        "web/css/powervision.css"
    ]
    
    for file in web_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) > 0:
                print(f"✅ {file} 存在且有内容 ({len(content)} 字符)")
            else:
                print(f"⚠️  {file} 存在但为空")
        else:
            print(f"❌ {file} 不存在")
            return False
    
    return True

def test_requirements():
    """测试依赖文件"""
    print("\n🔍 测试依赖文件...")
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含必要的依赖
        required_deps = ["torch", "transformers", "PIL", "numpy"]
        missing_deps = []
        
        for dep in required_deps:
            if dep.lower() not in content.lower():
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"⚠️  缺少依赖: {', '.join(missing_deps)}")
        else:
            print("✅ requirements.txt 包含必要依赖")
        
        print(f"✅ requirements.txt 存在 ({len(content)} 字符)")
        return True
    else:
        print("❌ requirements.txt 不存在")
        return False

def main():
    """主测试函数"""
    print("🚀 PowerVision 基础测试开始\n")
    
    tests = [
        ("文件结构", test_file_structure),
        ("代码语法", test_code_syntax),
        ("JSON文件", test_json_files),
        ("文档完整性", test_documentation),
        ("前端资源", test_web_assets),
        ("依赖文件", test_requirements),
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
        print("🎉 所有基础测试通过！PowerVision 插件结构完整。")
        print("\n📋 下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 将插件复制到 ComfyUI/custom_nodes/ 目录")
        print("3. 启动 ComfyUI 测试插件功能")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
