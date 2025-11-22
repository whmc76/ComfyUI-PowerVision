# Sa2VA 集成说明

## 概述

已成功将 ComfyUI-Sa2VA 项目移植到 ComfyUI-PowerVision 项目中。

## 完成的工作

### 1. 创建的文件

- **`config.py`**: 全局配置文件，包含 `be_quiet` 设置
- **`nodes/sa2va_nodes.py`**: Sa2VA 节点实现，包含 `PowerVisionSa2VASegmentation` 类

### 2. 更新的文件

- **`nodes/__init__.py`**: 
  - 添加了 Sa2VA 节点的导入
  - 更新了 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS`
  - 更新了 `__all__` 导出列表

- **`requirements.txt`**: 
  - 添加了 `qwen_vl_utils` 依赖（Sa2VA 所需）
  - 添加了相关注释说明

## 新增节点

### PowerVision Sa2VA Segmentation

**节点名称**: `PowerVisionSa2VASegmentation`  
**显示名称**: "PowerVision Sa2VA Segmentation"  
**分类**: `PowerVision/Sa2VA`

#### 功能特性

- **多模态理解**: 结合文本生成和视觉理解
- **密集分割**: 像素级精确的对象分割掩码
- **集成掩码转换**: 自动转换为 ComfyUI MASK 和 IMAGE 格式
- **内存管理**: 内置 VRAM 优化和模型生命周期管理
- **多种输出格式**: 文本、ComfyUI 掩码和可视化掩码图像

#### 输入参数

- `model_name`: 模型名称（默认: "ByteDance/Sa2VA-Qwen3-VL-4B"）
  - 支持的模型:
    - ByteDance/Sa2VA-InternVL3-2B
    - ByteDance/Sa2VA-InternVL3-8B
    - ByteDance/Sa2VA-InternVL3-14B
    - ByteDance/Sa2VA-Qwen2_5-VL-3B
    - ByteDance/Sa2VA-Qwen2_5-VL-7B
    - ByteDance/Sa2VA-Qwen3-VL-4B (推荐)
- `image`: 输入图像 (IMAGE)
- `mask_threshold`: 掩码阈值 (FLOAT, 默认: 0.5)
- `use_8bit_quantization`: 使用 8 位量化 (BOOLEAN, 默认: False)
- `use_flash_attn`: 使用 Flash Attention (BOOLEAN, 默认: True)
- `segmentation_prompt`: 分割提示词 (STRING, 多行)

#### 输出

- `text_outputs`: 文本输出列表 (LIST)
- `masks`: ComfyUI 掩码 (MASK)
- `mask_images`: 掩码图像 (IMAGE)

## 依赖要求

### 必需依赖

- `transformers >= 4.57.0` (关键！)
- `qwen_vl_utils` (Sa2VA 模型工具)
- `huggingface_hub`
- `bitsandbytes` (用于 8 位量化)

### 可选依赖

- `flash-attn`: 提高效率（需要 CUDA 支持，安装可能较复杂）

## 使用说明

### 基本图像描述

1. 添加 **Load Image** 节点并加载图像
2. 添加 **PowerVision Sa2VA Segmentation** 节点
3. 连接 Load Image → Sa2VA 节点
4. 根据需要调整 `model_name` 和 `mask_threshold`
5. 设置 `segmentation_prompt`: "请详细描述这张图片。"
6. 执行以获取文本描述

### 图像分割

1. 使用 **Load Image** 节点加载图像
2. 添加 **PowerVision Sa2VA Segmentation** 节点
3. 连接 Load Image → Sa2VA 节点
4. 根据需要调整 `model_name` 和 `mask_threshold`
5. 设置 `segmentation_prompt`: "请为图像中的所有对象提供分割掩码。"
6. 将 `masks` 输出连接到兼容掩码的节点，或将 `mask_images` 连接到 **Preview Image**
7. 节点自动提供 MASK tensor 和可视化 IMAGE tensor

## 故障排除

### 常见问题

**"No module named 'transformers.models.qwen3_vl'"**
```bash
pip install transformers>=4.57.0 --upgrade
```
这是最常见的问题 - transformers 版本太旧。

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```
此依赖是 Sa2VA 模型工具所必需的。

**CUDA Out of Memory**
- 使用 8 位量化
- 使用较小的模型变体（2B 或 3B 参数）
- 减少批次大小

**模型加载错误**
- 检查初始下载的互联网连接
- 确保有足够的磁盘空间（每个模型 20GB+）
- 验证 CUDA 兼容性
- 尝试: `torch.cuda.empty_cache()` 清理 VRAM

## 技术细节

### 模型精度

Sa2VA 模型默认使用 **bfloat16 精度**，可选择使用 bits-and-bytes 量化为 8 位。

### 内存优化

- 自动 dtype 选择（基于设备能力）
- 支持 8 位量化
- 可选的模型卸载到 CPU
- 自动 CUDA 缓存管理

### 下载和缓存

- 支持可取消的模型下载
- 显示下载进度和仓库大小
- 本地缓存管理（在 `.cache/huggingface/hub` 目录）

## 参考

- [Sa2VA Paper](https://arxiv.org/abs/2501.04001)
- [Sa2VA Models on HuggingFace](https://huggingface.co/ByteDance)
- 原始项目: [ComfyUI-Sa2VA](https://github.com/adambarbato/ComfyUI-Sa2VA)

## 版本信息

- 集成日期: 2024
- 基于: ComfyUI-Sa2VA
- PowerVision 版本: 1.2.2+

