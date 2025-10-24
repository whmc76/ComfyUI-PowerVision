# ComfyUI PowerVision 更新日志

## [1.2.1] - 2025-01-XX

### 🔧 修复
- **修复 Thinking 模型输出干扰**：增强 `parse_json` 函数，自动过滤 Thinking 模型的思考内容（`<think>...</think>` 和 `<thinking>...</thinking>` 标签）
- **改进 JSON 解析稳定性**：避免 Thinking 模型的思考内容干扰 JSON 格式解析

### 🎯 改进
- **添加正则表达式支持**：使用 `re` 模块过滤 Thinking 模型的思考内容
- **支持多种标签格式**：支持 `<think>`、`<thinking>` 等多种 Thinking 模型使用的标签格式

---

## [1.2.0] - 2025-01-XX

### 🚀 新功能
- **修复 Qwen3-VL 坐标系问题**：Qwen3-VL 使用归一化到 1000 的相对坐标系，现已正确转换
- **改进检测精度**：为 Qwen2.5-VL 使用 slow processor (`use_fast=False`)，提升检测准确度
- **增强格式兼容性**：支持多种 JSON 输出格式，包括字典和直接坐标数组

### 🔧 修复
- **修复 Qwen3-VL 坐标超出问题**：正确处理 Qwen3-VL 返回的归一化坐标（0-1000 范围）
- **修复 Qwen2.5-VL 检测偏差**：使用 slow processor 解决 fast processor 导致的检测偏移
- **修复 `NameError: name 'target' is not defined`**：在 `parse_boxes` 函数中添加 `target` 参数
- **修复边界框格式处理**：增强对 Qwen3-VL 直接返回坐标数组的支持

### 🎯 改进
- **优化提示词**：使用更精确的检测提示词，明确指定 JSON 格式和坐标系统
- **改进坐标转换逻辑**：
  - Qwen2.5-VL：直接使用输入尺寸坐标，缩放至原始图像
  - Qwen3-VL：先转换归一化坐标（1000）→ 输入尺寸坐标 → 原始图像坐标
- **添加调试信息**：输出原始坐标、图像尺寸、输入尺寸和缩放比例，便于问题排查
- **改进坐标范围验证**：确保边界框坐标不超过图像尺寸

### 📝 技术说明
- Qwen3-VL 坐标系详解：
  - Qwen3-VL 返回归一化到 1000 的相对坐标，与 Qwen2.5-VL 的绝对坐标不同
  - 转换公式：`abs_x = (norm_x / 1000.0) * img_width`
  - 这种设计提供了跨图像尺寸的统一坐标表示
- 参考文档：MS-Swift 最佳实践文档 ([swift.readthedocs.io](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3-VL%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html))

---

## [1.1.0] - 2024-12-19

### 🚀 新功能
- 支持 Qwen2.5-VL 和 Qwen3-VL 双模型架构
- 智能模型检测与自动回退机制
- 新增模型复用功能，支持 VQA 节点使用已加载的模型
- 优化节点分类，使用更描述性的分类名称

### 🔧 修复
- 修复 `Any` 类型导入错误
- 修复 `transformers` 库导入问题，添加 Qwen3-VL 回退机制
- 修复模型下载超时问题，优先使用现有模型文件夹
- 修复 `source_path` 类型错误，支持列表和字符串输入
- 修复 `file://` 协议错误，正确处理图片路径
- 修复空视频输入导致的索引错误
- 修复设备 `auto` 字符串错误，自动转换为具体设备类型
- 修复模型检测逻辑，避免不必要的重复下载

### 📦 依赖更新
- 更新 `transformers` 版本要求到 `>=4.57.1`
- 添加 `typing_extensions` 依赖
- 添加 `einops` 依赖

### 🎯 改进
- 改进图片输入优先级，VQA 节点优先使用直接传入的图片
- 优化模型加载路径，统一使用 `ComfyUI/models/Qwen/` 目录
- 增强错误处理和用户反馈
- 改进节点分类名称：
  - "PowerVision/Image" → "PowerVision/Image Caption"
  - "PowerVision/Detection" → "PowerVision/Target Detection"
  - "PowerVision/Utility" 保持不变

### 🛠️ 技术改进
- 实现智能模型选择逻辑
- 添加模型映射机制（Qwen3-VL → Qwen2.5-VL）
- 优化设备管理，支持自动设备检测
- 改进文件路径处理，支持多种模型文件夹格式

### 📝 文档
- 更新 README.md 文件
- 添加详细的依赖说明
- 完善项目结构说明

---

## [1.0.0] - 2024-12-19

### 🎉 初始版本
- 基础 ComfyUI PowerVision 插件功能
- 支持图像和视频加载
- 支持 Qwen 模型 VQA 功能
- 支持目标检测功能
- 基础工具节点
