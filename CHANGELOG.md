# ComfyUI PowerVision 更新日志

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
