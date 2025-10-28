# 更新日志

所有重要的项目更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且此项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

## [1.5.0] - 2024-12-19

### 新增
- Qwen2.5-VL 视频处理采用帧采样策略
- 添加视频帧提取功能（使用 OpenCV）
- 支持从视频文件中提取并采样关键帧
- 动态像素参数控制，自动优化内存使用

### 更改
- Qwen2.5-VL 视频处理改为使用 PIL Image 列表而非文件路径
- 优化默认像素参数（min_pixels: 64*28*28, max_pixels: 320*28*28）
- 采样模式下增加生成令牌数限制（max_new_tokens: 1024）
- 处理器初始化添加 trust_remote_code=True 参数
- 使用 @torch.no_grad() 装饰器减少内存使用
- 改进内存清理和同步机制

### 修复
- 修复 Qwen2.5-VL 视频处理内存不足问题
- 修复 source_path 视频输入未使用采样策略的问题
- 统一 video 和 source_path 两种视频输入方式的处理逻辑
- 修复无效的 generation 参数警告
- 优化 VideoFromFile 对象路径提取

### 技术改进
- 参考 ComfyUI-QwenVL 项目实现采样策略
- 添加视频帧采样函数，支持均匀采样
- 增强错误处理和资源管理
- 改进日志输出，提供更详细的处理信息
- 优化 GPU 内存管理和清理

## [1.4.0] - 2024-12-19

### 新增
- 支持直接视频输入处理，无需通过 source_path
- 增强的 VideoFromFile 对象处理，支持多种路径提取方法
- 改进的多模态输入处理逻辑

### 更改
- 重构节点结构，移除重复的 PowerVision Qwen3-VL VQA 节点
- 重命名 PowerVisionQwen3VQAWithModel 为 PowerVisionQwen3VQA
- 移除 PowerVision Batch Processor 和 PowerVision Config Manager 节点
- 优化视频文件路径处理，移除不必要的 file:// 前缀
- 清理调试日志，提升代码可读性

### 修复
- 修复视频文件被误识别为图像文件的问题
- 修复 VideoFromFile 对象路径提取失败的问题
- 修复空列表传递给处理器导致的 IndexError
- 修复字符串和列表拼接的类型错误

### 技术改进
- 增强错误处理机制
- 改进文件类型自动识别
- 优化内存使用和性能
- 完善备用 process_vision_info 实现

## [1.3.0] - 2024-12-18

### 新增
- 集成 Qwen3-VL 模型支持
- 多模态视觉问答功能
- 图像和视频理解能力
- 目标检测和分割功能

### 更改
- 更新 transformers 依赖版本至 >= 4.57.0
- 优化模型加载和推理性能
- 改进用户界面和体验

### 修复
- 修复模型加载兼容性问题
- 修复内存溢出问题
- 修复多线程处理问题

## [1.0.0] - 2024-01-XX

### 新增
- 🎉 初始版本发布
- ✨ 支持 Qwen-VL 模型
- ✨ 支持 SAM2 分割模型
- ✨ 基础图像处理功能
- 📚 完整的文档和示例

---

## 版本说明

- **主版本号**: 不兼容的 API 修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修正

## 贡献

如果您发现任何问题或有改进建议，请通过 [Issues](https://github.com/whmc76/ComfyUI-PowerVision/issues) 联系我们。