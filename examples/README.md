# PowerVision 示例工作流

本目录包含了 PowerVision 插件的示例工作流，展示了各种使用场景和功能。

## 示例文件

### 1. basic_image_recognition.json
**基础图像识别示例**

- **功能**: 演示基本的图像识别和描述功能
- **节点**: 
  - PowerVisionImageLoader: 加载图像
  - PowerVisionQwen3VQA: 视觉问答
  - PowerVisionQwenModelLoader: 模型加载
  - PowerVisionObjectDetection: 目标检测
- **用途**: 学习插件的基本使用方法

### 2. advanced_object_detection.json
**高级目标检测示例**

- **功能**: 演示多类别目标检测和结果处理
- **特点**:
  - 同时检测多个类别的对象
  - 使用不同的检测参数
  - 合并检测结果
  - 边界框处理
- **适用场景**: 智能监控、自动驾驶、图像分析

### 3. video_analysis.json
**视频分析示例**

- **功能**: 演示视频内容理解和分析
- **特点**:
  - 视频内容理解
  - 多模态分析
  - 目标检测
  - 结果展示
- **适用场景**: 视频内容审核、智能监控、视频摘要

## 使用方法

1. **导入工作流**:
   - 在 ComfyUI 中点击 "Load" 按钮
   - 选择对应的 JSON 文件
   - 工作流将自动加载

2. **准备输入**:
   - 将示例图像/视频文件放入 ComfyUI 的 `input` 目录
   - 修改节点中的文件名以匹配您的文件

3. **调整参数**:
   - 根据需要调整模型参数
   - 修改检测目标类别
   - 调整置信度阈值

4. **运行工作流**:
   - 点击 "Queue Prompt" 开始执行
   - 查看输出结果

## 参数说明

### 模型参数
- **model**: 选择使用的 Qwen3-VL 模型
- **quantization**: 量化方式 (none/4bit/8bit)
- **device**: 计算设备 (auto/cuda/cpu)
- **precision**: 精度设置 (BF16/FP16/FP32)

### 检测参数
- **target**: 检测目标类别
- **score_threshold**: 置信度阈值
- **merge_boxes**: 是否合并边界框
- **bbox_selection**: 边界框选择

### 生成参数
- **temperature**: 生成温度
- **max_new_tokens**: 最大生成令牌数
- **min_pixels/max_pixels**: 图像像素范围

## 自定义工作流

### 创建新的工作流
1. 从基础示例开始
2. 添加或删除节点
3. 调整连接关系
4. 修改参数设置
5. 测试和优化

### 常用组合
- **图像描述**: ImageLoader → Qwen3VQA
- **目标检测**: ImageLoader → ModelLoader → ObjectDetection
- **视频分析**: VideoLoader → Qwen3VQA
- **批量处理**: 多个 ImageLoader → 并行处理

## 故障排除

### 常见问题
1. **模型加载失败**: 检查网络连接和磁盘空间
2. **内存不足**: 降低模型精度或使用更小的模型
3. **检测结果为空**: 调整置信度阈值
4. **处理速度慢**: 使用量化模型或更小的模型

### 性能优化
- 使用 FP8 量化模型提高速度
- 调整图像尺寸减少计算量
- 使用 Flash Attention 2 加速
- 合理设置批处理大小

## 扩展功能

### 集成其他插件
- 与 SAM2 集成进行图像分割
- 与 ControlNet 集成进行图像控制
- 与 AnimateDiff 集成进行视频生成

### 自定义节点
- 创建专门的图像预处理节点
- 添加结果后处理节点
- 实现批量处理节点

## 技术支持

如果遇到问题，请：
1. 查看控制台错误信息
2. 检查模型文件完整性
3. 确认依赖包版本
4. 参考 GitHub Issues

---

**注意**: 示例工作流仅用于演示，实际使用时请根据具体需求进行调整和优化。
