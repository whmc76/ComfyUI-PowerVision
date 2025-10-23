# ComfyUI PowerVision 插件开发规范

## 代码规范

### Python 代码规范
1. **PEP 8 标准**: 严格遵循 Python PEP 8 代码风格指南
2. **类型注解**: 所有函数和类方法必须包含完整的类型注解
3. **文档字符串**: 所有公共函数、类和方法必须包含详细的 docstring
4. **命名规范**:
   - 类名使用 PascalCase (如: `PowerVisionNode`)
   - 函数和变量名使用 snake_case (如: `process_image`)
   - 常量使用 UPPER_CASE (如: `MAX_IMAGE_SIZE`)

### ComfyUI 特定规范

#### 节点类规范
```python
class PowerVisionNode:
    """
    PowerVision 节点基类
    
    Args:
        input_types: 输入类型定义
        output_types: 输出类型定义
        category: 节点分类
        description: 节点描述
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入类型"""
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "result_text")
    FUNCTION = "process"
    CATEGORY = "PowerVision"
    
    def process(self, image, prompt, strength=1.0):
        """
        处理图像和提示
        
        Args:
            image: 输入图像张量
            prompt: 文本提示
            strength: 处理强度
            
        Returns:
            tuple: (输出图像, 结果文本)
        """
        # 实现逻辑
        pass
```

#### 输入类型规范
- `IMAGE`: 图像张量，形状为 (batch, height, width, channels)
- `STRING`: 字符串输入
- `INT`: 整数输入
- `FLOAT`: 浮点数输入
- `BOOLEAN`: 布尔值输入
- `MASK`: 掩码张量

#### 输出类型规范
- 必须明确定义 `RETURN_TYPES` 和 `RETURN_NAMES`
- 输出类型必须与 ComfyUI 标准兼容

### 错误处理规范
```python
def safe_process(self, *args, **kwargs):
    """安全的处理函数，包含错误处理"""
    try:
        # 主要处理逻辑
        result = self.process(*args, **kwargs)
        return result
    except Exception as e:
        print(f"PowerVision 错误: {str(e)}")
        # 返回默认值或重新抛出异常
        raise e
```

### 性能优化规范
1. **内存管理**: 及时释放不需要的张量
2. **批处理**: 支持批量处理以提高效率
3. **缓存**: 对重复计算进行适当缓存
4. **异步处理**: 对耗时操作使用异步处理

### 测试规范
1. **单元测试**: 每个节点类必须有对应的单元测试
2. **集成测试**: 测试节点在 ComfyUI 中的集成
3. **性能测试**: 测试内存使用和处理速度

### 文档规范
1. **README.md**: 包含安装说明、使用示例和 API 文档
2. **代码注释**: 关键逻辑必须有详细注释
3. **类型注解**: 所有函数参数和返回值必须有类型注解

### 依赖管理规范
1. **requirements.txt**: 列出所有必需的依赖包
2. **版本固定**: 指定依赖包的具体版本
3. **可选依赖**: 将可选依赖单独列出

### 文件结构规范
```
ComfyUI_PowerVision/
├── __init__.py          # 插件入口文件
├── nodes.py             # 主要节点定义
├── utils.py             # 工具函数
├── models/              # 模型相关代码
├── web/                 # 前端资源
│   ├── js/             # JavaScript 文件
│   └── css/            # CSS 文件
├── examples/            # 示例工作流
├── tests/              # 测试文件
├── requirements.txt     # 依赖列表
├── README.md           # 项目说明
└── .gitignore          # Git 忽略文件
```

### 版本控制规范
1. **语义化版本**: 使用语义化版本号 (如: 1.0.0)
2. **变更日志**: 维护详细的变更日志
3. **标签**: 为每个发布版本创建标签

### 安全规范
1. **输入验证**: 验证所有用户输入
2. **资源限制**: 限制内存和计算资源使用
3. **错误信息**: 不暴露敏感信息给用户

### 国际化规范
1. **多语言支持**: 支持中英文界面
2. **本地化**: 根据用户语言环境显示相应文本
3. **字符编码**: 使用 UTF-8 编码

## 代码审查清单

- [ ] 代码符合 PEP 8 标准
- [ ] 包含完整的类型注解
- [ ] 有详细的文档字符串
- [ ] 错误处理完善
- [ ] 性能优化合理
- [ ] 测试覆盖充分
- [ ] 文档完整
- [ ] 依赖管理正确
- [ ] 安全考虑周全
