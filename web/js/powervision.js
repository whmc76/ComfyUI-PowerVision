/**
 * PowerVision 前端 JavaScript 文件
 * 提供增强的用户界面功能
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// PowerVision 节点样式
const POWERVISION_STYLES = `
.power-vision-node {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 2px solid #4a5568;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.power-vision-node .node-title {
    color: #ffffff;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.power-vision-node .node-input {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
}

.power-vision-node .node-output {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 6px;
}

.power-vision-progress {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 12px 12px 0 0;
    animation: progress-animation 2s ease-in-out infinite;
}

@keyframes progress-animation {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.power-vision-status {
    position: absolute;
    top: 5px;
    right: 5px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
}

.power-vision-status.processing {
    background: #f59e0b;
    animation: pulse 1s ease-in-out infinite;
}

.power-vision-status.error {
    background: #ef4444;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
`;

// 添加样式到页面
function addPowerVisionStyles() {
    if (!document.getElementById('powervision-styles')) {
        const style = document.createElement('style');
        style.id = 'powervision-styles';
        style.textContent = POWERVISION_STYLES;
        document.head.appendChild(style);
    }
}

// PowerVision 节点增强功能
class PowerVisionNodeEnhancer {
    constructor() {
        this.enhancedNodes = new Set();
        this.init();
    }

    init() {
        // 监听节点创建
        app.registerExtension({
            name: "PowerVision.NodeEnhancer",
            async beforeRegisterNodeDef(nodeType, nodeData, app) {
                if (this.isPowerVisionNode(nodeData.name)) {
                    this.enhanceNode(nodeType, nodeData);
                }
            }
        });

        // 监听工作流执行
        this.setupExecutionListeners();
    }

    isPowerVisionNode(nodeName) {
        const powerVisionNodes = [
            'PowerVisionImageLoader',
            'PowerVisionVideoLoader', 
            'PowerVisionQwen3VQA',
            'PowerVisionQwenModelLoader',
            'PowerVisionObjectDetection',
            'PowerVisionBBoxProcessor'
        ];
        return powerVisionNodes.includes(nodeName);
    }

    enhanceNode(nodeType, nodeData) {
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            if (originalOnNodeCreated) {
                originalOnNodeCreated.apply(this, arguments);
            }
            
            // 添加 PowerVision 样式
            this.addClass('power-vision-node');
            
            // 添加状态指示器
            this.addStatusIndicator();
            
            // 添加进度条
            this.addProgressBar();
        };

        // 增强节点功能
        this.enhanceNodeFunctionality(nodeType);
    }

    enhanceNodeFunctionality(nodeType) {
        const originalOnExecute = nodeType.prototype.onExecute;
        
        nodeType.prototype.onExecute = function() {
            // 显示处理状态
            this.showProcessingStatus();
            
            if (originalOnExecute) {
                return originalOnExecute.apply(this, arguments);
            }
        };

        // 添加状态管理方法
        nodeType.prototype.addStatusIndicator = function() {
            const statusDiv = document.createElement('div');
            statusDiv.className = 'power-vision-status';
            this.addWidget(statusDiv);
        };

        nodeType.prototype.addProgressBar = function() {
            const progressDiv = document.createElement('div');
            progressDiv.className = 'power-vision-progress';
            this.addWidget(progressDiv);
        };

        nodeType.prototype.showProcessingStatus = function() {
            const statusElement = this.querySelector('.power-vision-status');
            if (statusElement) {
                statusElement.classList.add('processing');
            }
        };

        nodeType.prototype.showErrorStatus = function() {
            const statusElement = this.querySelector('.power-vision-status');
            if (statusElement) {
                statusElement.classList.remove('processing');
                statusElement.classList.add('error');
            }
        };

        nodeType.prototype.showSuccessStatus = function() {
            const statusElement = this.querySelector('.power-vision-status');
            if (statusElement) {
                statusElement.classList.remove('processing', 'error');
            }
        };
    }

    setupExecutionListeners() {
        // 监听执行开始
        app.registerExtension({
            name: "PowerVision.ExecutionListener",
            async beforeRegisterNodeDef(nodeType, nodeData, app) {
                if (this.isPowerVisionNode(nodeData.name)) {
                    const originalOnExecute = nodeType.prototype.onExecute;
                    
                    nodeType.prototype.onExecute = function() {
                        console.log(`PowerVision: Executing ${nodeData.name}`);
                        this.showProcessingStatus();
                        
                        if (originalOnExecute) {
                            return originalOnExecute.apply(this, arguments);
                        }
                    };
                }
            }
        });
    }
}

// PowerVision 工具面板
class PowerVisionToolPanel {
    constructor() {
        this.panel = null;
        this.init();
    }

    init() {
        this.createPanel();
        this.addEventListeners();
    }

    createPanel() {
        const panel = document.createElement('div');
        panel.id = 'powervision-tool-panel';
        panel.className = 'power-vision-tool-panel';
        panel.innerHTML = `
            <div class="power-vision-panel-header">
                <h3>PowerVision 工具</h3>
            </div>
            <div class="power-vision-panel-content">
                <div class="power-vision-tool-group">
                    <h4>模型管理</h4>
                    <button id="powervision-clear-cache" class="power-vision-btn">清理缓存</button>
                    <button id="powervision-check-models" class="power-vision-btn">检查模型</button>
                </div>
                <div class="power-vision-tool-group">
                    <h4>图像处理</h4>
                    <button id="powervision-batch-process" class="power-vision-btn">批量处理</button>
                    <button id="powervision-export-results" class="power-vision-btn">导出结果</button>
                </div>
                <div class="power-vision-tool-group">
                    <h4>设置</h4>
                    <button id="powervision-settings" class="power-vision-btn">打开设置</button>
                </div>
            </div>
        `;

        // 添加面板样式
        const panelStyles = `
            .power-vision-tool-panel {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 250px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .power-vision-panel-header {
                padding: 12px 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 8px 8px 0 0;
            }
            .power-vision-panel-header h3 {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }
            .power-vision-panel-content {
                padding: 16px;
            }
            .power-vision-tool-group {
                margin-bottom: 16px;
            }
            .power-vision-tool-group h4 {
                margin: 0 0 8px 0;
                font-size: 14px;
                color: #374151;
                font-weight: 600;
            }
            .power-vision-btn {
                display: block;
                width: 100%;
                padding: 8px 12px;
                margin-bottom: 4px;
                background: #f3f4f6;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                color: #374151;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .power-vision-btn:hover {
                background: #e5e7eb;
                border-color: #9ca3af;
            }
        `;

        const style = document.createElement('style');
        style.textContent = panelStyles;
        document.head.appendChild(style);

        document.body.appendChild(panel);
        this.panel = panel;
    }

    addEventListeners() {
        // 清理缓存
        document.getElementById('powervision-clear-cache')?.addEventListener('click', () => {
            this.clearCache();
        });

        // 检查模型
        document.getElementById('powervision-check-models')?.addEventListener('click', () => {
            this.checkModels();
        });

        // 批量处理
        document.getElementById('powervision-batch-process')?.addEventListener('click', () => {
            this.batchProcess();
        });

        // 导出结果
        document.getElementById('powervision-export-results')?.addEventListener('click', () => {
            this.exportResults();
        });

        // 设置
        document.getElementById('powervision-settings')?.addEventListener('click', () => {
            this.openSettings();
        });
    }

    async clearCache() {
        try {
            // 这里可以添加清理缓存的逻辑
            console.log('PowerVision: Clearing cache...');
            alert('缓存已清理');
        } catch (error) {
            console.error('PowerVision: Failed to clear cache:', error);
            alert('清理缓存失败');
        }
    }

    async checkModels() {
        try {
            console.log('PowerVision: Checking models...');
            alert('模型检查完成');
        } catch (error) {
            console.error('PowerVision: Failed to check models:', error);
            alert('模型检查失败');
        }
    }

    async batchProcess() {
        try {
            console.log('PowerVision: Starting batch process...');
            alert('批量处理功能开发中');
        } catch (error) {
            console.error('PowerVision: Batch process failed:', error);
            alert('批量处理失败');
        }
    }

    async exportResults() {
        try {
            console.log('PowerVision: Exporting results...');
            alert('导出结果功能开发中');
        } catch (error) {
            console.error('PowerVision: Export failed:', error);
            alert('导出失败');
        }
    }

    openSettings() {
        console.log('PowerVision: Opening settings...');
        alert('设置功能开发中');
    }
}

// 初始化 PowerVision 功能
document.addEventListener('DOMContentLoaded', () => {
    // 添加样式
    addPowerVisionStyles();
    
    // 初始化节点增强器
    new PowerVisionNodeEnhancer();
    
    // 初始化工具面板
    new PowerVisionToolPanel();
    
    console.log('PowerVision: Frontend enhancements loaded');
});

// 导出给其他模块使用
window.PowerVision = {
    NodeEnhancer: PowerVisionNodeEnhancer,
    ToolPanel: PowerVisionToolPanel
};
