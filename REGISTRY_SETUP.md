# Comfy Registry 注册指南

本文档说明如何将 ComfyUI PowerVision 插件注册到 Comfy Registry。

**官方文档**: https://docs.comfy.org/registry/publishing

## ✅ 已完成的工作

✅ 已创建 `pyproject.toml` 配置文件  
✅ 已创建 `.github/workflows/publish.yaml` GitHub Actions 工作流  
✅ 已创建 `LICENSE` MIT 许可证文件  
✅ 已创建 `web/docs/MyNode.md` 示例文档  
✅ PublisherId 已设置为 `whmc76`

## 📋 待完成步骤

### 1. 注册 Comfy Registry 账号

1. 访问 https://registry.comfy.org
2. 登录并确认您的 Publisher ID 为 `whmc76`（这是您用户名中 @ 符号后的部分）
3. 如果还没有账号，请先创建

参考文档: https://docs.comfy.org/registry/publishing#create-a-publisher

### 2. 创建 GitHub Registry API Key

1. 访问 https://registry.comfy.org 并登录
2. 点击您的 Publisher 账号（whmc76）
3. 创建 API 密钥用于 GitHub Actions 发布
4. **保存好这个密钥**，如果丢失需要重新创建

参考文档: https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing

### 3. 配置 GitHub Secrets

1. 进入您的 GitHub 仓库设置页面
2. 导航到 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **"New repository secret"**
4. 创建一个名为 `REGISTRY_ACCESS_TOKEN` 的 secret
5. 将步骤 2 中获取的 API 密钥作为值

参考文档: https://docs.comfy.org/registry/publishing#option-2-github-actions

### 4. 确认分支名称

检查 `.github/workflows/publish.yaml` 文件中的分支名称是否正确：

```yaml
on:
  push:
    branches:
      - main  # 如果您的仓库使用 master，请改为 master
```

### 5. 测试发布

有两种方式可以发布到 Registry：

#### 方式 A: 手动发布（使用 Comfy CLI）

```bash
# 安装 comfy-cli
pip install comfy-cli

# 发布到 Registry
comfy node publish
# 会提示输入 API Key
```

参考文档: https://docs.comfy.org/registry/publishing#option-1-comfy-cli

#### 方式 B: 自动发布（GitHub Actions）✨ 推荐

1. 更新 `pyproject.toml` 中的版本号
2. 提交并推送到主分支
3. GitHub Actions 会自动检测并发布

### 6. 验证发布成功

发布成功后，您的节点将在以下地址可见：
- https://registry.comfy.org/whmc76/comfyui-powervision

## 📝 发布新版本

当您需要发布新版本时：

1. 更新 `pyproject.toml` 中的 `version` 字段（使用语义化版本号）
2. 提交更改
3. 推送到主分支（或手动触发 `workflow_dispatch`）
4. GitHub Actions 会自动发布到 Registry

## 🚨 故障排除

### 问题：GitHub Actions 工作流失败

- ✅ 检查 `REGISTRY_ACCESS_TOKEN` 是否正确设置
- ✅ 确认 Publisher ID (`whmc76`) 与您的 Registry 账号匹配
- ✅ 查看工作流日志获取详细错误信息
- ✅ 确认 API 密钥有效且未过期

### 问题：无法发布到 Registry

- ✅ 确认 API 密钥有效
- ✅ 检查 `pyproject.toml` 格式是否正确
- ✅ 验证所有必需的字段都已填写
- ✅ 确认 License 文件存在

### 问题：API Key 复制时出现奇怪字符

⚠️ **重要提示**：在 Windows 上使用 CTRL+V 复制 API Key 时，可能会在末尾出现 `\x16` 字符。

**解决方案**：使用**右键点击粘贴**而不是 CTRL+V 来避免这个问题。

## 📞 联系支持

如有问题，请：
- 联系 Robin 的 Discord: `robinken`
- 加入 [ComfyUI Discord 服务器](https://discord.comfy.org)
- 查看 [官方文档](https://docs.comfy.org/registry/publishing)

## 📚 参考资源

- [Comfy Registry 官方文档](https://docs.comfy.org/registry/publishing)
- [ComfyUI 官方文档](https://github.com/comfyanonymous/ComfyUI)
- [pyproject.toml 规范](https://docs.comfy.org/registry/publishing#add-metadata)
- [GitHub Actions 参考](https://docs.comfy.org/registry/publishing#option-2-github-actions)

## ✅ 检查清单

在发布之前，请确认：

- [ ] Publisher ID 已正确设置为 `whmc76`
- [ ] 已在 Registry 创建 API Key
- [ ] 已在 GitHub 设置 `REGISTRY_ACCESS_TOKEN` secret
- [ ] `LICENSE` 文件存在
- [ ] `pyproject.toml` 版本号已更新
- [ ] README.md 已更新（可选）
- [ ] 所有更改已提交并推送到主分支
