# 贡献指南

感谢您对Qwen3医学问答微调项目的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题

如果您发现了bug或有改进建议，请：

1. 检查现有的[Issues](https://github.com/your-username/qwen3-medical-finetune/issues)确保问题未被报告
2. 创建新的Issue，包含：
   - 清晰的问题描述
   - 复现步骤
   - 预期行为vs实际行为
   - 环境信息（Python版本、依赖版本等）

### 提交代码

1. **Fork项目**
   ```bash
   git clone https://github.com/your-username/qwen3-medical-finetune.git
   cd qwen3-medical-finetune
   ```

2. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **安装开发环境**
   ```bash
   pip install -r requirements.txt
   ```

4. **进行更改**
   - 遵循现有的代码风格
   - 添加适当的注释
   - 确保代码通过测试

5. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```

6. **推送并创建Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 代码规范

### Python代码风格

- 使用4个空格缩进
- 遵循PEP 8规范
- 函数和类添加docstring
- 变量名使用snake_case，类名使用PascalCase

### 提交信息格式

使用语义化提交信息：

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat: 添加LoRA微调支持
fix: 修复批量预测中的内存泄漏问题
docs: 更新README中的安装说明
```

## 🧪 测试

在提交代码前，请确保：

1. **运行现有测试**
   ```bash
   python scripts/eval_auto.py
   ```

2. **测试新功能**
   - 添加相应的测试用例
   - 确保新功能不会破坏现有功能

3. **代码质量检查**
   ```bash
   # 使用flake8检查代码风格
   pip install flake8
   flake8 scripts/
   ```

## 📚 文档

- 更新相关文档（README.md、代码注释等）
- 确保新功能有清晰的使用说明
- 添加示例代码

## 🏗️ 项目结构

贡献时请遵循现有的项目结构：

```
├── scripts/           # 主要脚本
├── data/             # 数据相关
├── models/           # 模型文件
├── docs/             # 文档（如有）
└── tests/            # 测试文件
```

## 🔍 代码审查

所有提交的代码都会经过审查，审查要点：

- 代码质量和风格
- 功能正确性
- 性能影响
- 安全性
- 文档完整性

## 🎯 贡献领域

我们特别欢迎以下领域的贡献：

- **数据处理**: 改进数据预处理流程
- **模型优化**: 提升训练效率和效果
- **评估指标**: 添加新的评估方法
- **用户界面**: 改进Web演示界面
- **文档**: 完善使用说明和API文档
- **测试**: 增加测试覆盖度

## 💡 想法和建议

如果您有想法但不确定如何实现，可以：

1. 在Issues中创建"Discussion"标签的issue
2. 描述您的想法和预期效果
3. 社区会讨论并提供实现建议

## 📞 联系

如有疑问，可以通过以下方式联系：

- GitHub Issues
- 项目维护者的联系方式

## 🙏 感谢

感谢所有为项目做出贡献的开发者！您的贡献让项目变得更好。

---

**注意**: 请确保您的贡献符合项目的开源许可证要求。
