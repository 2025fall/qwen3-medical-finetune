# Qwen3-1.7B 医学问答微调项目

基于Qwen3-1.7B模型的医学问答系统微调项目，支持全参数微调和LoRA微调两种方式。

## 🚀 项目特性

- **双微调策略**：支持全参数微调和LoRA微调
- **医学专业问答**：针对医学场景优化的问答系统
- **思考链生成**：模型输出包含显式思考过程
- **Web界面**：基于Gradio的交互式演示界面
- **批量推理**：支持批量预测和评估
- **自动评估**：包含多种评估指标的自动化评估系统

## 📋 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐)
- 内存: 16GB+ (全参数微调需要更多)
- 显存: 8GB+ (LoRA微调), 16GB+ (全参数微调)

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📁 项目结构

```
├── models/                    # 模型文件目录
│   ├── full/                 # 全参数微调模型
│   └── lora/                 # LoRA微调模型
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后数据
├── scripts/                  # 脚本文件
│   ├── prepare_data.py       # 数据预处理
│   ├── train_full.py         # 全参数微调训练
│   ├── train_lora.py         # LoRA微调训练
│   ├── batch_predict.py      # 批量预测
│   ├── demo_gradio.py        # Web演示界面
│   └── eval_auto.py          # 自动评估
├── requirements.txt          # 依赖包列表
└── README.md                # 项目说明
```

## 🚀 快速开始

### 1. 数据准备

```bash
python scripts/prepare_data.py
```

这将从ModelScope下载医学问答数据集，进行预处理、去重和分层切分。

### 2. 模型训练

#### LoRA微调（推荐）

```bash
python scripts/train_lora.py
```

#### 全参数微调

```bash
python scripts/train_full.py
```

### 3. 模型推理

#### 批量预测

```bash
python scripts/batch_predict.py
```

#### Web界面演示

```bash
python scripts/demo_gradio.py
```

访问 http://localhost:7860 进行交互式测试。

### 4. 模型评估

```bash
python scripts/eval_auto.py
```

## 📊 数据说明

项目使用`krisfu/delicate_medical_r1_data`数据集，包含：

- **训练集**: 80% 的数据用于模型训练
- **验证集**: 10% 的数据用于模型验证
- **测试集**: 10% 的数据用于模型测试
- **黄金标准集**: 从验证集和测试集中精选的高质量样本
- **红队测试集**: 人工构建的高风险测试用例

数据格式遵循指令微调的标准格式：
```json
{
    "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    "input": "用户问题",
    "output": "<think>思考过程</think>\n医学建议",
    "meta": {
        "source": "数据来源",
        "risk_level": "风险等级",
        "specialty": "专科领域",
        "complexity": "复杂度",
        "lang_style": "语言风格",
        "is_deidentified": true
    }
}
```

## 🎯 评估指标

自动评估系统包含以下指标：

- **思考链覆盖率**: 输出包含`<think>`标签的比例
- **紧急信号覆盖率**: 包含就医指征提示的比例
- **风险处方率**: 包含不当药物建议的比例

## 🔧 配置说明

### 训练参数

#### LoRA微调参数
- `r=64`: LoRA秩
- `lora_alpha=128`: LoRA缩放参数
- `lora_dropout=0.1`: LoRA dropout率
- `learning_rate=2e-4`: 学习率
- `num_train_epochs=3`: 训练轮数

#### 全参数微调参数
- `learning_rate=5e-6`: 学习率
- `num_train_epochs=1`: 训练轮数
- `gradient_accumulation_steps=8`: 梯度累积步数

### 模型配置

在脚本中可以切换使用不同的模型：

```python
# 使用全参数微调模型
MODEL_DIR = "models/full/final_model"
IS_LORA = False

# 使用LoRA微调模型
BASE_DIR = "models/Qwen/Qwen3-1.7B"
ADAPTER_DIR = "models/lora/final_lora"
IS_LORA = True
```

## ⚠️ 注意事项

1. **医学免责声明**: 本系统仅用于研究和学习目的，不应用于实际医疗诊断
2. **数据脱敏**: 所有训练数据已进行脱敏处理
3. **紧急情况**: 遇到紧急医疗情况请立即就医
4. **资源要求**: 全参数微调需要大量计算资源，建议优先使用LoRA微调

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Qwen团队](https://github.com/QwenLM/Qwen) 提供的优秀基础模型
- [ModelScope](https://modelscope.cn/) 提供的数据集和模型下载服务
- 医学领域的专家和研究者们提供的宝贵数据

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。
