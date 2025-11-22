# Qwen3-1.7B 医学问答微调项目

基于Qwen3-1.7B模型的医学问答系统微调项目，支持全参数微调和LoRA微调两种方式。

## 🚀 项目特性

- **三阶段训练**：支持全参数微调、LoRA微调和RL强化学习
- **医学专业问答**：针对医学场景优化的问答系统
- **思考链生成**：模型输出包含显式思考过程（`<think>`标签）
- **DeepSeek教师辅助**：RL阶段使用DeepSeek模型作为奖励评分教师
- **多维奖励函数**：结合规则奖励和AI教师评分的组合奖励系统
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
│   ├── lora/                 # LoRA微调模型
│   └── rl/                   # RL训练模型（NEW）
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据（SFT）
│   └── rl/                   # RL训练数据（NEW）
│       ├── training_prompts.jsonl
│       ├── teacher_judgements.jsonl
│       └── reward_rules.md
├── scripts/                  # 脚本文件
│   ├── prepare_data.py       # 数据预处理（SFT）
│   ├── prepare_rl_data.py    # RL数据准备（NEW）
│   ├── train_full.py         # 全参数微调训练
│   ├── train_lora.py         # LoRA微调训练
│   ├── train_ppo.py          # PPO强化学习训练（NEW）
│   ├── deepseek_teacher.py   # DeepSeek教师评分（NEW）
│   ├── reward_fn.py          # 奖励函数（NEW）
│   ├── batch_predict.py      # 批量预测
│   ├── demo_gradio.py        # Web演示界面
│   └── eval_auto.py          # 自动评估
├── docs/                     # 文档目录
│   ├── rl_quickstart.md      # RL快速入门（NEW）
│   └── ...
├── reports/rl_stage/         # RL阶段报告（NEW）
│   ├── feasibility_report.md
│   └── implementation_progress.md
├── requirements.txt          # 依赖包列表
└── README.md                # 项目说明
```

## 🚀 快速开始

### 阶段1: SFT（监督微调）

#### 1.1 数据准备

**选择数据集（推荐用于RL）**:

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行单源数据准备（基于 FreedomIntelligence/medical-o1-reasoning-SFT）
python scripts/prepare_data.py
```

**推荐数据集**：
- **FreedomIntelligence/medical-o1-reasoning-SFT** (最适合RL) - 包含完整推理链
- **krisfu/delicate_medical_r1_data** (原始) - 已适配代码

详见：[数据集选择指南](docs/dataset_selection_guide.md)

#### 1.2 SFT训练

**LoRA微调（推荐）**

```bash
python scripts/train_lora.py
```

**全参数微调**

```bash
python scripts/train_full.py
```

### 阶段2: RL（强化学习）🆕

#### 2.1 准备RL数据

```bash
# 生成RL训练提示（优先高风险样本）
python scripts/prepare_rl_data.py
```

#### 2.2 配置DeepSeek教师（可选）

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your_api_key_here"

# 测试教师模块
python scripts/deepseek_teacher.py --limit 5
```

**注**: 如果没有API密钥，系统会自动使用Mock模式进行训练。

#### 2.3 PPO训练

```bash
python scripts/train_ppo.py
```

**详细教程**: 参见 [docs/rl_quickstart.md](docs/rl_quickstart.md)

### 阶段3: 模型推理

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

### SFT阶段数据
- **训练集** (80%): 用于监督微调
- **验证集** (10%): 用于模型验证
- **测试集** (10%): 用于模型测试
- **黄金标准集**: 从验证集和测试集中精选的高质量样本
- **红队测试集**: 人工构建的高风险测试用例

### RL阶段数据🆕
- **training_prompts.jsonl**: RL训练提示（优先高风险+高质量样本）
- **teacher_judgements.jsonl**: DeepSeek教师评分缓存
- **reward_rules.md**: 奖励函数配置文档

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

### SFT阶段评估
- **思考链覆盖率**: 输出包含`<think>`标签的比例
- **紧急信号覆盖率**: 包含就医指征提示的比例
- **风险处方率**: 包含不当药物建议的比例

### RL阶段奖励🆕

**组合奖励公式**:
```
R = 0.6 × R_rules + 0.4 × R_teacher
```

**规则奖励 (R_rules)**:
- 格式奖励: `<think>`标签存在 (+0.2)
- 内容奖励: 包含"建议"、"诊断"等关键词 (+0.1)
- 长度惩罚: 回答过短 (-0.5)

**教师奖励 (R_teacher)** - 由DeepSeek评分:
- 安全性 (40%): 是否有误导、漏诊
- 逻辑性 (30%): 推理链是否严密
- 完整性 (20%): 是否覆盖鉴别诊断
- 同理心 (10%): 语气是否恰当

详见: [data/rl/reward_rules.md](data/rl/reward_rules.md)

## 🔧 配置说明

### 三阶段训练流程🆕

```
数据准备 → SFT训练 → RL训练
   ↓          ↓         ↓
  JSONL   LoRA模型   RL模型
```

1. **SFT阶段**: 学习基础医学知识和思考链格式
2. **RL阶段**: 通过强化学习优化安全性和逻辑性
3. **评估**: 在gold_set和red_team上对比SFT vs RL性能

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

#### PPO训练参数🆕
- `learning_rate=1.41e-5`: 学习率
- `batch_size=4`: 批次大小
- `mini_batch_size=1`: PPO mini-batch
- `target_kl=0.1`: KL散度阈值
- `ppo_epochs=4`: 每批数据的训练轮数

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

## 📚 相关文档

- **RL快速入门**: [docs/rl_quickstart.md](docs/rl_quickstart.md)
- **RL可行性分析**: [reports/rl_stage/feasibility_report.md](reports/rl_stage/feasibility_report.md)
- **RL实施进度**: [reports/rl_stage/implementation_progress.md](reports/rl_stage/implementation_progress.md)
- **奖励规则**: [data/rl/reward_rules.md](data/rl/reward_rules.md)

## 🙏 致谢

- [Qwen团队](https://github.com/QwenLM/Qwen) 提供的优秀基础模型
- [ModelScope](https://modelscope.cn/) 提供的数据集和模型下载服务
- [TRL库](https://github.com/huggingface/trl) 提供的强化学习框架
- [DeepSeek](https://www.deepseek.com/) 提供的AI评分服务
- 医学领域的专家和研究者们提供的宝贵数据

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。
