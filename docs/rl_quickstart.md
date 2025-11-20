# RL训练快速入门

本文档提供Qwen3-Medical项目RL阶段的快速启动指南。

---

## 前置要求

### 硬件
- GPU: ≥24GB显存 (推荐A5000/4090)
- 或使用gradient checkpointing在16GB GPU上运行

### 软件
- Python 3.13
- 已激活虚拟环境
- 所有依赖已安装（见`requirements.txt`）

---

## 快速启动（3步）

### Step 1: 准备数据

```bash
# 激活虚拟环境
source .venv/bin/activate

# 生成基础数据集（SFT数据）
python3 scripts/prepare_data.py

# 生成RL训练提示（优先高风险样本）
python3 scripts/prepare_rl_data.py
```

**输出**:
- `data/processed/train.jsonl`, `dev.jsonl`, `test.jsonl`
- `data/processed/gold_set.jsonl` (高质量样本)
- `data/processed/red_team.jsonl` (高风险样本)
- `data/rl/training_prompts.jsonl` (RL训练用)

---

### Step 2: 配置DeepSeek教师（可选）

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your_api_key_here"

# 测试教师模块（Mock模式）
python3 scripts/deepseek_teacher.py --input data/processed/gold_set.jsonl --limit 5
```

**如果没有API密钥**: 脚本会自动使用Mock模式进行测试。

---

### Step 3: 运行RL训练

```bash
# 确保SFT模型已训练完成
# 默认路径: models/lora/final_lora

# 启动PPO训练
python3 scripts/train_ppo.py
```

**训练配置**:
- Batch size: 4
- Mini-batch: 1
- Learning rate: 1.41e-5
- Target KL: 0.1
- Epochs: 自动（基于数据量）

**输出**:
- Checkpoints: `models/rl/checkpoints/step_*`
- Final model: `models/rl/checkpoints/final_rl_model`

---

## 数据流程图

```
原始数据 (data/raw/*.jsonl)
    ↓
[prepare_data.py] 清洗、去重、分层切分
    ↓
处理后数据 (data/processed/*.jsonl)
    ↓
[prepare_rl_data.py] 采样高风险+高质量提示
    ↓
RL训练提示 (data/rl/training_prompts.jsonl)
    ↓
[train_ppo.py] PPO训练
    ├─ [deepseek_teacher.py] 教师打分
    ├─ [reward_fn.py] 规则奖励
    └─ 组合奖励 → 策略更新
    ↓
RL模型 (models/rl/checkpoints/final_rl_model)
```

---

## 奖励函数配置

奖励公式在`data/rl/reward_rules.md`中定义：

```
R = 0.6 × R_rules + 0.4 × R_teacher

R_rules: 规则奖励（格式、长度、关键词）
R_teacher: DeepSeek教师评分（安全性、逻辑性、完整性、同理心）
```

**调整权重**: 修改`scripts/reward_fn.py`中的`compute_rewards`方法。

---

## 常见问题

### Q1: 如何只用规则奖励，不调用DeepSeek？
在`scripts/reward_fn.py`中修改组合公式：
```python
total = 1.0 * r + 0.0 * t  # 全规则奖励
```

### Q2: 数据太少怎么办？
- 最小推荐: 500-1000样本
- 当前示例数据仅用于流程验证
- 从ModelScope下载完整数据集: `krisfu/delicate_medical_r1_data`

### Q3: GPU显存不足？
在`scripts/train_ppo.py`中调整：
```python
config = PPOConfig(
    batch_size=2,           # 减小batch
    mini_batch_size=1,
    gradient_accumulation_steps=2  # 增加累积
)
```

并添加gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```

### Q4: 如何评估RL模型？
```bash
python3 scripts/eval_auto.py --model_path models/rl/checkpoints/final_rl_model
```

---

## 进阶选项

### 自定义教师提示
编辑`scripts/deepseek_teacher.py`中的`JUDGE_TEMPLATE`

### 调整PPO超参数
编辑`scripts/train_ppo.py`中的`PPOConfig`:
- `learning_rate`: 学习率
- `target_kl`: KL散度阈值（控制策略漂移）
- `ppo_epochs`: 每批数据的训练轮数

### 查看训练日志
```bash
# 查看DeepSeek教师调用记录
cat data/rl/teacher_judgements.jsonl | jq

# 查看奖励分布（需安装pandas）
python3 -c "
import pandas as pd
data = pd.read_json('data/rl/teacher_judgements.jsonl', lines=True)
print(data['judgement'].apply(lambda x: x['overall_score']).describe())
"
```

---

## 完整训练示例

```bash
#!/bin/bash
# full_rl_pipeline.sh

# 1. 环境准备
source .venv/bin/activate
export DEEPSEEK_API_KEY="sk-..."

# 2. 数据准备
python3 scripts/prepare_data.py
python3 scripts/prepare_rl_data.py

# 3. SFT训练（如果还没做）
# python3 scripts/train_lora.py --epochs 3 --batch_size 4

# 4. 测试教师模块
python3 scripts/deepseek_teacher.py --input data/processed/gold_set.jsonl --limit 10

# 5. RL训练
python3 scripts/train_ppo.py

# 6. 评估
python3 scripts/eval_auto.py --model_path models/rl/checkpoints/final_rl_model

echo "✅ RL训练流程完成！"
```

---

## 参考文档

- [可行性分析](../reports/rl_stage/feasibility_report.md)
- [实施进度](../reports/rl_stage/implementation_progress.md)
- [奖励规则](../data/rl/reward_rules.md)
- [数据说明](../data/DATA_CARD.md)

---

**最后更新**: 2024-11-20  
**维护者**: 项目团队
