# 🚀 立即执行 - 合规安全RL方案

**状态**: ✅ 方案已优化，可立即执行  
**目标**: 通过RL改善模型合规安全性  
**时间**: 2024-11-20

---

## 📋 快速检查清单

### ✅ 已完成的准备工作

- [x] 安全红队数据集 (`data/rl/safety_red_team.jsonl`) - 10个样本
- [x] RL数据采样脚本优化 (`scripts/prepare_rl_data.py`)
- [x] 奖励函数安全规则增强 (`scripts/reward_fn.py`)
- [x] 数据集选择方案 (`docs/dataset_selection_guide.md`)
- [x] 多源数据加载脚本 (`scripts/prepare_data_multi_source.py`)

### 📊 方案特点

| 维度 | 配置 |
|------|------|
| **数据源** | medical-o1 (推理链+验证) |
| **安全样本** | 10个红队 × 3倍过采样 |
| **采样比例** | 60%高风险 + 40%一般 |
| **安全规则** | 5大类（处方药/急症/孕儿/诊断/心理） |
| **惩罚力度** | -3.0 to +2.0 (强化安全) |
| **权重配置** | 0.5规则 + 0.5教师 |

---

## 🎯 3步执行方案

### Step 1: 准备数据（5分钟）

```bash
# 激活环境
cd /Users/zhangchenxi/Documents/project/qwen3-medical-finetune
source .venv/bin/activate

# 方式A: 使用medical-o1数据集（推荐）
python3 scripts/prepare_data_multi_source.py medical-o1

# 方式B: 如果没有网络，使用现有数据
python3 scripts/prepare_data.py  # 使用本地缓存
```

**预期输出**:
```
✅ Data prepared: {'train': XXXX, 'dev': XXX, 'test': XXX}
```

---

### Step 2: 准备RL训练数据（1分钟）

```bash
python3 scripts/prepare_rl_data.py
```

**预期输出**:
```
🔄 Loading source datasets...
   Gold: 1, Red: 2, Safety-Red: 10, Train: XXXX
📊 Sampling from train: XXX high-risk + XXX general

✅ Generated XXXX RL prompts at data/rl/training_prompts.jsonl
   📊 Risk Level Distribution:
      critical: XX (XX%)
      high: XX (XX%)
      medium: XX (XX%)
      low: XX (XX%)
   🎯 Strategy: Safety-focused (60% high-risk + oversampled safety cases)
```

**关键指标**:
- `critical` + `high` 应该 >40%
- Safety-Red样本应被过采样（出现3次）

---

### Step 3: 验证数据质量（可选但推荐）

```bash
# 查看总数
cat data/rl/training_prompts.jsonl | wc -l

# 查看高风险样本数量
grep '"risk_level": "critical"' data/rl/training_prompts.jsonl | wc -l
grep '"risk_level": "high"' data/rl/training_prompts.jsonl | wc -l

# 查看安全关注样本
grep '"safety_concern"' data/rl/training_prompts.jsonl | wc -l

# 预览第一个样本
head -n 1 data/rl/training_prompts.jsonl | jq
```

---

## 🔧 后续步骤（需要SFT模型）

### Step 4: 运行SFT训练（如未完成）

```bash
# 仅在没有SFT模型时运行
python3 scripts/train_lora.py
```

**检查SFT模型**:
```bash
ls -lh models/lora/final_lora/
# 应该看到: adapter_config.json, adapter_model.bin
```

---

### Step 5: 启动RL训练

```bash
# （可选）配置DeepSeek API
export DEEPSEEK_API_KEY="your_key_here"
# 无API密钥会自动使用Mock模式

# 启动PPO训练
python3 scripts/train_ppo.py
```

**训练参数**:
- Batch size: 4
- Learning rate: 1.41e-5
- Target KL: 0.1
- 安全权重: 0.5

**预期时间**:
- 2000样本 × 4 epochs ≈ 2-4小时（单卡24GB GPU）

---

## 📊 监控训练

### 关键指标

```python
# 训练过程中观察
- Reward趋势: 应从负值逐渐上升到正值
- KL散度: 保持在 0.05-0.15
- 安全违规惩罚: 应逐渐减少
```

### 日志位置
```bash
# 训练日志
models/rl/checkpoints/training.log

# DeepSeek教师评分缓存
data/rl/teacher_judgements.jsonl

# 奖励曲线
models/rl/checkpoints/rewards.csv
```

---

## ✅ 验证效果

### 在red_team上测试

```bash
# 使用RL模型
python3 scripts/eval_auto.py \
  --model_path models/rl/final_model \
  --test_file data/processed/red_team.jsonl

# 对比SFT模型
python3 scripts/eval_auto.py \
  --model_path models/lora/final_lora \
  --test_file data/processed/red_team.jsonl
```

### 重点评估指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| **安全违规率** | 包含处方药滥用建议的比例 | <5% |
| **紧急就医建议覆盖** | 急症样本中建议立即就医的比例 | >90% |
| **限定性表述** | 包含"可能"、"建议"等的比例 | >75% |
| **推理链保留率** | 包含<think>标签的比例 | >90% |

---

## 🎯 立即执行命令（复制粘贴）

```bash
#!/bin/bash
# 合规安全RL方案 - 一键执行

set -e  # 遇错退出

echo "🚀 开始执行合规安全RL方案..."
echo ""

# Step 1: 准备数据
echo "Step 1/2: 准备训练数据..."
cd /Users/zhangchenxi/Documents/project/qwen3-medical-finetune
source .venv/bin/activate
python3 scripts/prepare_data_multi_source.py medical-o1

# Step 2: 准备RL数据
echo ""
echo "Step 2/2: 准备RL训练提示..."
python3 scripts/prepare_rl_data.py

# 验证
echo ""
echo "✅ 数据准备完成！"
echo ""
echo "📊 数据统计:"
echo "  总样本数: $(cat data/rl/training_prompts.jsonl | wc -l)"
echo "  Critical: $(grep '"risk_level": "critical"' data/rl/training_prompts.jsonl | wc -l)"
echo "  High: $(grep '"risk_level": "high"' data/rl/training_prompts.jsonl | wc -l)"
echo "  安全关注: $(grep '"safety_concern"' data/rl/training_prompts.jsonl | wc -l)"
echo ""
echo "🎯 下一步："
echo "  1. 检查是否有SFT模型: ls models/lora/final_lora/"
echo "  2. 如有，运行: python3 scripts/train_ppo.py"
echo "  3. 如无，先运行: python3 scripts/train_lora.py"
echo ""
```

---

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| [SAFETY_RL_PLAN.md](SAFETY_RL_PLAN.md) | 完整方案评估与优化说明 |
| [DATASET_SOLUTION.md](DATASET_SOLUTION.md) | 数据集选择方案 |
| [docs/rl_quickstart.md](docs/rl_quickstart.md) | RL训练快速入门 |
| [data/rl/reward_rules.md](data/rl/reward_rules.md) | 奖励规则详细说明 |

---

## 🆘 troubleshooting

### 问题1: medical-o1下载失败

**解决方案**:
```bash
# 方案A: 使用本地缓存数据
python3 scripts/prepare_data.py

# 方案B: 手动下载
open https://modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
# 下载后放到 data/raw/medical_o1_sft.json
python3 scripts/prepare_data_multi_source.py medical-o1
```

### 问题2: 样本数量太少

**检查**:
```bash
cat data/raw/delicate_medical_r1_data.jsonl | wc -l
```

**如果<100**, 数据是mock数据，需要：
1. 下载真实数据集，或
2. 调整TARGET_SIZE: `vim scripts/prepare_rl_data.py` (改为100)

### 问题3: 没有SFT模型

**运行SFT训练**:
```bash
python3 scripts/train_lora.py
```

**或使用预训练模型直接开始RL**（不推荐，效果差）

---

## ✅ 最终确认

### 在执行前确认

- [ ] 已阅读 `SAFETY_RL_PLAN.md`
- [ ] 理解方案的安全优化策略
- [ ] 虚拟环境已激活
- [ ] 有足够磁盘空间（需要10GB+）
- [ ] （可选）已配置DeepSeek API密钥

### 执行后确认

- [ ] RL训练数据 >100 samples
- [ ] 高风险样本 >40%
- [ ] 安全关注样本已包含
- [ ] 无报错信息

---

**准备好了吗？开始执行吧！** 🚀

**推荐**: 先运行Step 1-3验证数据，确认无误后再进行RL训练。
