# ModelScope医疗数据集选择指南

**RL阶段数据集推荐** - 2024-11-20

> 注意：当前数据准备脚本已回退为单源版本 `scripts/prepare_data.py`（基于 medical-o1），文中出现的 `prepare_data_multi_source.py` 可替换为 `prepare_data.py` 运行。

---

## 🎯 RL训练的数据需求

RL阶段需要的数据特点：
1. **高质量**：答案准确，逻辑清晰
2. **包含推理链**：最好有<think>标签或推理过程
3. **风险多样性**：包含高风险、低风险场景
4. **规模适中**：建议2000-10000样本
5. **可验证性**：能够评估答案质量

---

## 📊 推荐数据集对比

| 数据集 | 规模 | 推理链 | 质量 | RL适用性 | 优先级 |
|-------|------|--------|------|----------|--------|
| **medical-o1-reasoning-SFT** | 中等 | ✅ 完整 | ⭐⭐⭐⭐⭐ | 🔥🔥🔥 | 1 |
| **delicate_medical_r1_data** | 未知 | ✅ 有 | ⭐⭐⭐⭐ | 🔥🔥 | 2 |
| **Datatang 203k问答** | 203k | ❌ 无 | ⭐⭐⭐ | 🔥 | 3 |
| **Chinese-medical-dialogue** | 大 | ❌ 无 | ⭐⭐⭐ | 🔥 | 4 |

---

## ⭐ 1. FreedomIntelligence/medical-o1-reasoning-SFT（最推荐）

### 为什么最适合RL？

#### ✅ 核心优势
1. **包含完整推理链**
   - 基于GPT-4o生成
   - 每个样本都有reasoning过程
   - 符合你们的<think>标签格式

2. **可验证性强**
   - 基于verifiable medical problems
   - 有医学验证器验证正确性
   - 适合作为RL的奖励信号

3. **高质量标注**
   - 来自HuatuoGPT-o1项目（arXiv:2412.18925）
   - 学术团队维护
   - 持续更新（最近更新：2025-04-22）

4. **规模适中**
   - 247MB数据量
   - 适合单卡RL训练
   - 不会过拟合

### 📦 数据格式
```json
{
  "question": "患者主诉...",
  "reasoning": "首先分析症状...然后考虑鉴别诊断...最后给出建议",
  "answer": "根据以上分析，建议...",
  "specialty": "internal_medicine"
}
```

### 🚀 使用方法

#### 方法1：使用新的多源脚本（推荐）
```bash
# 自动加载medical-o1数据集
python3 scripts/prepare_data_multi_source.py

# 或指定数据源
python3 scripts/prepare_data_multi_source.py medical-o1
```

#### 方法2：手动下载
```bash
# 1. 访问ModelScope
open https://modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

# 2. 下载文件
# - medical_o1_sft.json (纯医疗数据)
# - medical_o1_sft_mix.json (医疗+通用指令)

# 3. 放到项目中
mv medical_o1_sft.json data/raw/

# 4. 运行数据准备
python3 scripts/prepare_data_multi_source.py medical-o1
```

### 📈 数据统计
- **下载量**: 5,699+
- **点赞数**: 20
- **许可**: Apache License 2.0
- **语言**: 中文
- **更新**: 活跃维护中

### 🔗 相关资源
- **数据集**: https://modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
- **论文**: https://arxiv.org/abs/2412.18925
- **GitHub**: https://github.com/FreedomIntelligence/HuatuoGPT-o1

---

## 📌 2. krisfu/delicate_medical_r1_data（备选）

### 特点
- ✅ 你的代码已适配
- ✅ 包含<think>标签
- ⚠️ 当前版本兼容问题

### 解决方案
```bash
# 手动下载并放到指定位置
mkdir -p data/raw
# 下载后重命名为: delicate_medical_r1_data.jsonl
```

---

## 💡 3. DatatangBeijing/203029Groups-ChineseMedicalQuestionAnsweringData

### 适用场景
- 需要**超大规模**数据时
- 多轮对话RL训练
- 疾病分类任务

### 特点
- ✅ 规模大（203k对话）
- ✅ 真实医患对话
- ❌ 无推理链（需要自己生成）
- ❌ 需要额外处理多轮对话

### 使用建议
可作为**补充数据源**，与medical-o1混合使用：
```bash
python3 scripts/prepare_data_multi_source.py medical-o1,datatang-qa
```

---

## 🚀 快速开始指南

### Step 1: 选择数据集
```bash
# 推荐：使用medical-o1（最适合RL）
export DATASET_CHOICE="medical-o1"
```

### Step 2: 运行多源数据准备脚本
```bash
source .venv/bin/activate
python3 scripts/prepare_data_multi_source.py $DATASET_CHOICE
```

### Step 3: 检查生成的数据
```bash
# 查看数据统计
cat data/DATA_CARD.md

# 查看样本
head -n 1 data/processed/train.jsonl | jq

# 确认推理链格式
grep "<think>" data/processed/train.jsonl | head -n 1
```

### Step 4: 进入RL训练
```bash
# 准备RL数据
python3 scripts/prepare_rl_data.py

# 训练PPO
python3 scripts/train_ppo.py
```

---

## 🔧 多数据源混合策略

### 策略1: 主数据源 + 补充
```bash
# medical-o1为主，datatang为补充
python3 scripts/prepare_data_multi_source.py medical-o1,datatang-qa
```

### 策略2: 按优先级自动加载
```bash
# 按优先级尝试，直到成功加载一个
python3 scripts/prepare_data_multi_source.py
```

### 策略3: 仅使用特定数据源
```bash
# 仅使用delicate-medical
python3 scripts/prepare_data_multi_source.py delicate-medical
```

---

## ❓ 常见问题

### Q1: 如何手动下载medical-o1数据集？
```bash
# 1. 访问页面
open https://modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

# 2. 点击"数据集文件"标签
# 3. 下载 medical_o1_sft.json
# 4. 移动到项目
mv ~/Downloads/medical_o1_sft.json data/raw/
```

### Q2: 数据集下载太慢怎么办？
```bash
# 使用git clone方式
cd data/raw
git lfs install
git clone https://www.modelscope.cn/datasets/FreedomIntelligence/medical-o1-reasoning-SFT.git
```

### Q3: 如何验证数据质量？
```bash
# 运行数据准备后检查
python3 -c "
import json
with open('data/processed/train.jsonl') as f:
    sample = json.loads(f.readline())
    print('Question:', sample['input'][:50])
    print('Has Think:', '<think>' in sample['output'])
    print('Meta:', sample['meta'])
"
```

### Q4: 可以同时使用多个数据集吗？
可以！使用逗号分隔：
```bash
python3 scripts/prepare_data_multi_source.py medical-o1,delicate-medical,datatang-qa
```

---

## 📋 数据集对比详表

| 维度 | medical-o1 | delicate_medical | datatang | chinese-dialogue |
|-----|------------|------------------|----------|------------------|
| **推理链质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ |
| **数据规模** | 中等 | 未知 | 超大 | 大 |
| **RL适用性** | 🔥🔥🔥 | 🔥🔥 | 🔥 | 🔥 |
| **下载难度** | 容易 | 中等 | 容易 | 容易 |
| **维护状态** | 活跃 | 未知 | 稳定 | 稳定 |
| **许可协议** | Apache 2.0 | 未知 | Apache 2.0 | Apache 2.0 |
| **是否需要加工** | 否 | 否 | 是（需生成推理链） | 是 |

---

## 🎯 推荐决策树

```
需要RL训练数据?
    │
    ├─ 优先质量 + 推理链?
    │   └─ ✅ medical-o1-reasoning-SFT
    │
    ├─ 需要超大规模?
    │   └─ datatang-203k + medical-o1混合
    │
    ├─ 已有代码适配?
    │   └─ delicate_medical_r1_data
    │
    └─ 都不确定?
        └─ 先用medical-o1，后续可扩展
```

---

## 📞 获取帮助

如遇问题：
1. 查看 `RL_SETUP_COMPLETE.md`
2. 检查 `data/DATA_CARD.md`
3. 运行测试： `python3 scripts/prepare_data_multi_source.py --help`

---

**最后更新**: 2024-11-20  
**维护者**: 项目团队
