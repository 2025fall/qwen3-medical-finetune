# 强化学习奖励规则 (Reward Rules)

本文档定义了 Qwen3-Medical RL 阶段的奖励函数构成与计算逻辑。

## 1. 总分公式

最终奖励 $R$ 由规则分 $R_{rules}$ 和教师分 $R_{teacher}$ 线性组合而成：

$$ R = \alpha \cdot R_{teacher} + \beta \cdot R_{rules} + \gamma \cdot P_{penalty} $$

当前参数设置：
- $\alpha$ (Teacher Weight): **0.4**
- $\beta$ (Rule Weight): **0.6**
- $\gamma$ (Penalty Weight): **1.0** (用于严重的格式错误或安全违规)

---

## 2. 教师分 ($R_{teacher}$)

由 `scripts/deepseek_teacher.py` 调用 DeepSeek 模型打分，范围 `[0, 1]`。

| 维度 | 权重 | 说明 |
| :--- | :--- | :--- |
| **安全性 (Safety)** | 40% | 是否包含有害建议、遗漏紧急指征。 |
| **逻辑性 (Reasoning)** | 30% | 推理链 `<think>` 是否严密，因果关系是否成立。 |
| **完整性 (Completeness)** | 20% | 是否覆盖鉴别诊断、治疗建议、就医指导。 |
| **同理心 (Empathy)** | 10% | 语气是否恰当，是否体现人文关怀。 |

*注：若 Safety 得分为 0，则总分强制归 0。*

---

## 3. 规则分 ($R_{rules}$)

基于正则表达式和硬性指标计算，范围 `[-1, 1]`。

### 3.1 格式奖励 (Format Rewards)
- **Think Tag的存在**: 包含 `<think>...</think>` 且内容非空 -> **+0.2**
- **结构清晰**: 包含 "建议"、"指征"、"诊断" 等关键词 -> **+0.1**

### 3.2 长度惩罚 (Length Penalties)
- **过短回答**: Output (不含 think) < 20 tokens -> **-0.5**
- **思考过短**: Think < 10 tokens -> **-0.2**
- **重复啰嗦**: 检测到连续重复语句 -> **-0.3**

### 3.3 医疗安全规则 (Safety Rules)
- **拒绝回答逃避**: 出现 "我无法回答"、"我只是AI" (但在医疗微调场景下，我们希望它在安全范围内回答) -> **-0.1** (视情况调整)
- **缺失免责声明**: 未提及 "请咨询医生" 或 "仅供参考" -> **-0.1**

---

## 4. 训练与推理配置

### 4.1 训练 (PPO)
- **KL 惩罚系数**: 0.05 (动态调整)
- **Clip Range**: 0.2
- **Mini-batch Size**: 4

### 4.2 教师提示模板 (Version 1.0)
参见 `scripts/deepseek_teacher.py` 中的 `JUDGE_TEMPLATE`。
