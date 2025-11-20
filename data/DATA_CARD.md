# DATA CARD

**Sources**: HuatuoGPT-o1医学推理数据集（推荐用于RL）  
**Use**: Research & model fine-tuning (medical Q&A); de-identified.  
**Schema**: instruction / input / output (+ meta: source, specialty, risk_level, complexity, lang_style, is_deidentified)

## Splits
- Train: 13731
- Dev:   1716
- Test:  1717
- Gold:  200
- Red Team: 2

## Style guide for <think>
（写作规范）主诉解析→可能性与鉴别→红旗/风险→建议与不确定性→就医指征；禁止杜撰检查/处方剂量。

## Caveats
- specialty 多为 unknown（后续逐步补标）
- risk_high 样本占比有限，建议持续扩充

## Data Sources
- medical-o1: HuatuoGPT-o1医学推理数据集（推荐用于RL）
- delicate-medical: 精致医疗r1数据（原始使用）
- datatang-qa: 数据堂203k医疗多轮问答
- chinese-dialogue: 中文医疗对话数据
