# RL 微调阶段可行性分析与方案

## 1. 项目背景概述
- **数据准备**：`scripts/prepare_data.py` 已完成下载、清洗、去重与分层切分，并在 `build_gold_and_red()` 中生成 `data/processed/gold_set.jsonl` 和 `data/processed/red_team.jsonl`，同时更新 `DATA_CARD.md`。后续将通过 `scripts/deepseek_teacher.py` 让 DeepSeek 教师定期抽检 gold/red，形成“生成 → 教师复核”的闭环。
- **监督微调能力**：`scripts/train_lora.py`、`scripts/train_full.py` 已交付 LoRA 与全参数两套流程，可输出显式 `<think>` 推理链，权重落地在 `models/lora/`、`models/full/`，可直接用作 RL 策略与 reference。
- **评估与推理工具**：`scripts/eval_auto.py` 默认评估 `dev/test/gold_set/red_team` 四个 split；`scripts/batch_predict.py` 与 `scripts/demo_gradio.py` 支持批量及交互式推理，可在引入 RL 后对比 SFT 与 RL 版本。
- **工程依赖**：目前基于 `transformers`、`peft`、`datasets`，尚未引入 `trl`。脚本模块化程度高，便于添加 DeepSeek 教师、奖励函数与 PPO 训练脚本。

## 2. 强化学习方案选择
### 2.1 候选方案评估
- **PPO + 手工奖励模型**：需重新训练 reward model、采集偏好标注，成本高且难以覆盖医疗安全指标。
- **DPO/KTO**：对比学习流程轻量，但无法直接利用 DeepSeek 教师，也难以将多维安全指标融入统一损失。
- **DeepSeek 教师 + 规则增强 RLAIF-PPO（推荐）**：DeepSeek-R1/DeepSeek-V3 负责偏好判断，现有自动化指标提供硬约束，整体复用 LoRA-SFT 流程，成本与安全性可控。

### 2.2 推荐方案：DeepSeek 教师辅助的 RLAIF-PPO
- 教师模型可评审思考链质量、医学严谨性、沟通礼仪等软指标，弥补纯规则的盲区。
- 通过 DeepSeek 给出初判，仅在有争议样本上做医学顾问抽检，可显著降低人工配对成本。
- 继续使用 LoRA，仅新增 value head 与 `trl`，单张 24GB GPU 即可完成 PPO 训练。
- 奖励函数可配置组合 `teacher_score`、思考链覆盖度、紧急指征覆盖率及风险处方惩罚，确保安全底线。
- 教师脚本解耦在独立模块内，不影响现有推理与评估，SFT 模型直接作为策略与参考模型。

## 3. 可行性分析（DeepSeek 教师 RLAIF-PPO）
### 3.1 数据与奖励构建
- `gold_set.jsonl` 与 `red_team.jsonl` 覆盖急症、用药、敏感人群，可直接作为 PPO 采样提示。借助 `scripts/deepseek_teacher.py` 定期对 gold/red 样本执行 DeepSeek 复核，如识别到风险缺口则根据教师建议增补红队或调整抽样权重。
- 统一 DeepSeek 教师提示模板，输出 `teacher_score`（0-1）、`safety_flag` 与点评，再与 `scripts/eval_auto.py` 的规则分归一化组合，示例公式 `R = 0.35*chain_score + 0.25*emergency_score - 0.15*risk_penalty + 0.25*teacher_score`。对金标样例保留教师“最佳回答”片段供 RL 提示参考。
- 当教师分与规则分差异较大时写入 `data/rl/teacher_conflicts.jsonl` 并保存 DeepSeek 建议；每周抽检 40-60 条由医学顾问复核，实现“教师 → 专家”双通道反馈。
- 新增 `data/rl/training_prompts.jsonl`（RL 采样提示）、`data/rl/teacher_judgements.jsonl`（教师缓存）、`data/rl/reward_rules.md`（奖励配置），确保提示、版本、审计信息可追溯。

### 3.2 模型与工程实现
- 使用 `AutoModelForCausalLMWithValueHead.from_pretrained` 加载 LoRA SFT 模型并挂载 value head，可训练参数 <5%。
- 冻结一份 SFT 模型作为 `ref_model`，通过 `target_kl` 控制策略漂移，保持安全回归基线。
- 新增 `scripts/deepseek_teacher.py`（DeepSeek 批量/异步调用、鉴权、缓存）、`scripts/train_ppo.py`（封装 `trl.PPOTrainer`、提示加载、奖励计算、日志）、`scripts/reward_fn.py`（规则 + 教师得分融合）。
- 扩展 `scripts/eval_auto.py` 支持加载 RL 权重，在 `scripts/batch_predict.py` 与 `scripts/demo_gradio.py` 中添加模型版本下拉与安全提示，方便产品侧比对 SFT/RL。

### 3.3 资源与成本评估
- **GPU**：单张 24GB (A5000/4090) 即可完成 2k 提示 × 3 epoch 的 PPO；若显存紧张，可启用 gradient checkpointing + micro-batch 在 16GB 设备上运行（时间增加约 25%）。
- **时间**：单轮“采样→教师打分→PPO 更新”约 4-6 小时；每周两轮训练，对单卡负载 <10 小时。
- **DeepSeek API 成本**：按 3k tokens/样本、0.002-0.004 USD/1k tokens 估算，2k 样本花费 <25 USD，通过缓存与只对高价值样本全量调用可进一步降低成本。
- **存储**：LoRA+value head+奖励缓存 <5GB；新增 `logs/deepseek_teacher/` 保存请求、版本、失败重试，满足审计。
- **人力**：工程 0.8 FTE 负责脚本与训练自动化；医学顾问 0.3 FTE 负责冲突样本审校；数据治理 0.2 FTE 维护提示模板与缓存合规。

### 3.4 依赖与交付
- 新增 `trl`、`accelerate`、`rich` 以及 DeepSeek 官方 SDK/HTTP 客户端，其余沿用 `transformers`、`peft`、`datasets`。
- 交付物包含 LoRA 适配器、value head、`config.json`、`reward_rules.md`、`teacher_judgements.jsonl`、训练/评估日志，可直接部署或继续迭代。
- 奖励权重、DeepSeek 提示与版本号均配置化，便于在不修改主训练流程的情况下快速调参或替换教师模型。

## 4. 实施方案
### 阶段 A：奖励与数据准备（第 1 周）
1. 基于分层策略抽样生成 `data/rl/training_prompts.jsonl`，确保 ≥40% 红队/高风险提示，并调用 DeepSeek 对代表性样本给出点评。
2. 开发 `scripts/deepseek_teacher.py`，实现鉴权、提示模板、指数退避重试、缓存写入 `data/rl/teacher_judgements.jsonl`，并支持“数据质检模式”复核 gold/red。
3. 封装 `reward_fn_rules` 与 `reward_fn_teacher`（可独立存放在 `scripts/reward_fn.py`），奖励日志中保留 DeepSeek 判词，便于追溯。
4. 编写 `scripts/inspect_rl_samples.py`，整合模型输出、规则分、教师分、异常标记生成 Markdown/CSV 报告，支持回放教师建议回答。
5. 撰写 `data/rl/reward_rules.md`，记录指标定义、权重区间、DeepSeek 提示模板与版本管理策略，区分质检模式与训练模式。

### 阶段 B：PPO 训练（第 2 周）
1. 实现 `scripts/train_ppo.py`，集成策略/参考模型、奖励函数、日志、checkpoint 与 CLI 超参。
2. 执行首轮 2k 提示 × 3 epoch 的 PPO，生成 `models/rl/round1`、`rewards.csv`、`kl_metrics.json`、`teacher_usage.csv`。
3. 运行 `scripts/eval_auto.py` 对 SFT 与 RL 模型做对比，输出 `reports/rl_stage/eval_report_round1.md`，标记明显退化样本。
4. 医学顾问基于抽检报告审阅 ≥50 条高风险样本，给出奖励权重、提示或数据策略的调整建议，并同步到配置。

### 阶段 C：交付与推广（第 3 周）
1. 根据反馈进行第二轮 PPO，验证指标提升的稳定性并固化奖励配置。
2. 更新 `scripts/batch_predict.py` 与 `scripts/demo_gradio.py`，加入 RL 模型选项与安全提示，同时在 README/docs 中补充 DeepSeek 教师 + RL 的使用说明。
3. 打包 `models/rl/v1.0-rl`（LoRA+value head）、`reward_rules.md`、`teacher_judgements.jsonl`、`rl_pipeline.md`、《RL 评估报告》，并记录 DeepSeek 教师版本与提示模板。
4. 复盘未覆盖风险点、后续数据扩展计划及 DeepSeek 调用成本，为常态化 RL 迭代制定节奏。

## 5. 风险与缓解
- **奖励错配**：定期审查高奖励样本，必要时在奖励函数加入冗长/重复惩罚或调整 DeepSeek 提示，防止模型“刷分”。
- **安全回退**：依靠 reference+KL 控制，并使用红队集做 A/B，对风险升高的版本立即回滚至 SFT。
- **资源或 API 波动**：DeepSeek 调用通过缓存、优先队列和速率限制控制成本；若 GPU 紧张，则缩短序列或降低 batch，延长训练轮次。
- **教师模型漂移**：记录 DeepSeek 版本与请求日志，若服务异常则降级为规则奖励 + 人工抽检，并在恢复后重新校准奖励分布。
- **合规**：所有新增数据保持脱敏，DeepSeek 调用与人工审查日志归档，确保医疗合规审计可追溯。

## 6. 里程碑与交付
1. **第 1 周**：交付 `training_prompts.jsonl`、`reward_rules.md`、`teacher_judgements.jsonl`、抽检报告模板。
2. **第 2 周**：完成首轮 RL 训练与评估，产出 `models/rl/round1`、`rewards.csv`、`teacher_usage.csv`、`eval_report_round1.md`。
3. **第 3 周**：完成第二轮训练与产品化集成，交付 `models/rl/v1.0-rl`、`rl_pipeline.md`、《RL 评估报告》，并归档 DeepSeek 教师版本/提示配置。