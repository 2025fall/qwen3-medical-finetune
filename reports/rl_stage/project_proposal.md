# COMP 7103C Project Proposal: Safety-Driven Medical Question-Answering System

Group Info
- Zhao Yizhan: 3036657500
- Zhang Chenxi: 3036657354

## 1. Purpose
Launch an end-to-end training initiative that produces a medically reliable question-answering model through structured data curation, supervised fine-tuning, reinforcement learning, and rigorous evaluation.

## 2. Objectives
- Deliver a well-governed dataset that captures high-risk clinical scenarios, emergency triage needs, and safe prescription practices.
- Train supervised policies that internalize medical reasoning and reasoning-structure adherence while remaining compatible with LoRA and full-parameter deployments.
- Optimize the policy with reinforcement learning to prioritize safety, emergency awareness, and professional tone.
- Establish an evaluation and governance framework that combines automated scores, expert review, and deployment readiness checks.

## 3. Scope
- **Data Engineering**: Design pipelines for acquisition, normalization, deduplication, stratified sampling, annotation, and documentation of medical QA data.
- **Supervised Training**: Implement LoRA and full-parameter training workflows, including configuration management, logging, and versioned checkpoints.
- **Reinforcement Learning**: Build reward modeling and policy optimization modules that integrate rule-based and human-derived feedback.
- **Evaluation & Deployment**: Produce automated metrics, human auditing protocols, release packaging, and operational guides for downstream integration.

## 4. Workstreams
### 4.1 Data Strategy & Preparation
- Source medical QA prompts covering routine, urgent, and adversarial scenarios; formalize consent and privacy safeguards.
- Normalize records, ensure de-identification, remove duplicates, and enforce stratified splits for training, validation, and safety auditing.
- Create preference and scoring datasets with rubric-driven annotations and language-model-assisted augmentation.
- Capture metadata, lineage, and quality checks in living documentation and data cards.

### 4.2 Supervised Fine-Tuning
- Define training recipes for LoRA adapters and full-parameter runs, including tokenization, curriculum pacing, and learning-rate schedules.
- Automate experiment tracking, checkpoint management, and regression detection across successive supervised iterations.
- Export models with consistent packaging for immediate evaluation and future RL initialization.

### 4.3 Reinforcement Learning Optimization
- Establish reward functions that combine rule-based scores (chain-of-thought fidelity, emergency directives, prescription safety), expert feedback, and model-assisted preference judgments.
- Implement PPO as the primary optimization path with configurable KL penalties; support DPO/KTO alternatives when scalar rewards are insufficient.
- Maintain reference policies, value heads, and sampling pipelines to monitor exploration boundaries and stability.
- Log reward trends, constraint metrics, and qualitative notes for cross-functional review.

### 4.4 Evaluation, Safety, and Deployment Enablement
- Expand automated evaluation scripts to track safety, reward, KL divergence, and reasoning-structure adherence across supervised and RL stages.
- Coordinate medical expert audits on targeted prompt subsets, capture findings, and feed adjustments back into reward design and training loops.
- Prepare deployment deliverables: inference APIs, batch processing tools, demo interfaces, and rollback strategies.
- Compile runbooks detailing validation steps, release criteria, monitoring plans, and incident response workflows.

## 5. Deliverables
- Data governance package: acquisition plans, cleaning pipelines, annotation rubrics, preference datasets, and data cards.
- Supervised training toolkit: LoRA/full-parameter scripts, configuration templates, experiment logs, and versioned checkpoints.
- RL training suite: reward computation modules, PPO/DPO configurations, sampling workflows, policy artifacts, and audit logs.
- Evaluation and deployment assets: automated metric dashboards, expert review protocols, release documentation, and integration guides.

## 6. Resource Plan
- **Personnel**: ML engineers for data pipelines and modeling, medical QA specialists for rubric design and audits, MLOps engineers for deployment readiness, and compliance advisors for data governance.
- **Infrastructure**: GPU capacity starting with 24 GB-class devices for LoRA workflows and scalable clusters for full-parameter or large-batch PPO runs; secure storage for datasets, annotations, and experiment artifacts.
- **Tooling**: Experiment tracking platform, annotation interfaces, continuous integration for training pipelines, observability stack for evaluation metrics, and collaboration tools for cross-team alignment.

## 7. Risk Management
- **Reward Misalignment**: Blend automated scores with human oversight, calibrate KL constraints, and schedule regular red-team evaluations.
- **Safety Regression**: Gate releases behind automated regression checks and medical expert approval; maintain supervised baselines as fallback models.
- **Data Quality Drift**: Enforce continuous monitoring of dataset shifts, institute sampling audits, and refresh annotation guidelines as medical standards evolve.
- **Operational Constraints**: Prioritize efficient LoRA paths, utilize gradient checkpointing, and plan contingency tracks for resource limitations or model instability.
- **Compliance & Privacy**: Maintain de-identification pipelines, document data lineage, and align all processes with healthcare privacy regulations.

## 8. Success Criteria
- Demonstrable reduction in unsafe prescription recommendations and improved emergency indicator coverage on evaluation suites.
- Consistent reasoning-structure adherence and clinical reasoning quality verified through automated metrics and expert review.
- Reproducible training runs with documented configurations, datasets, and checkpoints across data, supervised, and RL stages.
- Deployment readiness validated through runbooks, rollback strategies, and stakeholder acceptance of safety metrics and audit outcomes.
