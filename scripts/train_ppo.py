import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import transformers

# 兼容 TRL 0.7.x 对 transformers.top_k_top_p_filtering 的依赖
try:
    from transformers.generation.utils import top_k_top_p_filtering  # 旧版入口
except Exception:
    try:
        from transformers.generation.logits_process import top_k_top_p_filtering  # 更旧版入口
    except Exception:
        # Fallback: 简单实现一个 top-k / top-p 过滤函数
        def top_k_top_p_filtering(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = -float("inf"),
            min_tokens_to_keep: int = 1,
        ) -> torch.Tensor:
            """轻量版 top-k/top-p 过滤，供 TRL 依赖调用。"""
            if top_k > 0:
                top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, filter_value)

            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留至少 min_tokens_to_keep
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, filter_value)

            return logits

# 将函数挂到 transformers 命名空间，供 TRL import
setattr(transformers, "top_k_top_p_filtering", top_k_top_p_filtering)
from transformers import AutoTokenizer, Adafactor, DataCollatorWithPadding

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from reward_fn import RewardEngine

# ================= 配置 =================
@dataclass
class ScriptArguments:
    model_name: str = field(default="models/Qwen/Qwen3-1.7B", metadata={"help": "Base model path"})
    adapter_path: str = field(default="models/lora/final_lora", metadata={"help": "SFT LoRA path"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log"})
    learning_rate: float = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: int = field(default=4, metadata={"help": "the batch size"})
    mini_batch_size: int = field(default=1, metadata={"help": "the PPO mini batch size"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    output_dir: str = field(default="models/rl/checkpoints", metadata={"help": "Output directory"})

def main():
    parser = PPOConfig(
        model_name="qwen3-medical-rl",
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
    )
    # 这里简化参数解析，实际可用 HfArgumentParser
    config = parser

    # 1. 初始化模型与 Tokenizer
    base_model_path = os.environ.get("BASE_MODEL_PATH", "models/Qwen/Qwen3-1.7B")  # 环境变量可覆盖
    sft_adapter_path = os.environ.get("LORA_ADAPTER_PATH", "models/lora/final_lora")
    
    print(f"Loading model from {base_model_path} and adapter {sft_adapter_path}...")
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_path,
        peft_config=LoraConfig.from_pretrained(sft_adapter_path) if os.path.exists(sft_adapter_path) else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果 peft_config 没生效（例如 AutoModel... 不直接支持从目录加载 adapter），
    # 可能需要手动 merge，但在 TRL 中通常推荐直接用 peft_config 重新初始化 LoRA，
    # 或者让 model 已经是一个 peft model。
    # 简单起见，我们假设这里是在 SFT 基础上继续微调，所以让 Value Head 也是随机初始化的，
    # 而 Policy 继承了 SFT 的权重。
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备数据（若缺失则生成一个小型安全 RL 数据集）
    data_path = "data/rl/training_prompts.jsonl"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        print(f"⚠️ {data_path} not found, auto-generating a small RL dataset.")
        seed_data = [
            {
                "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                "input": "我爸突然剧烈胸痛出冷汗，还呼吸困难，该怎么办？",
                "output": "<think>评估急性冠脉综合征/主动脉夹层/肺栓塞红旗...</think> 请立即拨打120并保持安静体位，等待急救。",
            },
            {
                "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                "input": "两岁孩子持续高烧40度，还抽搐了怎么办？",
                "output": "<think>儿童高热惊厥，高风险；需紧急就医。</think> 侧卧位防误吸，立即急诊或拨打120。",
            },
            {
                "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                "input": "头痛三天了，可以吃阿莫西林吗？",
                "output": "<think>头痛病因复杂，抗生素滥用风险。</think> 不建议自行用阿莫西林，若持续或有红旗症状请就医。",
            },
            {
                "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                "input": "我正在怀孕，能不能吃布洛芬？",
                "output": "<think>孕期药物需谨慎。</think> 布洛芬孕早晚期需避免，建议咨询产科医生或用对乙酰氨基酚。",
            },
            {
                "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                "input": "感觉心慌胸闷，需要去医院吗？",
                "output": "<think>需排除心血管风险。</think> 若伴胸痛/出汗/呼吸困难请立即就医，必要时拨打120。",
            },
        ]
        with open(data_path, "w", encoding="utf-8") as f:
            for it in seed_data:
                json.dump(it, f, ensure_ascii=False)
                f.write("\n")

    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def tokenize(sample):
        prompt_text = (
            f"<|im_start|>system\n你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。<|im_end|>\n"
            f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tokenized = tokenizer(prompt_text, add_special_tokens=False)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "query_text": sample["input"],  # 原始用户问题，用于奖励
        }

    dataset = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)

    def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 将列表形式的 input_ids/attention_mask pad 成张量，同时保留原始 query_text
        batch = tokenizer.pad(
            {k: [f[k] for f in features] for k in ["input_ids", "attention_mask"]},
            padding=True,
            return_tensors="pt",
        )
        batch["query_text"] = [f["query_text"] for f in features]
        return batch

    # 3. 初始化 Trainer
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None, # TRL 会自动复制一份作为 ref_model
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collate_fn,  # 自定义 padding，保留 query_text
        optimizer=optimizer,
    )

    # 4. 初始化奖励引擎
    reward_engine = RewardEngine()

    # 5. 训练循环
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 256,
    }

    print("🚀 Starting PPO training...")
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
    
        # Get response from Policy
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **generation_kwargs
        )
        
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        prompts_for_reward = batch["query_text"]
        rewards = reward_engine.compute_rewards(prompts_for_reward, batch["response"])

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log
        ppo_trainer.log_stats(stats, batch, rewards)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Mean Reward = {torch.stack(rewards).mean().item():.2f}")
            
        # Save periodically
        if epoch > 0 and epoch % 50 == 0:
            ppo_trainer.save_pretrained(os.path.join(config.output_dir, f"step_{epoch}"))

    # Save final
    ppo_trainer.save_pretrained(os.path.join(config.output_dir, "final_rl_model"))
    print("✅ Training finished. Model saved.")

if __name__ == "__main__":
    main()
