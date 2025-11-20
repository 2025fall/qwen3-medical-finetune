import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, Adafactor

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
    # Load the base model and attach the value head
    # 注意：通常我们需要先加载 SFT 后的模型。
    # 如果 adapter_path 存在，我们应该加载 base + adapter，然后转为 AutoModelForCausalLMWithValueHead
    
    base_model_path = "models/Qwen/Qwen3-1.7B" # 请确保此路径正确，或从参数传入
    sft_adapter_path = "models/lora/final_lora"
    
    print(f"Loading model from {base_model_path} and adapter {sft_adapter_path}...")
    
    # TRL 的这个类会自动处理 PEFT
    # 但我们需要确保它加载了我们的 SFT adapter
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

    # 2. 准备数据
    # 优先寻找 RL 专用提示，否则用 SFT 数据
    data_path = "data/rl/training_prompts.jsonl"
    if not os.path.exists(data_path):
        print(f"⚠️ {data_path} not found, falling back to data/processed/train.jsonl")
        data_path = "data/processed/train.jsonl"
        
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def tokenize(sample):
        # 构建 Prompt
        # 格式： <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
        # 这里我们只负责把 query 变成 input_ids
        # 假设 tokenizer.apply_chat_template 可用
        # 但 PPO generate 需要纯 tensor
        
        prompt_text = f"你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。\nUser: {sample['input']}\nAssistant:"
        # 简单拼接，避免 template 复杂性
        
        sample["input_ids"] = tokenizer.encode(prompt_text, return_tensors="pt")[0]
        sample["query"] = sample["input"] # 用于 Reward Function
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    # 3. 初始化 Trainer
    # 定义优化器
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
        data_collator=None, # TRL 默认
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
        batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True) # Decode for teacher

        # Compute Rewards
        # 注意：reward_engine 需要纯文本的 query 和 response
        # 这里 batch["query"] 可能包含 system prompt，teacher 需要纯问题吗？
        # 是的，teacher 需要纯问题。我们在 dataset 构建时保留了原始 input。
        # 但 dataloader 出来的 batch 只有 tensors，除非我们自定义 collator。
        # 简单起见，我们从 decoded query 中提取 User 的问题，或者如果 batch 中保留了 raw text (PPOTrainer 不一定保留)。
        # 修正：我们需要在 tokenize 时不把 query 丢掉，或者重新解析。
        # 为了稳健，我们尝试从 decoded query 提取问题。
        
        prompts_for_reward = [q.split("User: ")[-1].split("\nAssistant")[0] for q in batch["query"]]
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
