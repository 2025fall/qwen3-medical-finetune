# scripts/train_lora.py
# -*- coding: utf-8 -*-
"""
Qwen3-1.7B LoRA 训练脚本（已修复）：
- 兼容老/新 transformers：自动选择 evaluation_strategy / eval_strategy
- 启用 gradient checkpointing 时，显式开启 input grads，关闭 use_cache
- 使用 DataCollatorForSeq2Seq 动态 padding，并对 labels 的 pad 填 -100
- 修正 tokenizer 的 pad_token_id（若无则回退到 eos）
- 默认不做 4/8bit 量化；如需量化见文末注释
"""

import os
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model
import transformers

# 可选：纯文本任务避免 transformers 误导入 torchvision
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

BASE_DIR = os.path.join("models", "Qwen", "Qwen3-1.7B")  # 模型本地目录
DATA = "data/processed"
OUT = "models/lora"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def build_example(example, tokenizer):
    """构造单条样本（仅对 assistant 段计 loss）"""
    instr_text = (
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    instr = tokenizer(instr_text, add_special_tokens=False)
    resp = tokenizer(example["output"], add_special_tokens=False)

    eos = tokenizer.eos_token_id
    input_ids = instr["input_ids"] + resp["input_ids"] + [eos]
    attention_mask = instr["attention_mask"] + resp["attention_mask"] + [1]
    # 仅对 assistant 段（resp + 末尾EOS）计算 loss，system/user 段用 -100 屏蔽
    labels = ([-100] * len(instr["input_ids"])) + resp["input_ids"] + [eos]

    # 截断到 MAX_LENGTH（按相同长度裁剪）
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) 如首次运行，下载模型（之后会直接命中本地缓存）
    snapshot_download("Qwen/Qwen3-1.7B", cache_dir="models", revision="master")

    # 2) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 3) model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        device_map="auto",
        dtype=torch.bfloat16,   # 若不支持 bf16，可改为 fp16：dtype=torch.float16 并在 args 里 fp16=True, bf16=False
        trust_remote_code=True,
    )

    # 4) 注入 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64, lora_alpha=128, lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    # 5) 梯度检查点兼容（关键两行）
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 6) 数据集
    train_df = pd.read_json(os.path.join(DATA, "train.jsonl"), lines=True)
    dev_df   = pd.read_json(os.path.join(DATA, "dev.jsonl"),   lines=True)

    train_ds = Dataset.from_pandas(train_df)
    dev_ds   = Dataset.from_pandas(dev_df)

    train_set = train_ds.map(lambda x: build_example(x, tokenizer), remove_columns=train_ds.column_names)
    dev_set   = dev_ds.map(lambda x: build_example(x, tokenizer), remove_columns=dev_ds.column_names)

    # 7) collator：自动对齐到 batch 内统一长度，并把 labels 的 pad 填 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # 可选：利于 Tensor Core
    )

    # 8) 兼容老/新 transformers 的 evaluation_strategy 参数名
    strategy_key = "evaluation_strategy"
    try:
        _ = transformers.TrainingArguments(output_dir="/tmp", evaluation_strategy="no")
    except TypeError:
        strategy_key = "eval_strategy"

    args_kwargs = dict(
        output_dir=OUT,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=10,
        eval_steps=100,
        num_train_epochs=3,
        save_steps=400,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    args_kwargs[strategy_key] = "steps"
    args = TrainingArguments(**args_kwargs)

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=data_collator,
    )

    # 10) 训练
    trainer.train()

    # 11) 保存
    save_dir = os.path.join(OUT, "final_lora")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ LoRA saved to:", save_dir)


if __name__ == "__main__":
    main()
