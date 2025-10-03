# scripts/train_full.py
import os, pandas as pd, torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

BASE = "models/qwen3-1.7b"
DATA = "data/processed"
OUT = "models/full"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def process_func(example, tokenizer):
    instr = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    resp = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instr["input_ids"] + resp["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instr["attention_mask"] + resp["attention_mask"] + [1]
    labels = [-100]*len(instr["input_ids"]) + resp["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    os.makedirs(OUT, exist_ok=True)
    snapshot_download("Qwen/Qwen3-1.7B", cache_dir="models", revision="master")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("models","Qwen","Qwen3-1.7B"), use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(os.path.join("models","Qwen","Qwen3-1.7B"), device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for p in model.parameters(): p.requires_grad = True

    train_df = pd.read_json(os.path.join(DATA,"train.jsonl"), lines=True)
    dev_df   = pd.read_json(os.path.join(DATA,"dev.jsonl"),   lines=True)
    train_ds = Dataset.from_pandas(train_df)
    dev_ds   = Dataset.from_pandas(dev_df)
    train_set = train_ds.map(lambda x: process_func(x, tokenizer), remove_columns=train_ds.column_names)
    dev_set   = dev_ds.map(lambda x: process_func(x, tokenizer), remove_columns=dev_ds.column_names)

    args = TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps", eval_steps=50, logging_steps=10,
        num_train_epochs=1, save_steps=200, learning_rate=5e-6,
        warmup_ratio=0.1, lr_scheduler_type="cosine",
        gradient_checkpointing=True, bf16=True, fp16=False,
        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.95,
        weight_decay=0.1, max_grad_norm=1.0,
        report_to="none", save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_set, eval_dataset=dev_set,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("="*40,"\n开始全参数微调\n","="*40)
    trainer.train()

    save_dir = os.path.join(OUT, "final_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Full FT saved to:", save_dir)

if __name__ == "__main__":
    main()
