import json
import os
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab
from peft import LoraConfig, TaskType, get_peft_model

# SwanLab 项目配置
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

swanlab.config.update(
    {
        "model": "Qwen/Qwen3-1.7B",
        "prompt": PROMPT,
        "data_max_length": MAX_LENGTH,
    }
)


def dataset_jsonl_transfer(origin_path, new_path):
    """将原始数据集转换为大模型微调所需数据格式的新数据集"""
    messages = []

    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            input_text = data["question"]
            output_text = f"<think>{data['think']}</think> \n {data['answer']}"
            message = {
                "instruction": PROMPT,
                "input": f"{input_text}",
                "output": output_text,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """将数据集进行预处理"""
    instr = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    resp = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instr["input_ids"] + resp["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instr["attention_mask"] + resp["attention_mask"] + [1]
    labels = [-100] * len(instr["input_ids"]) + resp["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 1) 下载模型（命中本地缓存则不会重复下载）
model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="/root/autodl-tmp/", revision="master")

# 2) 加载 tokenizer 和基础模型
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen3-1.7B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
base_model.config.use_cache = False  # 兼容梯度检查点
base_model.enable_input_require_grads()

# 2.1) 注入 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,
)
model = get_peft_model(base_model, lora_config)

# 3) 加载、处理数据集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

# 4) 训练参数
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B-LORA",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",  # transformers 4.35+ 的参数名
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-1.7B",
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 6) 保存 LoRA 适配器
save_dir = "/root/autodl-tmp/output/Qwen3-1.7B-LORA/final_lora"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ LoRA adapter saved to: {save_dir}")

# 7) 简单主观测试（前 3 条）
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []

for _, row in test_df.iterrows():
    instruction = row["instruction"]
    input_value = row["input"]
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"},
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"\nQuestion: {input_value}\n\nLLM:{response}\n"
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
