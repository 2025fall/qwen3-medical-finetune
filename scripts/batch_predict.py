# scripts/batch_predict.py
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DATA = "data/processed/test.jsonl"
OUT  = "data/processed/test_pred.jsonl"

# 选择其一：
# 1) 全参：
MODEL_DIR = "models/full/final_model"
IS_LORA = False
# 2) LoRA：
# BASE_DIR = "models/Qwen/Qwen3-1.7B"
# ADAPTER_DIR = "models/lora/final_lora"
# IS_LORA = True

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

def apply_template(tokenizer, question):
    return tokenizer.apply_chat_template(
        [
            {"role":"system","content":PROMPT},
            {"role":"user","content":question}
        ],
        tokenize=False, add_generation_prompt=True
    )

def main():
    if IS_LORA:
        base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)

    outs = []
    with open(DATA,"r",encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = apply_template(tokenizer, r["input"])
            inputs = tokenizer([text], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    inputs.input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
                )
            gen_ids = gen[0][len(inputs.input_ids[0]):]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outs.append({
                "instruction": r["instruction"],
                "input": r["input"],
                "output": resp,
                "meta": r.get("meta",{})
            })
    with open(OUT,"w",encoding="utf-8") as f:
        for o in outs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("✅ Predictions saved:", OUT)

if __name__ == "__main__":
    main()
