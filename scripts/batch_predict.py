# scripts/batch_predict.py
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DATA = "data/processed/test.jsonl"
OUT  = "data/processed/test_pred.jsonl"

# ================= 模型配置 =================
# 模式选择: "full" (全参), "sft" (LoRA SFT), "rl" (PPO LoRA)
MODE = "sft" 

BASE_DIR = "models/Qwen/Qwen3-1.7B"

MODELS = {
    "full": {"path": "models/full/final_model", "is_lora": False},
    "sft":  {"path": "models/lora/final_lora", "is_lora": True},
    "rl":   {"path": "models/rl/checkpoints/final_rl_model", "is_lora": True}
}
# ===========================================

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
    print(f"🚀 Running inference in [{MODE.upper()}] mode...")
    cfg = MODELS[MODE]
    
    if cfg["is_lora"]:
        print(f"   Base: {BASE_DIR}")
        print(f"   Adapter: {cfg['path']}")
        base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, cfg["path"]).eval()
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, trust_remote_code=True)
    else:
        print(f"   Model: {cfg['path']}")
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"], use_fast=False, trust_remote_code=True)

    if not os.path.exists(DATA):
        print(f"❌ Data file {DATA} not found.")
        return
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
