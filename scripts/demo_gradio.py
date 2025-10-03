# scripts/demo_gradio.py
import os, re, torch, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

# 选择其一：
MODEL_DIR = "models/full/final_model"
IS_LORA = False
# BASE_DIR = "models/Qwen/Qwen3-1.7B"; ADAPTER_DIR = "models/lora/final_lora"; IS_LORA = True

def load_model():
    if IS_LORA:
        base = AutoModelForCausalLM.from_pretrained(BASE_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

def split_think_answer(text:str):
    m = re.search(r"<think>(.*?)</think>\s*(.*)", text, flags=re.S)
    if not m:
        return "", text
    return m.group(1).strip(), m.group(2).strip()

def respond(message, history):
    msgs = [{"role":"system","content":PROMPT}]
    for u,b in history:
        msgs += [{"role":"user","content":u},{"role":"assistant","content":b}]
    msgs.append({"role":"user","content":message})

    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
        )
    out = tokenizer.decode(gen[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    think, ans = split_think_answer(out)
    if think:
        return f"**思考**：\n{think}\n\n**建议**：\n{ans}"
    return ans

demo = gr.ChatInterface(
    fn=respond, title="Qwen3 医学助手（微调版）",
    description="输出包含显式思考链（<think>）与医学建议；非诊断，紧急情况请及时就医。",
    examples=["我最近总是失眠，应该怎么办？","两岁小孩发热39.5℃该如何处理？","餐后上腹痛伴反酸应该注意什么？"]
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
