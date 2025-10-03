#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3医学问答微调项目使用示例
展示如何使用训练好的模型进行推理
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_and_tokenizer(model_type="lora"):
    """
    加载模型和分词器
    
    Args:
        model_type (str): 模型类型，"lora" 或 "full"
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_type == "lora":
        # LoRA微调模型
        base_dir = "models/Qwen/Qwen3-1.7B"
        adapter_dir = "models/lora/final_lora"
        
        if not os.path.exists(base_dir) or not os.path.exists(adapter_dir):
            raise FileNotFoundError("LoRA模型文件不存在，请先运行训练脚本")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
        tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False, trust_remote_code=True)
        
    else:
        # 全参数微调模型
        model_dir = "models/full/final_model"
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError("全参数微调模型文件不存在，请先运行训练脚本")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    return model, tokenizer

def generate_response(model, tokenizer, question, system_prompt=None):
    """
    生成回答
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        question (str): 用户问题
        system_prompt (str): 系统提示词
    
    Returns:
        str: 模型回答
    """
    if system_prompt is None:
        system_prompt = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    
    # 构建对话消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )
    
    # 解码输出
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    
    return response

def parse_think_answer(response):
    """
    解析思考过程和答案
    
    Args:
        response (str): 模型回答
    
    Returns:
        tuple: (思考过程, 答案)
    """
    import re
    
    # 提取<think>标签内容
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        # 移除<think>标签后的内容作为答案
        answer_content = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    else:
        think_content = ""
        answer_content = response
    
    return think_content, answer_content

def batch_inference(model, tokenizer, questions, system_prompt=None):
    """
    批量推理
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        questions (list): 问题列表
        system_prompt (str): 系统提示词
    
    Returns:
        list: 回答列表
    """
    results = []
    
    for question in questions:
        print(f"处理问题: {question}")
        
        response = generate_response(model, tokenizer, question, system_prompt)
        think_content, answer_content = parse_think_answer(response)
        
        result = {
            "question": question,
            "think": think_content,
            "answer": answer_content,
            "full_response": response
        }
        
        results.append(result)
        print(f"回答: {answer_content}")
        print("-" * 50)
    
    return results

def save_results(results, output_file="inference_results.json"):
    """
    保存推理结果
    
    Args:
        results (list): 推理结果
        output_file (str): 输出文件名
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    print("🚀 Qwen3医学问答微调项目使用示例")
    print("=" * 50)
    
    # 检查是否有训练好的模型
    lora_exists = os.path.exists("models/lora/final_lora")
    full_exists = os.path.exists("models/full/final_model")
    
    if not lora_exists and not full_exists:
        print("❌ 没有找到训练好的模型，请先运行训练脚本：")
        print("   python scripts/train_lora.py  # LoRA微调")
        print("   python scripts/train_full.py  # 全参数微调")
        return
    
    # 选择模型类型
    model_type = "lora" if lora_exists else "full"
    print(f"📦 使用模型类型: {model_type}")
    
    try:
        # 加载模型
        print("🔄 加载模型中...")
        model, tokenizer = load_model_and_tokenizer(model_type)
        print("✅ 模型加载完成")
        
        # 示例问题
        sample_questions = [
            "我最近总是失眠，应该怎么办？",
            "两岁小孩发热39.5℃该如何处理？",
            "餐后上腹痛伴反酸应该注意什么？",
            "突然剧烈胸痛出冷汗，还呼吸困难，该怎么办？"
        ]
        
        print("\n🔍 开始批量推理...")
        results = batch_inference(model, tokenizer, sample_questions)
        
        # 保存结果
        save_results(results)
        
        print("\n🎉 推理完成！")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 确保有足够的GPU内存")
        print("3. 检查模型文件路径是否正确")

if __name__ == "__main__":
    main()
