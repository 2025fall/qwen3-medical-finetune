import os
import json
import time
import hashlib
import random
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"  # 假设的 Endpoint，实际需按 DeepSeek 官方文档
MODEL_NAME = "deepseek-chat" # 或 deepseek-reasoner
CACHE_FILE = "data/rl/teacher_judgements.jsonl"
LOG_DIR = "logs/deepseek_teacher"

# 评分模板
JUDGE_TEMPLATE = """
作为一个资深医学专家，请评估以下医生（AI模型）对患者问题的回答。

【患者问题】：
{question}

【医生回答】：
{answer}

请基于以下维度打分（0-10分）：
1. 安全性（Safety）：是否有误导、漏诊急症、错误处方风险？（严重错误得0分）
2. 逻辑性（Reasoning）：思考过程是否清晰，诊断逻辑是否符合医学指南？
3. 完整性（Completeness）：是否覆盖了鉴别诊断、建议与指征？
4. 同理心（Empathy）：语气是否专业且安抚患者？

请以 JSON 格式输出，包含字段：
- scores: {{"safety": float, "reasoning": float, "completeness": float, "empathy": float}}
- overall_score: float (0-1, 归一化总分)
- critique: "简短点评"
"""

# ===========================================

os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class DeepSeekTeacher:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None
        self.cache = self._load_cache()
        if not self.client:
            print("⚠️  WARNING: DEEPSEEK_API_KEY not found. Running in MOCK mode.")

    def _load_cache(self) -> Dict[str, Dict]:
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        cache[item["id"]] = item
                    except:
                        pass
        return cache

    def _save_to_cache(self, item: Dict):
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.cache[item["id"]] = item

    def _get_hash(self, question: str, answer: str) -> str:
        content = f"{question}::{answer}"
        return hashlib.md5(content.encode()).hexdigest()

    def mock_judge(self, question: str, answer: str) -> Dict:
        """模拟打分，用于测试流程"""
        time.sleep(0.1) # Simulate latency
        # 简单的规则：如果回答长一点，分高一点；如果有<think>，分高一点
        base_score = 0.6
        if "<think>" in answer: base_score += 0.2
        if len(answer) > 100: base_score += 0.1
        
        score = min(0.95, base_score + random.uniform(-0.05, 0.05))
        return {
            "scores": {
                "safety": 9.0,
                "reasoning": score * 10,
                "completeness": 8.0,
                "empathy": 8.5
            },
            "overall_score": score,
            "critique": "【Mock】回答尚可，包含思考过程，但建议补充更多鉴别诊断细节。"
        }

    def judge(self, question: str, answer: str) -> Dict:
        item_id = self._get_hash(question, answer)
        if item_id in self.cache:
            return self.cache[item_id]["judgement"]

        if not self.client:
            result = self.mock_judge(question, answer)
        else:
            try:
                prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                result = json.loads(content)
            except Exception as e:
                print(f"❌ API Error: {e}")
                # Fallback or raise
                return self.mock_judge(question, answer)

        # Save result
        entry = {
            "id": item_id,
            "question": question,
            "answer": answer,
            "judgement": result,
            "timestamp": time.time(),
            "model": "mock" if not self.client else MODEL_NAME
        }
        self._save_to_cache(entry)
        return result

def main():
    # 简单的测试 CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/gold_set.jsonl", help="Input JSONL file to judge")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to test")
    args = parser.parse_args()

    teacher = DeepSeekTeacher()
    
    print(f"🔍 Judging first {args.limit} samples from {args.input}...")
    
    if not os.path.exists(args.input):
        print(f"❌ Input file {args.input} not found.")
        return

    with open(args.input, "r") as f:
        lines = f.readlines()
        
    samples = [json.loads(line) for line in lines[:args.limit]]
    
    for s in tqdm(samples):
        q = s["input"]
        a = s["output"]
        res = teacher.judge(q, a)
        print(f"\nQ: {q[:30]}...")
        print(f"Score: {res['overall_score']:.2f} | Critique: {res.get('critique','')}")

if __name__ == "__main__":
    main()
