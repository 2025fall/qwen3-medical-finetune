import json
import os
import random

# 确保随机性可控
random.seed(42)

INPUT_FILES = {
    "gold": "data/processed/gold_set.jsonl",
    "red": "data/processed/red_team.jsonl",
    "safety_red": "data/rl/safety_red_team.jsonl",  # 新增：合规安全红队数据
    "train": "data/processed/train.jsonl"
}
OUTPUT_FILE = "data/rl/training_prompts.jsonl"
TARGET_SIZE = 2000

# 针对合规安全目标的采样策略
SAFETY_FOCUS_SAMPLING = {
    "safety_red_oversampling": 3,  # 安全红队样本重复3次（强化训练）
    "high_risk_ratio": 0.6,        # 60%高风险样本
    "general_ratio": 0.4           # 40%一般样本
}

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    print("🔄 Loading source datasets...")
    gold = load_jsonl(INPUT_FILES["gold"])
    red = load_jsonl(INPUT_FILES["red"])
    safety_red = load_jsonl(INPUT_FILES["safety_red"])
    train = load_jsonl(INPUT_FILES["train"])
    
    print(f"   Gold: {len(gold)}, Red: {len(red)}, Safety-Red: {len(safety_red)}, Train: {len(train)}")
    
    # 新策略（针对合规安全目标）：
    # 1. 安全红队样本 × 3 (过采样，强化安全性学习)
    # 2. 一般红队样本 (原有高风险场景)
    # 3. Gold Set 样本 (高质量锚点)
    # 4. 从 Train 中按风险等级采样：60%高风险 + 40%一般
    
    dataset = []
    
    # 安全红队样本过采样（重复3次强化训练）
    oversampling_times = SAFETY_FOCUS_SAMPLING["safety_red_oversampling"]
    for _ in range(oversampling_times):
        dataset.extend(safety_red)
    
    # 添加其他高优先级样本
    dataset.extend(red)
    dataset.extend(gold)
    
    # 去重 (以 input 为 key)
    seen = set()
    unique_dataset = []
    for item in dataset:
        if item["input"] not in seen:
            unique_dataset.append(item)
            seen.add(item["input"])
            
    current_count = len(unique_dataset)
    needed = max(0, TARGET_SIZE - current_count)
    
    if needed > 0 and len(train) > 0:
        # 过滤掉已存在的
        train_filtered = [x for x in train if x["input"] not in seen]
        
        # 按风险等级分组
        high_risk_train = [x for x in train_filtered 
                          if x.get("meta", {}).get("risk_level") in ["high", "critical"]]
        general_train = [x for x in train_filtered 
                        if x.get("meta", {}).get("risk_level") not in ["high", "critical"]]
        
        # 计算采样数量
        high_risk_count = int(needed * SAFETY_FOCUS_SAMPLING["high_risk_ratio"])
        general_count = needed - high_risk_count
        
        # 采样
        sampled_high = random.sample(high_risk_train, min(len(high_risk_train), high_risk_count))
        sampled_general = random.sample(general_train, min(len(general_train), general_count))
        
        unique_dataset.extend(sampled_high)
        unique_dataset.extend(sampled_general)
        
        print(f"   📊 Sampling from train: {len(sampled_high)} high-risk + {len(sampled_general)} general")
        
    # 简化字段，RL 只需要 prompt (input)
    # 但为了兼容性，我们保留原始结构
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in unique_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 统计最终构成
    risk_stats = {}
    for item in unique_dataset:
        risk = item.get("meta", {}).get("risk_level", "unknown")
        risk_stats[risk] = risk_stats.get(risk, 0) + 1
            
    print(f"\n✅ Generated {len(unique_dataset)} RL prompts at {OUTPUT_FILE}")
    print(f"   📊 Risk Level Distribution:")
    for risk, count in sorted(risk_stats.items()):
        print(f"      {risk}: {count} ({count/len(unique_dataset)*100:.1f}%)")
    print(f"   🎯 Strategy: Safety-focused (60% high-risk + oversampled safety cases)")

if __name__ == "__main__":
    main()
