#!/usr/bin/env python3
"""
从 HuggingFace 下载 medical-o1 数据集
"""
import os
import json
from datasets import load_dataset

print("=" * 60)
print("从 HuggingFace 下载 medical-o1 数据集")
print("=" * 60)

try:
    print("\n📥 开始下载...")
    # 从 HuggingFace 加载数据集（使用中文配置）
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh")
    
    print(f"✅ 数据集加载成功！")
    print(f"   Splits: {list(dataset.keys())}")
    
    # 获取训练集
    if 'train' in dataset:
        train_data = dataset['train']
    else:
        # 如果没有 train split，使用第一个可用的
        train_data = dataset[list(dataset.keys())[0]]
    
    print(f"   样本数: {len(train_data)}")
    
    # 转换为列表
    data_list = []
    for item in train_data:
        data_list.append(dict(item))
    
    # 保存到本地
    target_dir = "data/raw"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "medical_o1_sft.json")
    
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 保存成功: {target_file}")
    print(f"   样本数: {len(data_list)}")
    
    # 显示第一条数据的结构
    if data_list:
        print(f"\n数据结构示例:")
        print(f"   Keys: {list(data_list[0].keys())}")
        
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n💡 如果网络问题，可以尝试：")
    print("   1. 使用代理")
    print("   2. 或从 https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT 手动下载")
