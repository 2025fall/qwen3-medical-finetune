#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件示例
展示如何自定义训练和推理参数
"""

# ===== 训练配置 =====

class TrainingConfig:
    """训练配置类"""
    
    # 基础配置
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "models"
    
    # 训练参数
    LEARNING_RATE = 2e-4  # LoRA推荐值，全参数微调建议5e-6
    NUM_EPOCHS = 3
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # LoRA配置
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # 优化器配置
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    LR_SCHEDULER_TYPE = "cosine"
    
    # 保存配置
    SAVE_STEPS = 400
    EVAL_STEPS = 100
    LOGGING_STEPS = 10

# ===== 推理配置 =====

class InferenceConfig:
    """推理配置类"""
    
    # 生成参数
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    REPETITION_PENALTY = 1.1
    
    # 系统提示词
    SYSTEM_PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
    
    # 模型路径
    MODEL_PATHS = {
        "lora": {
            "base_dir": "models/Qwen/Qwen3-1.7B",
            "adapter_dir": "models/lora/final_lora"
        },
        "full": {
            "model_dir": "models/full/final_model"
        }
    }

# ===== 数据处理配置 =====

class DataConfig:
    """数据处理配置类"""
    
    # 数据路径
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    
    # 数据分割比例
    TRAIN_RATIO = 0.8
    DEV_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 文本处理
    MAX_LENGTH = 2048
    MIN_LENGTH = 10
    
    # 去重配置
    DEDUP_BY_QUESTION = True
    DEDUP_BY_SEMANTIC = True

# ===== 评估配置 =====

class EvalConfig:
    """评估配置类"""
    
    # 评估指标
    METRICS = [
        "think_coverage",      # 思考链覆盖率
        "urgent_coverage",     # 紧急信号覆盖率
        "risky_prescription_rate"  # 风险处方率
    ]
    
    # 评估数据集
    EVAL_DATASETS = [
        "dev.jsonl",
        "test.jsonl", 
        "gold_set.jsonl",
        "red_team.jsonl"
    ]
    
    # 输出配置
    REPORT_DIR = "eval_report"
    SAVE_DETAILED_RESULTS = True

# ===== 使用示例 =====

def get_training_args():
    """获取训练参数"""
    config = TrainingConfig()
    
    return {
        "learning_rate": config.LEARNING_RATE,
        "num_train_epochs": config.NUM_EPOCHS,
        "per_device_train_batch_size": config.BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "warmup_ratio": config.WARMUP_RATIO,
        "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
        "weight_decay": config.WEIGHT_DECAY,
        "save_steps": config.SAVE_STEPS,
        "eval_steps": config.EVAL_STEPS,
        "logging_steps": config.LOGGING_STEPS,
    }

def get_lora_config():
    """获取LoRA配置"""
    config = TrainingConfig()
    
    return {
        "r": config.LORA_R,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "target_modules": config.LORA_TARGET_MODULES,
    }

def get_inference_params():
    """获取推理参数"""
    config = InferenceConfig()
    
    return {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "top_k": config.TOP_K,
        "repetition_penalty": config.REPETITION_PENALTY,
    }

# ===== 自定义配置示例 =====

def create_custom_config():
    """创建自定义配置示例"""
    
    # 快速训练配置（适合调试）
    quick_config = TrainingConfig()
    quick_config.NUM_EPOCHS = 1
    quick_config.BATCH_SIZE = 1
    quick_config.GRADIENT_ACCUMULATION_STEPS = 1
    quick_config.SAVE_STEPS = 50
    quick_config.EVAL_STEPS = 25
    
    # 高质量训练配置（适合生产）
    quality_config = TrainingConfig()
    quality_config.NUM_EPOCHS = 5
    quality_config.BATCH_SIZE = 4
    quality_config.GRADIENT_ACCUMULATION_STEPS = 4
    quality_config.LEARNING_RATE = 1e-4
    
    # 保守推理配置（适合医疗场景）
    conservative_inference = InferenceConfig()
    conservative_inference.TEMPERATURE = 0.3
    conservative_inference.TOP_P = 0.8
    conservative_inference.REPETITION_PENALTY = 1.2
    
    return {
        "quick": quick_config,
        "quality": quality_config,
        "conservative": conservative_inference
    }

if __name__ == "__main__":
    print("📋 配置文件示例")
    print("=" * 30)
    
    print("训练参数:")
    print(get_training_args())
    
    print("\nLoRA配置:")
    print(get_lora_config())
    
    print("\n推理参数:")
    print(get_inference_params())
    
    print("\n自定义配置:")
    custom_configs = create_custom_config()
    for name, config in custom_configs.items():
        print(f"{name}: {config.__dict__}")
