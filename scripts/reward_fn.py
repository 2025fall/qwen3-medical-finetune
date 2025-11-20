import re
import torch
import os
import sys
from typing import List

# 添加脚本目录到路径，以便导入同目录下的模块
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from deepseek_teacher import DeepSeekTeacher

class RewardEngine:
    def __init__(self, teacher_mode="mock", safety_weight=0.5):
        # 初始化教师模型
        self.teacher = DeepSeekTeacher()
        # 编译正则表达式
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        
        # 安全性权重（针对合规安全目标）
        self.safety_weight = safety_weight
        
        # 合规安全规则库
        self.SAFETY_RULES = {
            # 处方药滥用风险（提到可自行购买/使用处方药）
            "prescription_abuse": {
                "keywords": ["自己买", "自行购买", "网上买", "能吃", "可以吃"],
                "drugs": ["抗生素", "阿莫西林", "头孢", "降压药", "二甲双胍", 
                         "安定", "地西泮", "抗抑郁药", "降糖药"],
                "penalty": -1.0,
                "description": "建议自行购买或使用处方药"
            },
            # 急危重症处理不当（未建议立即就医）
            "emergency_mishandling": {
                "symptoms": ["剧烈胸痛", "呼吸困难", "大出血", "昏厥", "抽搐", 
                            "高热惊厥", "意识不清"],
                "required_advice": ["立即", "急救", "120", "急诊"],
                "penalty": -1.5,
                "description": "急危重症未建议立即就医"
            },
            # 孕儿禁用药风险
            "pregnancy_child_drug": {
                "contexts": ["怀孕", "孕妇", "婴儿", "幼儿", "岁"],
                "risky_drugs": ["可待因", "右美沙芬", "伪麻黄碱", "阿司匹林"],
                "penalty": -1.2,
                "description": "孕儿用药风险"
            },
            # 诊断断言（非医生身份断定诊断）
            "diagnosis_assertion": {
                "patterns": ["就是", "肯定是", "一定是", "确诊"],
                "diseases": ["癌", "肿瘤", "心梗", "中风"],
                "penalty": -0.8,
                "description": "未就医即断定诊断"
            },
            # 自杀风险忽视
            "suicide_risk": {
                "keywords": ["抑郁", "自杀", "想死", "活不下去"],
                "required_action": ["心理", "精神科", "危机", "热线"],
                "penalty": -2.0,
                "description": "自杀风险未提供干预"
            }
        }
        
        # 正向安全指标（鼓励）
        self.SAFETY_POSITIVE = {
            "medical_referral": ["建议就医", "请到", "科就诊", "咨询医生", "医生指导"],
            "risk_warning": ["严重", "危险", "禁用", "禁忌", "不建议"],
            "emergency_action": ["立即", "尽快", "急诊", "120", "急救"],
            "qualified_statement": ["可能", "需要评估", "具体情况", "因人而异"]
        }
    
    def check_safety_violations(self, prompt: str, completion: str) -> dict:
        """
        检查合规安全违规
        返回: {"penalty": float, "violations": list}
        """
        violations = []
        total_penalty = 0.0
        
        # 1. 处方药滥用检测
        rule = self.SAFETY_RULES["prescription_abuse"]
        for drug in rule["drugs"]:
            if drug in prompt or drug in completion:
                # 检查是否建议自行购买
                if any(kw in completion for kw in rule["keywords"]):
                    violations.append(f"{rule['description']}: {drug}")
                    total_penalty += rule["penalty"]
                    break
        
        # 2. 急危重症处理不当
        rule = self.SAFETY_RULES["emergency_mishandling"]
        has_emergency = any(symptom in prompt for symptom in rule["symptoms"])
        if has_emergency:
            # 检查是否有紧急就医建议
            has_emergency_advice = any(adv in completion for adv in rule["required_advice"])
            if not has_emergency_advice:
                violations.append(rule["description"])
                total_penalty += rule["penalty"]
        
        # 3. 孕儿用药风险
        rule = self.SAFETY_RULES["pregnancy_child_drug"]
        has_special_context = any(ctx in prompt for ctx in rule["contexts"])
        if has_special_context:
            # 检查是否提到禁用药
            for drug in rule["risky_drugs"]:
                if drug in completion and "禁用" not in completion:
                    violations.append(f"{rule['description']}: {drug}")
                    total_penalty += rule["penalty"]
                    break
        
        # 4. 诊断断言
        rule = self.SAFETY_RULES["diagnosis_assertion"]
        for disease in rule["diseases"]:
            if disease in completion:
                # 检查是否有断言性表述
                if any(pat in completion for pat in rule["patterns"]):
                    violations.append(f"{rule['description']}: {disease}")
                    total_penalty += rule["penalty"]
                    break
        
        # 5. 自杀风险
        rule = self.SAFETY_RULES["suicide_risk"]
        has_suicide_risk = any(kw in prompt for kw in rule["keywords"])
        if has_suicide_risk:
            # 检查是否提供心理危机干预
            has_crisis_intervention = any(act in completion for act in rule["required_action"])
            if not has_crisis_intervention:
                violations.append(rule["description"])
                total_penalty += rule["penalty"]
        
        return {"penalty": total_penalty, "violations": violations}
    
    def check_safety_positives(self, completion: str) -> float:
        """
        检查正向安全指标
        返回正向加分
        """
        bonus = 0.0
        
        for category, keywords in self.SAFETY_POSITIVE.items():
            if any(kw in completion for kw in keywords):
                bonus += 0.15  # 每个类别加0.15分
        
        # 上限0.6分
        return min(0.6, bonus)
        
    def get_rule_rewards(self, completions: List[str], prompts: List[str] = None) -> List[float]:
        """计算规则部分的奖励（包含合规安全检查）"""
        rewards = []
        
        for idx, text in enumerate(completions):
            score = 0.0
            
            # === 基础规则 ===
            # 1. 格式奖励：包含 <think> 标签
            think_match = self.think_pattern.search(text)
            if think_match:
                score += 0.2
                think_content = think_match.group(1).strip()
                # 思考过程不能太短
                if len(think_content) > 20:
                    score += 0.1
            else:
                # 惩罚：没有思考过程
                score -= 0.3
                
            # 2. 内容奖励：包含建议关键词
            if any(k in text for k in ["建议", "就医", "诊断", "检查"]):
                score += 0.1
                
            # 3. 长度惩罚
            # 移除 think 标签后计算回复长度
            response_only = self.think_pattern.sub("", text).strip()
            if len(response_only) < 10:
                score -= 0.5
            elif len(response_only) < 30:
                score -= 0.1
            
            # === 合规安全检查（重点）===
            if prompts and idx < len(prompts):
                prompt = prompts[idx]
                
                # 安全违规惩罚
                safety_check = self.check_safety_violations(prompt, text)
                score += safety_check["penalty"] * self.safety_weight
                
                # 正向安全加分
                safety_bonus = self.check_safety_positives(text)
                score += safety_bonus * self.safety_weight
                
            rewards.append(score)
        return rewards

    def get_teacher_rewards(self, prompts: List[str], completions: List[str]) -> List[float]:
        """调用 DeepSeek 教师打分"""
        rewards = []
        for q, a in zip(prompts, completions):
            # 去掉 instruction 部分，只留 input
            # 假设 prompt 格式是 tokenizer 处理过的，这里简化处理，直接用
            # 实际调用时，q 应该是纯 input
            
            # 调用教师 (带缓存)
            res = self.teacher.judge(q, a)
            rewards.append(res["overall_score"])
        return rewards

    def compute_rewards(self, prompts: List[str], completions: List[str]) -> List[torch.Tensor]:
        """
        组合奖励（针对合规安全优化）：
        Total = 0.5 * Rule(含安全检查) + 0.5 * Teacher(安全性40%权重)
        安全性整体权重提升
        """
        rule_scores = self.get_rule_rewards(completions, prompts)  # 传入prompts用于安全检查
        teacher_scores = self.get_teacher_rewards(prompts, completions)
        
        final_rewards = []
        for r, t in zip(rule_scores, teacher_scores):
            # 组合公式（安全性导向：规则和教师各50%）
            total = 0.5 * r + 0.5 * t
            # 截断范围，避免梯度爆炸
            total = max(-3.0, min(2.0, total))  # 允许更大惩罚（-3.0）以强化安全性
            final_rewards.append(torch.tensor(total))
            
        return final_rewards
