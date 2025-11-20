#!/bin/bash
# 合规安全RL方案 - 一键执行脚本
# 用途：准备数据并启动RL训练
# 作者：项目团队
# 日期：2024-11-20

set -e  # 遇错退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 合规安全RL方案 - 自动执行${NC}"
echo ""

# 检查目录
if [ ! -d "scripts" ]; then
    echo -e "${RED}❌ 错误：请在项目根目录运行此脚本${NC}"
    exit 1
fi

# Step 1: 激活虚拟环境
echo -e "${YELLOW}Step 1/4: 激活虚拟环境...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ 虚拟环境已激活${NC}"
else
    echo -e "${RED}⚠️  .venv不存在，跳过激活${NC}"
fi

# Step 2: 准备训练数据
echo ""
echo -e "${YELLOW}Step 2/4: 准备训练数据...${NC}"

# 检查是否已有数据
if [ -f "data/processed/train.jsonl" ] && [ $(cat data/processed/train.jsonl | wc -l) -gt 50 ]; then
    echo -e "${GREEN}✅ 发现现有训练数据（$(cat data/processed/train.jsonl | wc -l) samples）${NC}"
    read -p "是否重新准备数据？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过数据准备"
    else
        python3 scripts/prepare_data_multi_source.py medical-o1
    fi
else
    echo "准备数据（使用medical-o1数据集）..."
    python3 scripts/prepare_data_multi_source.py medical-o1
fi

# Step 3: 准备RL训练数据
echo ""
echo -e "${YELLOW}Step 3/4: 准备RL训练数据...${NC}"
python3 scripts/prepare_rl_data.py

# 验证结果
echo ""
echo -e "${GREEN}📊 数据统计:${NC}"
TOTAL=$(cat data/rl/training_prompts.jsonl | wc -l | tr -d ' ')
CRITICAL=$(grep -c '"risk_level": "critical"' data/rl/training_prompts.jsonl || echo "0")
HIGH=$(grep -c '"risk_level": "high"' data/rl/training_prompts.jsonl || echo "0")
SAFETY=$(grep -c '"safety_concern"' data/rl/training_prompts.jsonl || echo "0")

echo "  总样本数: $TOTAL"
echo "  Critical: $CRITICAL"
echo "  High: $HIGH"
echo "  安全关注样本: $SAFETY"

# 检查质量
if [ "$TOTAL" -lt 10 ]; then
    echo -e "${RED}⚠️  警告：样本数量太少（<10），建议先准备更多数据${NC}"
    exit 1
fi

if [ "$SAFETY" -lt 5 ]; then
    echo -e "${RED}⚠️  警告：安全样本太少（<5），请检查safety_red_team.jsonl${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 数据质量检查通过${NC}"

# Step 4: 检查SFT模型
echo ""
echo -e "${YELLOW}Step 4/4: 检查SFT模型...${NC}"

if [ -d "models/lora/final_lora" ] && [ -f "models/lora/final_lora/adapter_config.json" ]; then
    echo -e "${GREEN}✅ 发现SFT LoRA模型${NC}"
    SFT_READY=true
elif [ -d "models/full/final_model" ]; then
    echo -e "${GREEN}✅ 发现SFT全参数模型${NC}"
    SFT_READY=true
else
    echo -e "${YELLOW}⚠️  未发现SFT模型${NC}"
    SFT_READY=false
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📋 数据准备完成！${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# 下一步建议
if [ "$SFT_READY" = true ]; then
    echo ""
    echo -e "${GREEN}🎯 下一步：启动RL训练${NC}"
    echo ""
    echo "配置DeepSeek API（可选）:"
    echo "  export DEEPSEEK_API_KEY=\"your_key_here\""
    echo ""
    echo "启动训练:"
    echo -e "  ${YELLOW}python3 scripts/train_ppo.py${NC}"
    echo ""
    
    read -p "是否现在启动RL训练？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}🚀 启动PPO训练...${NC}"
        python3 scripts/train_ppo.py
    else
        echo ""
        echo "稍后可手动运行: python3 scripts/train_ppo.py"
    fi
else
    echo ""
    echo -e "${YELLOW}⚠️  需要先完成SFT训练${NC}"
    echo ""
    echo "运行SFT训练（LoRA）:"
    echo -e "  ${YELLOW}python3 scripts/train_lora.py${NC}"
    echo ""
    echo "或全参数微调:"
    echo -e "  ${YELLOW}python3 scripts/train_full.py${NC}"
    echo ""
    echo "SFT完成后，再运行此脚本或直接:"
    echo "  python3 scripts/train_ppo.py"
fi

echo ""
echo -e "${GREEN}✅ 完成！${NC}"
