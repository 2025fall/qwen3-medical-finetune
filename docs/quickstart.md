# 快速开始指南

本指南将帮助您在几分钟内运行Qwen3医学问答微调项目。

## 🚀 5分钟快速体验

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python scripts/prepare_data.py
```

这将自动下载并处理医学问答数据集。

### 3. 启动Web演示

```bash
python scripts/demo_gradio.py
```

打开浏览器访问 http://localhost:7860，开始与模型对话！

## 📋 完整流程

### 步骤1: 环境准备

确保您已安装Python 3.8+和必要的依赖：

```bash
# 检查Python版本
python --version

# 检查CUDA (如果有GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### 步骤2: 数据准备

```bash
python scripts/prepare_data.py
```

输出示例：
```
✅ Data prepared: {'train': 8000, 'dev': 1000, 'test': 1000, 'gold': 200, 'red': 2}
```

### 步骤3: 模型训练

#### 选项A: LoRA微调 (推荐新手)

```bash
python scripts/train_lora.py
```

训练时间：约2-4小时（取决于硬件）

#### 选项B: 全参数微调

```bash
python scripts/train_full.py
```

训练时间：约8-16小时（取决于硬件）

### 步骤4: 模型测试

#### 批量测试

```bash
python scripts/batch_predict.py
```

#### 交互式测试

```bash
python scripts/demo_gradio.py
```

### 步骤5: 模型评估

```bash
python scripts/eval_auto.py
```

## 💡 使用技巧

### 1. 快速测试（使用预训练模型）

如果您想快速体验而不进行训练，可以修改`scripts/demo_gradio.py`：

```python
# 使用原始Qwen3模型而不是微调模型
MODEL_DIR = "models/Qwen/Qwen3-1.7B"
IS_LORA = False
```

### 2. 调整推理参数

在`scripts/demo_gradio.py`中调整生成参数：

```python
gen = model.generate(
    inputs.input_ids, 
    max_new_tokens=512,    # 最大生成长度
    temperature=0.7,       # 控制随机性
    top_p=0.9,            # 核采样参数
    # ... 其他参数
)
```

### 3. 自定义数据

要使用自己的数据，修改`scripts/prepare_data.py`中的数据加载部分：

```python
def load_raw():
    # 加载您自己的数据
    with open("your_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data
```

## 🎯 示例对话

启动Web界面后，您可以尝试以下问题：

- **普通症状**: "我最近总是失眠，应该怎么办？"
- **儿童医疗**: "两岁小孩发热39.5℃该如何处理？"
- **消化问题**: "餐后上腹痛伴反酸应该注意什么？"
- **紧急情况**: "突然剧烈胸痛出冷汗，还呼吸困难，该怎么办？"

## 🔧 常见问题解决

### 问题1: 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 使用LoRA微调
python scripts/train_lora.py

# 或减小batch size
# 在训练脚本中修改 per_device_train_batch_size=1
```

### 问题2: 模型下载慢

**症状**: 模型下载卡住

**解决方案**:
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题3: Web界面无法访问

**症状**: 浏览器无法打开 http://localhost:7860

**解决方案**:
```bash
# 检查端口是否被占用
netstat -an | grep 7860

# 使用其他端口
python scripts/demo_gradio.py --port 8080
```

## 📊 性能基准

在RTX 3080 (10GB)上的参考性能：

| 任务 | LoRA微调 | 全参数微调 |
|------|----------|------------|
| 训练时间 | 2-4小时 | 8-16小时 |
| 显存占用 | 8GB | 16GB+ |
| 推理速度 | 50 tokens/s | 60 tokens/s |

## 🎓 下一步学习

完成快速开始后，建议学习：

1. [数据准备详解](data_preparation.md)
2. [训练参数调优](training.md)
3. [模型评估方法](evaluation.md)
4. [部署和优化](deployment.md)

## 💬 获取帮助

如果遇到问题：

1. 查看[常见问题](faq.md)
2. 在GitHub Issues中提问
3. 查看项目文档

祝您使用愉快！🎉
