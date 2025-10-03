# 安装指南

## 系统要求

### 硬件要求

- **CPU**: 推荐8核心以上
- **内存**: 16GB以上（全参数微调需要32GB+）
- **GPU**: 推荐NVIDIA RTX 3080/4080或更高（8GB显存以上）
- **存储**: 至少50GB可用空间

### 软件要求

- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **CUDA**: 11.8+ (如果使用GPU)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-username/qwen3-medical-finetune.git
cd qwen3-medical-finetune
```

### 2. 创建虚拟环境

#### 使用venv (推荐)

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 使用conda

```bash
# 创建conda环境
conda create -n qwen3-medical python=3.9
conda activate qwen3-medical
```

### 3. 安装依赖

#### 基础安装

```bash
pip install -r requirements.txt
```

#### 可选：开发环境

```bash
pip install -e .[dev]
```

#### 可选：文档环境

```bash
pip install -e .[docs]
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import peft; print(f'PEFT版本: {peft.__version__}')"
```

## 可选组件安装

### CUDA支持

如果您有NVIDIA GPU，建议安装CUDA版本的PyTorch：

```bash
# 卸载CPU版本
pip uninstall torch torchvision torchaudio

# 安装CUDA版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Flash Attention (可选)

对于支持的GPU，可以安装Flash Attention以提升性能：

```bash
pip install flash-attn --no-build-isolation
```

### BitsAndBytes (可选)

用于4bit/8bit量化：

```bash
pip install bitsandbytes
```

### DeepSpeed (可选)

用于分布式训练：

```bash
pip install deepspeed
```

## 常见问题

### 1. 内存不足

如果遇到内存不足的问题：

- 使用LoRA微调而不是全参数微调
- 减小batch_size
- 使用梯度累积
- 启用gradient_checkpointing

### 2. CUDA版本不匹配

确保PyTorch版本与CUDA版本兼容：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### 3. 依赖冲突

如果遇到依赖冲突：

```bash
# 清理缓存
pip cache purge

# 重新安装
pip install --no-cache-dir -r requirements.txt
```

### 4. ModelScope连接问题

如果ModelScope下载缓慢：

```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

或使用国内镜像：

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## Docker安装 (可选)

如果您喜欢使用Docker：

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/demo_gradio.py"]
```

构建和运行：

```bash
docker build -t qwen3-medical .
docker run -p 7860:7860 --gpus all qwen3-medical
```

## 下一步

安装完成后，请查看：

- [快速开始指南](quickstart.md)
- [数据准备](data_preparation.md)
- [模型训练](training.md)
