#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen3-medical-finetune",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于Qwen3-1.7B的医学问答系统微调项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qwen3-medical-finetune",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-medical-prepare=scripts.prepare_data:main",
            "qwen3-medical-train-full=scripts.train_full:main",
            "qwen3-medical-train-lora=scripts.train_lora:main",
            "qwen3-medical-predict=scripts.batch_predict:main",
            "qwen3-medical-demo=scripts.demo_gradio:main",
            "qwen3-medical-eval=scripts.eval_auto:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
)
