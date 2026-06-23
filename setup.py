"""setup.py: QS Music 项目安装脚本"""

from setuptools import setup, find_packages

setup(
    name="qsmusic",
    version="1.0.0",
    description="音乐风格迁移与生成系统",
    author="QS Music Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "huggingface_hub>=0.19.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "gradio>=4.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qsmusic=scripts.cli:cli",
        ],
    },
)
