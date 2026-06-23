# QS Music - 音乐风格学习与生成系统

基于深度学习的音乐风格学习与生成平台，集成 MusicGen 微调训练、ACE-Step v1.5 高质量音乐生成、AudioX-Turbo 快速音频生成三大引擎，提供从数据管理到模型训练、风格融合和音乐生成的完整工作流。

## 项目特性

- **多引擎支持**：集成 MusicGen（微调训练）、ACE-Step v1.5（DiT 扩散生成）、AudioX-Turbo（4步快速生成）
- **风格学习**：通过自有数据集微调模型，学习特定歌手/曲风的音乐特征
- **高质量生成**：ACE-Step base 模型支持 50 步扩散推理 + CFG 引导 + ADG 双引导，生成清晰人声和丰富编曲
- **数据集管理**：支持音频导入、格式统一、采样率标准化、音频切片等预处理
- **GPU 训练**：支持 NVIDIA CUDA 显卡，混合精度训练，断点续训
- **REST API**：基于 FastAPI 的推理服务，支持异步任务提交与轮询
- **Web UI**：基于 Gradio 的交互式 Web 界面

## 目录结构

```
qsmusic/
├── config/                         # 配置管理
│   ├── default_config.yaml         # 默认配置文件
│   └── config_loader.py            # 配置加载器
├── dataset/                        # 数据集管理
│   ├── dataset_manager.py          # 数据集导入、分割、加载
│   ├── preprocessor.py             # 音频预处理管线
│   └── feature_extractor.py        # 风格特征提取
├── models/                         # 模型模块
│   ├── model_manager.py            # HuggingFace 模型下载与管理
│   ├── base_model.py               # 模型抽象基类
│   ├── musicgen_model.py           # MusicGen 实现
│   └── musecoco_model.py           # MuseCoco 实现
├── training/                       # 训练系统
│   ├── trainer.py                  # PyTorch 训练循环
│   ├── monitor.py                  # TensorBoard 监控
│   └── callbacks.py                # 检查点、早停法等回调
├── generation/                     # 生成模块
│   ├── generator.py                # 统一的音乐生成 API
│   └── style_fusion.py             # 风格融合算法
├── scripts/                        # 脚本入口
│   ├── cli.py                      # 命令行工具
│   ├── run_training.py             # 训练脚本
│   ├── generate_samples.py         # 生成示例脚本
│   └── benchmark.py                # 性能基准测试
├── ui/                             # Web 界面
│   └── web_ui.py                   # Gradio Web UI
├── utils/                          # 工具函数
│   ├── logger.py                   # 日志系统
│   ├── device.py                   # GPU 设备检测
│   └── audio_utils.py              # 音频基础操作
├── tests/                          # 测试
├── ace_step/                       # ACE-Step v1.5 音乐生成模型
│   ├── acestep/                    # 核心 Python 包
│   ├── docs/                       # 多语言文档
│   ├── scripts/                    # 工具脚本
│   ├── checkpoints/                # 模型权重（需单独下载）
│   └── README.md                   # ACE-Step 官方文档
├── audiox_turbo/                   # AudioX-Turbo 音频生成模型
│   ├── audiox_turbo/               # 核心 Python 包
│   ├── configs/                    # 模型配置
│   ├── checkpoints/                # 模型权重（需单独下载）
│   └── README.md                   # AudioX-Turbo 文档
├── music_dataset/                  # 音乐数据集（需自行准备）
│   ├── raw/                        # 原始音频文件
│   ├── processed/                  # 预处理后的音频
│   └── slices/                     # 切片后的训练数据
├── checkpoints/                    # MusicGen 检查点（需单独下载）
├── logs/                           # 训练日志
├── outputs/                        # 生成的音频输出
├── requirements.txt                # Python 依赖
├── setup.py                        # 安装脚本
└── .gitignore
```

## 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.10+ | 3.11-3.13 |
| PyTorch | 2.0+ | 2.7+ (CUDA 12.8) |
| GPU | 8GB VRAM | 12GB+ VRAM (RTX 3060+) |
| 内存 | 16GB | 32GB+ |
| 磁盘 | 50GB | 100GB+ (含模型权重) |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/masterball-w/Singer_trainer.git
cd Singer_trainer
```

### 2. 安装依赖

#### QS Music 主项目

```bash
pip install -r requirements.txt

# NVIDIA CUDA 用户请确保安装正确版本的 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### ACE-Step 子项目

```bash
cd ace_step
pip install -r requirements.txt
# 或使用 uv（推荐）
uv sync
cd ..
```

#### AudioX-Turbo 子项目

```bash
cd audiox_turbo
pip install -r requirements.txt
cd ..
```

### 3. 下载模型权重

由于模型权重文件较大（总计约 60GB），不包含在 Git 仓库中。请按以下指南下载并放置到指定位置。

#### 3.1 ACE-Step v1.5 模型

从 HuggingFace 下载（中国大陆用户可使用 hf-mirror.com 镜像）：

```bash
# 设置镜像（可选，中国大陆用户推荐）
set HF_ENDPOINT=https://hf-mirror.com

# 方法一：使用 ACE-Step 内置下载工具
cd ace_step
python -m acestep.model_downloader --model acestep-v15-base
python -m acestep.model_downloader --model acestep-v15-turbo
python -m acestep.model_downloader --model vae
python -m acestep.model_downloader --model acestep-5Hz-lm-1.7B

# 方法二：使用 huggingface-cli
huggingface-cli download ACE-Step/acestep-v15-base --local-dir checkpoints/acestep-v15-base
huggingface-cli download ACE-Step/acestep-v15-turbo --local-dir checkpoints/acestep-v15-turbo
huggingface-cli download ACE-Step/vae --local-dir checkpoints/vae
huggingface-cli download ACE-Step/acestep-5Hz-lm-1.7B --local-dir checkpoints/acestep-5Hz-lm-1.7B

# 方法三：使用 git lfs
git lfs install
git clone https://huggingface.co/ACE-Step/acestep-v15-base checkpoints/acestep-v15-base
```

**放置位置**：`ace_step/checkpoints/`

| 模型 | 文件大小 | 用途 |
|------|---------|------|
| acestep-v15-base | ~4.5 GB | 基础 DiT 模型（50步，高质量） |
| acestep-v15-turbo | ~4.5 GB | 加速 DiT 模型（8步，快速） |
| vae | ~0.3 GB | VAE 编解码器 |
| acestep-5Hz-lm-1.7B | ~3.5 GB | LLM 音频规划模型（可选） |
| Qwen3-Embedding-0.6B | ~1.1 GB | 文本嵌入模型（可选） |

#### 3.2 MusicGen 预训练模型

```bash
# 使用 QS Music CLI 下载
python scripts/cli.py model download musicgen-small

# 或从 HuggingFace 下载
huggingface-cli download facebook/musicgen-small --local-dir checkpoints/pretrained/musicgen-small
```

**放置位置**：`checkpoints/pretrained/musicgen-small/`

#### 3.3 AudioX-Turbo 模型

```bash
cd audiox_turbo
# 参考 audiox_turbo/README.md 获取下载说明
```

**放置位置**：`audiox_turbo/checkpoints/`

### 4. 准备数据集

将你拥有合法授权的音乐文件放入 `music_dataset/raw/` 目录：

```bash
# 从外部目录导入音乐
python scripts/cli.py data import /path/to/your/music

# 预处理数据集（格式统一 + 切片）
python scripts/cli.py data preprocess --sample-rate 32000 --slice-duration 30

# 查看数据集统计
python scripts/cli.py data stats
```

> **版权声明**：请确保你使用的训练数据具有合法授权。本项目不包含任何版权音乐文件，`music_dataset/` 目录下的 `.gitkeep` 仅为占位符。

### 5. 训练模型（MusicGen 微调）

```bash
# 基础训练
python scripts/cli.py train \
    --model-name musicgen-small \
    --dataset-dir ./music_dataset \
    --learning-rate 5e-5 \
    --batch-size 4 \
    --max-epochs 50 \
    --device auto

# 从检查点恢复训练
python scripts/cli.py train --resume ./checkpoints/checkpoint_step_005000_epoch_3
```

### 6. 生成音乐

#### 使用 QS Music CLI

```bash
# 使用文本提示生成
python scripts/cli.py generate \
    --prompt "一段欢快的钢琴曲，带有爵士风格" \
    --duration 30 \
    --output ./outputs/my_music.wav

# 使用训练后的检查点生成
python scripts/cli.py generate \
    --prompt "古典风格的弦乐四重奏" \
    --checkpoint ./checkpoints/best_model \
    --temperature 0.8 \
    --output ./outputs/classical.wav
```

#### 使用 ACE-Step API 服务器

```bash
cd ace_step

# 配置环境
copy .env.example .env
# 编辑 .env 设置模型路径等参数

# 启动 API 服务器（延迟加载模式）
set ACESTEP_CONFIG_PATH=acestep-v15-base
set ACESTEP_DEVICE=auto
set ACESTEP_OFFLOAD_TO_CPU=true
set ACESTEP_NO_INIT=true
set HF_ENDPOINT=https://hf-mirror.com

python -c "from acestep.api_server import main; main()"
```

API 服务器启动后，可通过 HTTP 接口提交生成任务：

```python
import json, urllib.request

params = {
    "prompt": "Modern J-pop with hip-hop beats, electronic synths, male vocals",
    "lyrics": "[Verse]\n...",
    "model": "acestep-v15-base",
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "use_adg": True,
    "audio_duration": 95.0,
    "audio_format": "wav",
}

# 提交任务
data = json.dumps(params).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:8001/release_task",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = urllib.request.urlopen(req, timeout=300)
result = json.loads(resp.read().decode("utf-8"))
task_id = result["data"]["task_id"]

# 轮询结果
import time
while True:
    payload = json.dumps({"task_id": task_id}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8001/query_result",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read().decode("utf-8"))
    status = result.get("data", {}).get("status", "") if isinstance(result.get("data"), dict) else ""
    if status in ("succeeded", "success", "completed"):
        print("Done!", result)
        break
    elif status in ("failed", "error"):
        print("Failed!", result)
        break
    time.sleep(10)
```

#### 使用 ACE-Step Gradio UI

```bash
cd ace_step
python -c "from acestep.acestep_v15_pipeline import main; main()"
# 访问 http://127.0.0.1:7860
```

### 7. 启动 Web UI

```bash
python scripts/cli.py ui --port 7860
```

访问 http://localhost:7860 即可使用交互式界面。

## ACE-Step 生成参数优化指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| model | acestep-v15-base | base 模型质量优于 turbo |
| inference_steps | 50 | 步数越多质量越高（turbo 仅需 8 步） |
| guidance_scale | 7.0 | CFG 引导强度，提高提示遵循度 |
| use_adg | true | 自适应双引导（仅 base 模型） |
| shift | 3.0 | 时间步偏移 |
| infer_method | ode | ODE 扩散采样 |
| audio_format | wav | 无损输出 |
| use_tiled_decode | true | 分块解码，节省显存 |

**低显存优化（<16GB VRAM）**：
- 设置 `ACESTEP_OFFLOAD_TO_CPU=true` 启用 CPU 卸载
- 设置 `ACESTEP_INIT_LLM=false` 禁用 LLM（纯 DiT 模式）
- 启用 `use_tiled_decode=true` 分块解码

## 配置说明

所有配置项集中在 `config/default_config.yaml` 中：

| 类别 | 关键参数 | 说明 |
|------|----------|------|
| model | architecture | 模型架构（musicgen/musecoco） |
| training | learning_rate | 学习率 |
| training | batch_size | 批处理大小 |
| training | max_epochs | 最大训练轮数 |
| training | mixed_precision | 混合精度训练开关 |
| dataset | target_sample_rate | 目标采样率 |
| dataset | slice_duration | 音频切片时长 |
| generation | temperature | 生成温度（控制随机性） |
| generation | guidance_scale | CFG 缩放因子 |

## 支持的模型

| 模型 | 参数量 | 特点 | 许可证 |
|------|--------|------|--------|
| musicgen-small | 300M | 轻量，适合快速实验 | MIT |
| musicgen-medium | 1.5B | 平衡质量和资源消耗 | MIT |
| musicgen-large | 3.3B | 最佳生成质量 | MIT |
| acestep-v15-base | 2B | DiT 扩散，高质量 | MIT |
| acestep-v15-turbo | 2B | DiT 扩散，8步快速 | MIT |
| audiox-turbo | - | 4步快速音频生成 | CC-BY-NC 4.0 |

## 高级用法

### Python API 生成音乐

```python
from generation.generator import MusicGenerator

generator = MusicGenerator(architecture="musicgen", model_name="musicgen-small")
generator.load_model("./checkpoints/pretrained/musicgen-small")

audio = generator.generate(
    prompt="a peaceful piano melody with nature sounds",
    duration=60,
    temperature=0.8,
    output_path="output.wav",
)
```

### 风格分析与融合

```python
from generation.style_fusion import StyleAnalyzer, StyleFusion

analyzer = StyleAnalyzer()
profiles = analyzer.analyze_directory("./music_dataset/slices")
avg_style = analyzer.get_average_style()

# 计算两个风格的相似度
similarity = analyzer.get_style_similarity("song_a", "song_b")
```

### 自定义训练参数

```python
from config import get_config

config = get_config(overrides={
    "training.learning_rate": 1e-4,
    "training.batch_size": 8,
    "training.max_epochs": 200,
})
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v --cov=.
```

## 许可证

- **QS Music 主项目**：MIT License
- **ACE-Step v1.5**：MIT License（见 `ace_step/README.md`）
- **AudioX-Turbo**：CC-BY-NC 4.0（非商用，见 `audiox_turbo/LICENSE`）
- **MusicGen**：MIT License（Facebook/Meta）

本项目仅供学习和研究使用。使用第三方模型时，请遵循各模型自身的许可证。训练数据请确保具有合法授权。

## 免责声明

本项目不包含任何版权音乐文件。`music_dataset/` 目录仅提供结构占位符，用户需自行准备合法授权的训练数据。项目开发者不对用户使用本工具生成的任何内容的版权问题负责。
