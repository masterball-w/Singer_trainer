"""示例：完整训练流程演示

该脚本演示了完整的训练工作流：
1. 数据集准备和预处理
2. 模型加载与配置
3. 训练器设置（包括回调和监控）
4. 执行训练
5. 使用训练后的模型生成音乐
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from models.model_manager import ModelManager
from models.musicgen_model import MusicGenModel
from dataset.dataset_manager import MusicDatasetManager
from training.trainer import MusicTrainer
from training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    SampleGenerationCallback,
)
from generation.generator import MusicGenerator
from utils.device import get_device, get_device_info
from utils.logger import setup_logger


def main():
    setup_logger(log_level="INFO")

    # ===== 配置 =====
    config = get_config(overrides={
        "training.learning_rate": 5e-5,
        "training.batch_size": 2,       # 小批量用于演示
        "training.max_epochs": 3,       # 少量轮次用于演示
        "training.max_steps": 100,      # 限制步数用于演示
    })

    # ===== 步骤 1: 设备检测 =====
    print("=" * 60)
    print("步骤 1: 设备检测")
    print("=" * 60)

    device = get_device("auto")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    # ===== 步骤 2: 下载模型 =====
    print("\n" + "=" * 60)
    print("步骤 2: 下载预训练模型")
    print("=" * 60)

    model_name = "musicgen-small"
    manager = ModelManager()
    model_path = manager.get_model_path(model_name)

    if model_path is None:
        print(f"正在下载 {model_name}...")
        model_path = manager.download_model(model_name)

    print(f"模型路径: {model_path}")

    # ===== 步骤 3: 准备数据集 =====
    print("\n" + "=" * 60)
    print("步骤 3: 准备数据集")
    print("=" * 60)

    dataset_dir = config.get("project.dataset_dir", "./music_dataset")
    ds_manager = MusicDatasetManager(dataset_dir=dataset_dir)

    # 检查是否有数据
    stats = ds_manager.get_dataset_stats()
    print(f"数据集统计: {stats}")

    if stats.get("slices", 0) == 0:
        print("\n数据集为空！请先使用以下命令导入和预处理数据:")
        print("  python scripts/cli.py data import /path/to/music")
        print("  python scripts/cli.py data preprocess")
        print("\n提示：请确保 music_dataset/raw/ 目录中有音频文件")
        return

    # 创建数据加载器
    batch_size = config.get("training.batch_size", 4)
    try:
        train_loader, eval_loader = ds_manager.create_dataloader(
            batch_size=batch_size,
            split_ratio=0.9,
            num_workers=0,  # 演示中使用 0 避免多进程问题
        )
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(eval_loader)}")
    except Exception as e:
        print(f"数据加载器创建失败: {e}")
        return

    # ===== 步骤 4: 初始化模型 =====
    print("\n" + "=" * 60)
    print("步骤 4: 初始化模型")
    print("=" * 60)

    model = MusicGenModel(model_name=model_name, device=device)
    model.load_pretrained(model_path)
    model.prepare_for_training(
        learning_rate=config.get("training.learning_rate", 5e-5)
    )

    params = model.count_parameters()
    print(f"总参数: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")

    # ===== 步骤 5: 设置训练器 =====
    print("\n" + "=" * 60)
    print("步骤 5: 配置训练器")
    print("=" * 60)

    trainer = MusicTrainer(
        model=model,
        device=device,
        learning_rate=config.get("training.learning_rate", 5e-5),
        max_epochs=config.get("training.max_epochs", 100),
        max_steps=config.get("training.max_steps", -1),
        gradient_accumulation_steps=config.get("training.gradient_accumulation_steps", 4),
        mixed_precision=config.get("training.mixed_precision", True),
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
    )

    # 添加回调
    trainer.add_callback(CheckpointCallback(
        checkpoint_dir="./checkpoints",
        save_every_n_steps=50,
        keep_last_n=3,
    ))
    trainer.add_callback(EarlyStoppingCallback(
        monitor_metric="loss",
        patience=10,
    ))

    print("训练器配置完成")

    # ===== 步骤 6: 开始训练 =====
    print("\n" + "=" * 60)
    print("步骤 6: 开始训练")
    print("=" * 60)

    summary = trainer.train(train_loader, eval_loader)

    print("\n训练总结:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ===== 步骤 7: 使用训练后的模型生成 =====
    print("\n" + "=" * 60)
    print("步骤 7: 使用训练后的模型生成音乐")
    print("=" * 60)

    generator = MusicGenerator(architecture="musicgen", model_name=model_name)
    generator.model = model

    audio = generator.generate(
        prompt="根据训练数据风格生成的音乐",
        duration=15,
        temperature=0.9,
    )

    output_path = "./outputs/trained_sample.wav"
    generator.save_audio(audio, output_path)
    print(f"训练后生成的音频: {output_path}")

    print("\n" + "=" * 60)
    print("完整训练流程演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
