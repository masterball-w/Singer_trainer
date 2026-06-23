"""QS Music 命令行接口：提供完整的项目管理和操作命令。

使用 Click 框架实现命令行工具，支持模型下载、数据预处理、
模型训练、音乐生成等操作。
"""

import sys
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="QS Music")
def cli():
    """QS Music - 音乐风格迁移与生成系统

    支持音乐模型的下载、训练、风格融合和生成。
    """
    pass


# ============================================================
# 模型管理命令
# ============================================================

@cli.group(name="model")
def model_group():
    """模型管理：下载、查看和管理音乐生成模型"""
    pass


@model_group.command(name="list")
def model_list():
    """列出所有支持的音乐生成模型"""
    from models.model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_available_models()

    table = Table(title="支持的音乐生成模型")
    table.add_column("名称", style="cyan")
    table.add_column("架构", style="green")
    table.add_column("描述")
    table.add_column("已缓存", style="yellow")

    for m in models:
        table.add_row(
            m["name"],
            m["architecture"],
            m["description"],
            "✓" if m["cached"] else "✗",
        )

    console.print(table)


@model_group.command(name="download")
@click.argument("model_name")
@click.option("--force", is_flag=True, help="强制下载最新版本")
def model_download(model_name: str, force: bool):
    """从 Hugging Face Hub 下载指定模型"""
    from models.model_manager import ModelManager

    manager = ModelManager()

    with console.status(f"正在下载模型: {model_name}"):
        try:
            path = manager.download_model(model_name, force_update=force)
            console.print(f"[green]✓ 模型已下载: {path}[/green]")
        except Exception as e:
            console.print(f"[red]✗ 下载失败: {e}[/red]")


@model_group.command(name="info")
@click.argument("model_name")
def model_info(model_name: str):
    """查看模型的详细信息"""
    from models.model_manager import ModelManager

    manager = ModelManager()
    info = manager.check_model_version(model_name)

    panel = Panel.fit(
        f"[cyan]仓库:[/cyan] {info.get('repo_id', 'N/A')}\n"
        f"[cyan]SHA:[/cyan] {info.get('sha', 'N/A')}\n"
        f"[cyan]最后修改:[/cyan] {info.get('last_modified', 'N/A')}\n"
        f"[cyan]任务类型:[/cyan] {info.get('pipeline_tag', 'N/A')}",
        title=f"模型信息: {model_name}",
    )
    console.print(panel)


# ============================================================
# 数据集管理命令
# ============================================================

@cli.group(name="data")
def data_group():
    """数据集管理：导入、预处理和管理音乐数据集"""
    pass


@data_group.command(name="import")
@click.argument("source_path")
@click.option("--dataset-dir", default="./music_dataset", help="数据集目录")
@click.option("--no-copy", is_flag=True, help="使用符号链接而非复制")
def data_import(source_path: str, dataset_dir: str, no_copy: bool):
    """从指定路径导入音乐文件到数据集"""
    from dataset.dataset_manager import MusicDatasetManager

    manager = MusicDatasetManager(dataset_dir=dataset_dir)
    count = manager.import_music(source_path, copy_files=not no_copy)
    console.print(f"[green]✓ 已导入 {count} 个文件[/green]")


@data_group.command(name="preprocess")
@click.option("--dataset-dir", default="./music_dataset", help="数据集目录")
@click.option("--sample-rate", default=32000, help="目标采样率")
@click.option("--slice-duration", default=30.0, help="切片时长（秒）")
def data_preprocess(dataset_dir: str, sample_rate: int, slice_duration: float):
    """对数据集中的音频进行预处理"""
    from dataset.dataset_manager import MusicDatasetManager

    manager = MusicDatasetManager(
        dataset_dir=dataset_dir,
        target_sample_rate=sample_rate,
        slice_duration=slice_duration,
    )

    with console.status("正在预处理音频数据..."):
        stats = manager.preprocess_dataset()

    console.print(
        Panel.fit(
            f"[cyan]总文件数:[/cyan] {stats['total_files']}\n"
            f"[cyan]已处理:[/cyan] {stats['processed_files']}\n"
            f"[cyan]生成切片:[/cyan] {stats['total_slices']}\n"
            f"[cyan]错误数:[/cyan] {len(stats.get('errors', []))}",
            title="预处理结果",
        )
    )


@data_group.command(name="stats")
@click.option("--dataset-dir", default="./music_dataset", help="数据集目录")
def data_stats(dataset_dir: str):
    """查看数据集统计信息"""
    from dataset.dataset_manager import MusicDatasetManager

    manager = MusicDatasetManager(dataset_dir=dataset_dir)
    stats = manager.get_dataset_stats()

    table = Table(title="数据集统计")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")

    for key, value in stats.items():
        table.add_row(str(key), str(value))

    console.print(table)


# ============================================================
# 训练命令
# ============================================================

@cli.command(name="train")
@click.option("--model-name", default="musicgen-small", help="模型名称")
@click.option("--architecture", default="musicgen", help="模型架构")
@click.option("--dataset-dir", default="./music_dataset", help="数据集目录")
@click.option("--checkpoint-dir", default="./checkpoints", help="检查点目录")
@click.option("--learning-rate", default=5e-5, type=float, help="学习率")
@click.option("--batch-size", default=4, type=int, help="批处理大小")
@click.option("--max-epochs", default=100, type=int, help="最大训练轮数")
@click.option("--max-steps", default=-1, type=int, help="最大训练步数")
@click.option("--gradient-accumulation", default=4, type=int, help="梯度累积步数")
@click.option("--device", default="auto", help="计算设备 (auto/cpu/cuda)")
@click.option("--resume", default=None, help="从检查点恢复训练")
@click.option("--mixed-precision/--no-mixed-precision", default=True, help="混合精度训练")
def train_command(
    model_name, architecture, dataset_dir, checkpoint_dir,
    learning_rate, batch_size, max_epochs, max_steps,
    gradient_accumulation, device, resume, mixed_precision,
):
    """启动模型训练"""
    from utils.device import get_device, get_device_info
    from utils.logger import setup_logger
    from models.model_manager import ModelManager
    from models.musicgen_model import MusicGenModel
    from models.musecoco_model import MuseCocoModel
    from dataset.dataset_manager import MusicDatasetManager
    from training.trainer import MusicTrainer
    from training.callbacks import (
        CheckpointCallback,
        EarlyStoppingCallback,
        SampleGenerationCallback,
    )

    setup_logger(log_level="INFO")

    # 获取设备
    torch_device = get_device(device)
    device_info = get_device_info()
    console.print(Panel.fit(
        "\n".join(f"[cyan]{k}:[/cyan] {v}" for k, v in device_info.items()),
        title="设备信息",
    ))

    # 下载/加载模型
    manager = ModelManager()
    model_path = manager.get_model_path(model_name)
    if model_path is None:
        console.print(f"[yellow]模型 {model_name} 未缓存，开始下载...[/yellow]")
        model_path = manager.download_model(model_name)

    # 创建模型
    if architecture == "musicgen":
        model = MusicGenModel(model_name=model_name, device=torch_device)
    elif architecture == "musecoco":
        model = MuseCocoModel(model_name=model_name, device=torch_device)
    else:
        console.print(f"[red]不支持的架构: {architecture}[/red]")
        return

    model.load_pretrained(model_path)
    model.prepare_for_training(learning_rate=learning_rate)

    # 准备数据集
    ds_manager = MusicDatasetManager(dataset_dir=dataset_dir)
    try:
        train_loader, eval_loader = ds_manager.create_dataloader(
            batch_size=batch_size,
            split_ratio=0.9,
        )
    except Exception:
        train_loader = ds_manager.create_dataloader(
            batch_size=batch_size,
            split_ratio=1.0,
        )
        eval_loader = None

    # 创建训练器
    trainer = MusicTrainer(
        model=model,
        device=torch_device,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation,
        mixed_precision=mixed_precision,
        checkpoint_dir=checkpoint_dir,
    )

    # 添加回调
    trainer.add_callback(CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        save_every_n_steps=1000,
        keep_last_n=5,
    ))
    trainer.add_callback(EarlyStoppingCallback(
        monitor_metric="eval_loss",
        patience=15,
    ))
    trainer.add_callback(SampleGenerationCallback(
        output_dir="./outputs/samples",
        generate_every_n_steps=2000,
    ))

    # 恢复训练
    if resume:
        trainer.resume_from_checkpoint(resume)

    # 开始训练
    summary = trainer.train(train_loader, eval_loader)

    console.print(Panel.fit(
        "\n".join(f"[cyan]{k}:[/cyan] {v}" for k, v in summary.items()),
        title="训练总结",
    ))


# ============================================================
# 生成命令
# ============================================================

@cli.command(name="generate")
@click.option("--prompt", default=None, help="文本描述提示")
@click.option("--model-name", default="musicgen-small", help="模型名称")
@click.option("--architecture", default="musicgen", help="模型架构")
@click.option("--checkpoint", default=None, help="训练检查点路径")
@click.option("--duration", default=30.0, type=float, help="生成时长（秒）")
@click.option("--temperature", default=1.0, type=float, help="采样温度")
@click.option("--top-k", default=250, type=int, help="Top-k 采样")
@click.option("--guidance-scale", default=3.0, type=float, help="CFG 缩放因子")
@click.option("--output", default="./outputs/generated.wav", help="输出文件路径")
@click.option("--style", default=None, help="风格描述")
def generate_command(
    prompt, model_name, architecture, checkpoint,
    duration, temperature, top_k, guidance_scale, output, style,
):
    """生成音乐"""
    from utils.logger import setup_logger
    from generation.generator import MusicGenerator
    from models.model_manager import ModelManager

    setup_logger(log_level="INFO")

    # 创建生成器
    generator = MusicGenerator(
        architecture=architecture,
        model_name=model_name,
    )

    # 加载模型
    manager = ModelManager()
    model_path = manager.get_model_path(model_name)
    if model_path is None:
        console.print(f"[yellow]模型未缓存，开始下载...[/yellow]")
        model_path = manager.download_model(model_name)

    generator.load_model(model_path)

    # 加载训练检查点（如果有）
    if checkpoint:
        generator.load_from_checkpoint(checkpoint)

    # 生成音乐
    console.print(f"[cyan]正在生成音乐: '{prompt}' ({duration}s)[/cyan]")

    audio = generator.generate(
        prompt=prompt,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        style_description=style,
        output_path=output,
    )

    console.print(f"[green]✓ 音乐已生成: {output} ({len(audio)/32000:.1f}s)[/green]")


# ============================================================
# 风格分析命令
# ============================================================

@cli.command(name="analyze")
@click.argument("directory")
@click.option("--output", default=None, help="分析报告输出路径")
def analyze_command(directory: str, output: str):
    """分析目录中音乐文件的风格特征"""
    from generation.style_fusion import StyleAnalyzer
    import json

    analyzer = StyleAnalyzer()
    profiles = analyzer.analyze_directory(directory)
    avg_style = analyzer.get_average_style()

    table = Table(title="风格分析结果")
    table.add_column("文件", style="cyan")
    table.add_column("调性", style="green")
    table.add_column("速度(BPM)", style="yellow")
    table.add_column("音色亮度", style="magenta")

    for name, profile in list(profiles.items())[:20]:
        summary = profile.get("summary", {})
        table.add_row(
            name[:30],
            summary.get("dominant_pitch", "N/A"),
            f"{summary.get('tempo', 0):.0f}",
            f"{summary.get('timbral_brightness', 0):.2f}",
        )

    console.print(table)

    if output:
        report = {
            "profiles": {
                k: {"summary": v.get("summary", {})}
                for k, v in profiles.items()
            },
            "average_style": avg_style,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✓ 报告已保存: {output}[/green]")


# ============================================================
# Web UI 命令
# ============================================================

@cli.command(name="ui")
@click.option("--port", default=7860, type=int, help="服务端口")
@click.option("--share", is_flag=True, help="创建公开链接")
def ui_command(port: int, share: bool):
    """启动 Web UI 界面"""
    from ui.web_ui import create_app

    app = create_app()
    app.launch(server_port=port, share=share)


# ============================================================
# 系统信息命令
# ============================================================

@cli.command(name="info")
def info_command():
    """显示系统环境和设备信息"""
    from utils.device import get_device_info

    info = get_device_info()

    table = Table(title="系统信息")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


if __name__ == "__main__":
    cli()
