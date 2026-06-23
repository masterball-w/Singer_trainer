"""性能基准测试：评估训练效率和生成质量。

该脚本执行以下性能测试：
1. 数据预处理速度测试
2. 模型前向传播速度测试
3. 训练步骤吞吐量测试
4. 音乐生成速度测试
5. GPU 显存使用分析
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from utils.device import get_device, get_device_info, get_gpu_memory_usage
from utils.logger import setup_logger

console = Console()


def benchmark_preprocessing(num_files: int = 10, duration: float = 30.0):
    """测试数据预处理速度。

    Args:
        num_files: 测试文件数量。
        duration: 每个文件的时长（秒）。
    """
    from dataset.preprocessor import AudioPreprocessor
    import tempfile
    import soundfile as sf

    console.print("\n[bold cyan]=== 数据预处理性能测试 ===[/bold cyan]")

    pp = AudioPreprocessor(
        target_sample_rate=32000,
        slice_duration=30.0,
        slice_overlap=2.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 创建测试文件
        for i in range(num_files):
            audio = np.random.randn(32000 * int(duration)).astype(np.float32)
            sf.write(str(tmp_path / f"test_{i}.wav"), audio, 32000)

        # 测试
        output_dir = tmp_path / "output"
        start = time.time()
        total_slices = 0

        for i in range(num_files):
            slices = pp.process_file(
                tmp_path / f"test_{i}.wav",
                output_dir,
            )
            total_slices += len(slices)

        elapsed = time.time() - start

    table = Table(title="预处理性能")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("文件数", str(num_files))
    table.add_row("总切片数", str(total_slices))
    table.add_row("总耗时", f"{elapsed:.2f}s")
    table.add_row("平均处理时间/文件", f"{elapsed/num_files:.2f}s")
    table.add_row("吞吐量", f"{num_files/elapsed:.1f} files/s")
    console.print(table)

    return {"elapsed": elapsed, "files_per_sec": num_files / elapsed}


def benchmark_feature_extraction(num_samples: int = 20, duration: float = 30.0):
    """测试特征提取速度。

    Args:
        num_samples: 测试样本数量。
        duration: 每个样本的时长（秒）。
    """
    from dataset.feature_extractor import StyleFeatureExtractor

    console.print("\n[bold cyan]=== 特征提取性能测试 ===[/bold cyan]")

    extractor = StyleFeatureExtractor(sample_rate=32000, feature_method="combined")

    times = []
    for _ in range(num_samples):
        audio = np.random.randn(32000 * int(duration)).astype(np.float32)

        start = time.time()
        features = extractor.extract(audio, 32000)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    table = Table(title="特征提取性能")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("样本数", str(num_samples))
    table.add_row("平均时间", f"{avg_time:.3f}s")
    table.add_row("标准差", f"{std_time:.3f}s")
    table.add_row("吞吐量", f"{1/avg_time:.1f} samples/s")
    table.add_row("特征维度", str(extractor.get_feature_dimension()))
    console.print(table)

    return {"avg_time": avg_time, "throughput": 1 / avg_time}


def benchmark_style_fusion():
    """测试风格融合模块速度。"""
    from generation.style_fusion import StyleEmbedding, StyleFusion

    console.print("\n[bold cyan]=== 风格融合模块性能测试 ===[/bold cyan]")

    device = get_device("auto")
    embedding_dim = 512
    num_styles = 5
    batch_size = 8

    embedding_net = StyleEmbedding(input_dim=25, embedding_dim=embedding_dim).to(device)

    results = {}

    for strategy in ["weighted_avg", "attention", "adaptive"]:
        fusion = StyleFusion(
            embedding_dim=embedding_dim,
            strategy=strategy,
            num_heads=8,
        ).to(device)

        # 生成测试数据
        embeddings = [torch.randn(batch_size, embedding_dim, device=device) for _ in range(num_styles)]

        # 预热
        for _ in range(10):
            _ = fusion(embeddings)

        # 测试
        num_runs = 100
        start = time.time()
        for _ in range(num_runs):
            _ = fusion(embeddings)
        elapsed = time.time() - start

        results[strategy] = {
            "avg_time_ms": elapsed / num_runs * 1000,
            "throughput": num_runs * batch_size / elapsed,
        }

    table = Table(title="风格融合性能")
    table.add_column("策略", style="cyan")
    table.add_column("平均时间(ms)", style="green")
    table.add_column("吞吐量(samples/s)", style="yellow")

    for strategy, metrics in results.items():
        table.add_row(
            strategy,
            f"{metrics['avg_time_ms']:.2f}",
            f"{metrics['throughput']:.0f}",
        )

    console.print(table)
    return results


def benchmark_model_forward():
    """测试模型前向传播速度。

    使用模拟数据测试训练前向步骤的吞吐量。
    此测试不需要预训练模型权重，仅测试架构计算性能。
    """
    from generation.style_fusion import StyleEmbedding

    console.print("\n[bold cyan]=== 模型前向传播性能测试 ===[/bold cyan]")

    device = get_device("auto")

    # 使用 StyleEmbedding 作为轻量级测试代理
    # 实际模型前向传播需要加载完整预训练权重
    embedding = StyleEmbedding(input_dim=25, embedding_dim=512).to(device)
    embedding.train()

    batch_sizes = [1, 4, 8, 16]
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, 25, device=device)

        # 预热
        for _ in range(20):
            _ = embedding(x)

        # 测试
        num_runs = 200
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_runs):
            _ = embedding(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        avg_ms = elapsed / num_runs * 1000
        throughput = num_runs * bs / elapsed

        results[bs] = {
            "avg_time_ms": avg_ms,
            "throughput": throughput,
        }

    table = Table(title="前向传播性能 (StyleEmbedding)")
    table.add_column("批大小", style="cyan")
    table.add_column("平均时间(ms)", style="green")
    table.add_column("吞吐量(samples/s)", style="yellow")

    for bs, metrics in results.items():
        table.add_row(
            str(bs),
            f"{metrics['avg_time_ms']:.2f}",
            f"{metrics['throughput']:.0f}",
        )

    console.print(table)
    return results


def benchmark_training_step():
    """测试训练步骤吞吐量。

    模拟完整的训练步骤（前向 + 反向 + 优化器），
    测量每步耗时和吞吐量。
    """
    from generation.style_fusion import StyleEmbedding

    console.print("\n[bold cyan]=== 训练步骤吞吐量测试 ===[/bold cyan]")

    device = get_device("auto")

    # 使用 StyleEmbedding 作为代理模型
    model = StyleEmbedding(input_dim=25, embedding_dim=512).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    batch_sizes = [4, 8]
    results = {}

    for bs in batch_sizes:
        # 预热
        for _ in range(10):
            x = torch.randn(bs, 25, device=device)
            target = torch.randn(bs, 512, device=device)
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 测试
        num_runs = 100
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(num_runs):
            x = torch.randn(bs, 25, device=device)
            target = torch.randn(bs, 512, device=device)
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        avg_ms = elapsed / num_runs * 1000
        throughput = num_runs * bs / elapsed

        results[bs] = {
            "avg_time_ms": avg_ms,
            "throughput": throughput,
        }

    table = Table(title="训练步骤吞吐量")
    table.add_column("批大小", style="cyan")
    table.add_column("平均时间(ms)", style="green")
    table.add_column("吞吐量(samples/s)", style="yellow")

    for bs, metrics in results.items():
        table.add_row(
            str(bs),
            f"{metrics['avg_time_ms']:.2f}",
            f"{metrics['throughput']:.0f}",
        )

    console.print(table)
    return results


def benchmark_memory():
    """测试 GPU 显存使用。"""
    console.print("\n[bold cyan]=== GPU 显存分析 ===[/bold cyan]")

    info = get_device_info()

    table = Table(title="设备信息")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)

    if torch.cuda.is_available():
        mem = get_gpu_memory_usage()
        if mem:
            mem_table = Table(title="GPU 显存使用")
            mem_table.add_column("指标", style="cyan")
            mem_table.add_column("值(MB)", style="green")
            mem_table.add_row("已分配", f"{mem['allocated_mb']:.1f}")
            mem_table.add_row("已预留", f"{mem['reserved_mb']:.1f}")
            mem_table.add_row("可用", f"{mem['free_mb']:.1f}")
            console.print(mem_table)


def run_all_benchmarks():
    """运行所有性能基准测试。"""
    setup_logger(log_level="WARNING")

    console.print("[bold]QS Music 性能基准测试[/bold]\n")

    results = {}

    results["preprocessing"] = benchmark_preprocessing()
    results["feature_extraction"] = benchmark_feature_extraction()
    results["style_fusion"] = benchmark_style_fusion()
    results["model_forward"] = benchmark_model_forward()
    results["training_step"] = benchmark_training_step()
    benchmark_memory()

    # 总结
    console.print("\n" + "=" * 60)
    console.print("[bold green]性能测试完成![/bold green]")
    console.print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_benchmarks()
