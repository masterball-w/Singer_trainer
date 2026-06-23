"""训练监控：实时追踪训练指标、生成可视化报告。

集成 TensorBoard 日志、文件日志和实时指标追踪功能，
提供训练过程的全面可视化。
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger


class TrainingMonitor:
    """训练过程监控器。

    追踪损失值、学习率、梯度范数等训练指标，
    支持 TensorBoard 日志记录和训练样本生成预览。

    Attributes:
        log_dir: 日志输出目录。
        tb_writer: TensorBoard SummaryWriter 实例。
        metrics_history: 指标历史记录。
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "./logs",
        tensorboard_enabled: bool = True,
        log_file: Optional[Union[str, Path]] = None,
        sample_dir: Optional[Union[str, Path]] = None,
    ):
        """初始化训练监控器。

        Args:
            log_dir: TensorBoard 日志目录。
            tensorboard_enabled: 是否启用 TensorBoard。
            log_file: 文本日志文件路径。
            sample_dir: 生成样本保存目录。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = Path(sample_dir) if sample_dir else self.log_dir / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = None
        self.tensorboard_enabled = tensorboard_enabled

        # 指标历史
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.step_timestamps: List[float] = []
        self.current_step = 0
        self.current_epoch = 0
        self.start_time: Optional[float] = None

        # 最佳指标记录
        self.best_loss = float("inf")
        self.best_step = 0

        # 初始化 TensorBoard
        if tensorboard_enabled:
            self._init_tensorboard()

        # 设置文件日志
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(str(log_path), rotation="10 MB", retention="7 days")

    def _init_tensorboard(self) -> None:
        """初始化 TensorBoard writer。"""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard 日志已启动: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard 不可用，将仅使用文件日志")
            self.tensorboard_enabled = False

    def on_train_start(self, config: Optional[Dict] = None) -> None:
        """训练开始回调。

        Args:
            config: 训练配置字典。
        """
        self.start_time = time.time()
        self.current_step = 0
        self.metrics_history.clear()
        logger.info("=== 训练开始 ===")

        if config and self.tb_writer:
            # 记录超参数
            hparams = {}
            for key in ["learning_rate", "batch_size", "max_epochs", "optimizer"]:
                if key in config:
                    hparams[key] = config[key]
            self.tb_writer.add_text("hparams", json.dumps(hparams, indent=2))

    def on_step_end(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        lr: Optional[float] = None,
    ) -> None:
        """训练步骤结束回调。

        记录训练指标并更新可视化。

        Args:
            step: 当前全局步数。
            epoch: 当前训练轮数。
            metrics: 本步骤的训练指标。
            lr: 当前学习率。
        """
        self.current_step = step
        self.current_epoch = epoch
        self.step_timestamps.append(time.time())

        # 更新历史记录
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # 追踪最佳损失
        loss = metrics.get("loss", float("inf"))
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = step

        # TensorBoard 记录
        if self.tb_writer and self.tensorboard_enabled:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"train/{key}", value, step)
            if lr is not None:
                self.tb_writer.add_scalar("train/learning_rate", lr, step)

        # 控制台输出（每 50 步一次）
        if step % 50 == 0:
            loss_str = f"loss={loss:.4f}"
            if "perplexity" in metrics:
                loss_str += f", ppl={metrics['perplexity']:.2f}"
            if lr is not None:
                loss_str += f", lr={lr:.2e}"

            speed = self._compute_speed()
            if speed > 0:
                loss_str += f", {speed:.1f} steps/s"

            logger.info(f"[Epoch {epoch}] Step {step}: {loss_str}")

    def on_eval_end(self, step: int, metrics: Dict[str, float]) -> None:
        """评估结束回调。

        Args:
            step: 当前全局步数。
            metrics: 评估指标。
        """
        for key, value in metrics.items():
            self.metrics_history[f"eval_{key}"].append(value)

        if self.tb_writer and self.tensorboard_enabled:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"eval/{key}", value, step)

        logger.info(f"[Eval] Step {step}: {metrics}")

    def on_sample_generated(
        self,
        step: int,
        audio: "np.ndarray",
        sample_rate: int = 32000,
        prompt: Optional[str] = None,
    ) -> None:
        """生成样本保存回调。

        Args:
            step: 当前步数。
            audio: 生成的音频数组。
            sample_rate: 采样率。
            prompt: 生成使用的提示。
        """
        import soundfile as sf

        sample_path = self.sample_dir / f"sample_step_{step:06d}.wav"
        sf.write(str(sample_path), audio, sample_rate)

        if self.tb_writer and self.tensorboard_enabled:
            # TensorBoard 音频记录
            try:
                self.tb_writer.add_audio(
                    f"generated/step_{step}",
                    audio.reshape(1, -1),
                    step,
                    sample_rate=sample_rate,
                )
            except Exception:
                pass

        logger.debug(f"生成样本已保存: {sample_path}")

    def on_train_end(self) -> Dict[str, Any]:
        """训练结束回调。返回训练总结。"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        summary = {
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": f"{elapsed/3600:.1f} 小时",
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "avg_loss": float(np.mean(self.metrics_history.get("loss", [0]))),
        }

        if self.tb_writer:
            self.tb_writer.close()

        logger.info(f"=== 训练结束 === {summary['elapsed_formatted']}")
        logger.info(f"最佳损失: {self.best_loss:.4f} (Step {self.best_step})")

        # 保存总结到文件
        summary_path = self.log_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def _compute_speed(self) -> float:
        """计算训练速度（steps/s）。"""
        if len(self.step_timestamps) < 2:
            return 0.0
        recent = self.step_timestamps[-min(10, len(self.step_timestamps)):]
        elapsed = recent[-1] - recent[0]
        if elapsed <= 0:
            return 0.0
        return (len(recent) - 1) / elapsed

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的统计摘要。"""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    "current": values[-1],
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std": float(np.std(values)),
                }
        return summary

    def export_metrics(self, path: Union[str, Path]) -> None:
        """将指标历史导出为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "metrics": dict(self.metrics_history),
            "summary": self.get_metrics_summary(),
            "best_loss": self.best_loss,
            "best_step": self.best_step,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"指标已导出: {path}")
