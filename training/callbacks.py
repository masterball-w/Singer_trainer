"""训练回调：模块化的训练事件处理。

提供检查点保存、早停法和生成预览等回调功能，
使训练过程更加灵活和可控。
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger


class TrainingCallback:
    """训练回调基类。

    所有回调应继承此类并实现需要的事件处理方法。
    """

    def on_train_start(self, **kwargs) -> None:
        """训练开始时调用。"""
        pass

    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        """训练轮开始时调用。"""
        pass

    def on_step_end(
        self, step: int, epoch: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        """训练步骤结束时调用。"""
        pass

    def on_eval_end(self, step: int, metrics: Dict[str, float], **kwargs) -> None:
        """评估结束时调用。"""
        pass

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """训练轮结束时调用。"""
        pass

    def on_train_end(self, **kwargs) -> None:
        """训练结束时调用。"""
        pass


class CheckpointCallback(TrainingCallback):
    """检查点保存回调。

    定期保存模型检查点，支持按步数和按性能两种触发方式。
    同时保留最近的 N 个检查点以管理磁盘空间。
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_every_n_steps: int = 1000,
        keep_last_n: int = 5,
        save_on_best: bool = True,
        monitor_metric: str = "loss",
    ):
        """初始化检查点回调。

        Args:
            checkpoint_dir: 检查点保存目录。
            save_every_n_steps: 每隔多少步保存一次。
            keep_last_n: 保留最近的 N 个检查点。
            save_on_best: 是否在指标最优时额外保存。
            monitor_metric: 监控的指标名称。
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.save_on_best = save_on_best
        self.monitor_metric = monitor_metric

        self.best_value = float("inf") if monitor_metric == "loss" else float("-inf")
        self.saved_checkpoints = []

    def on_step_end(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        """在指定步数保存检查点。"""
        model = kwargs.get("model")
        if model is None:
            return

        should_save = False

        # 按步数保存
        if step % self.save_every_n_steps == 0:
            should_save = True

        # 按最佳指标保存
        if self.save_on_best and self.monitor_metric in metrics:
            value = metrics[self.monitor_metric]
            is_better = (
                value < self.best_value
                if self.monitor_metric == "loss"
                else value > self.best_value
            )
            if is_better:
                self.best_value = value
                should_save = True

        if should_save:
            self._save_checkpoint(model, step, epoch, metrics)

    def _save_checkpoint(
        self,
        model: Any,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """执行检查点保存。"""
        checkpoint_name = f"checkpoint_step_{step:06d}_epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            model.save_checkpoint(checkpoint_path, metadata=metadata)
            self.saved_checkpoints.append(checkpoint_path)
            logger.info(f"检查点已保存: {checkpoint_path}")

            # 清理旧检查点
            self._cleanup_old_checkpoints()
        except Exception as e:
            logger.error(f"检查点保存失败: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """清理超出保留数量的旧检查点。"""
        while len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.debug(f"已清理旧检查点: {old_checkpoint}")


class EarlyStoppingCallback(TrainingCallback):
    """早停法回调。

    当监控指标在指定耐心轮数内不再改善时停止训练。
    """

    def __init__(
        self,
        monitor_metric: str = "eval_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """初始化早停法回调。

        Args:
            monitor_metric: 监控的指标名称。
            patience: 允许的无改善轮数。
            min_delta: 最小改善幅度。
            mode: "min" 表示指标越小越好，"max" 表示越大越好。
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def on_eval_end(
        self, step: int, metrics: Dict[str, float], **kwargs
    ) -> None:
        """评估结束后检查是否应该停止训练。"""
        if self.monitor_metric not in metrics:
            return

        value = metrics[self.monitor_metric]

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
            logger.debug(f"早停法: 指标改善 ({value:.4f})")
        else:
            self.counter += 1
            logger.debug(
                f"早停法: 无改善 ({self.counter}/{self.patience})"
            )

            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"早停法触发: {self.patience} 次评估无改善，"
                    f"最佳 {self.monitor_metric}={self.best_value:.4f}"
                )


class SampleGenerationCallback(TrainingCallback):
    """生成预览回调。

    在训练过程中定期生成音乐样本，用于直观评估训练效果。
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        generate_every_n_steps: int = 2000,
        prompts: Optional[list] = None,
        duration: float = 15.0,
        sample_rate: int = 32000,
    ):
        """初始化生成预览回调。

        Args:
            output_dir: 样本输出目录。
            generate_every_n_steps: 每隔多少步生成一次。
            prompts: 生成使用的文本提示列表。
            duration: 生成时长（秒）。
            sample_rate: 采样率。
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generate_every_n_steps = generate_every_n_steps
        self.prompts = prompts or ["一段欢快的音乐", "舒缓的钢琴曲"]
        self.duration = duration
        self.sample_rate = sample_rate

    def on_step_end(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        """在指定步数生成预览样本。"""
        if step % self.generate_every_n_steps != 0:
            return

        model = kwargs.get("model")
        monitor = kwargs.get("monitor")
        if model is None:
            return

        logger.info(f"正在生成预览样本 (Step {step})...")

        for i, prompt in enumerate(self.prompts):
            try:
                audio = model.generate(
                    prompt=prompt,
                    duration=self.duration,
                    temperature=1.0,
                )

                # 保存样本
                sample_path = self.output_dir / f"sample_step_{step:06d}_prompt_{i}.wav"

                import soundfile as sf
                sf.write(str(sample_path), audio, self.sample_rate)

                # 通知监控器
                if monitor:
                    monitor.on_sample_generated(step, audio, self.sample_rate, prompt)

                logger.debug(f"预览样本已保存: {sample_path}")

            except Exception as e:
                logger.error(f"预览样本生成失败: {e}")
