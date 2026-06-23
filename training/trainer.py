"""训练器：实现完整的模型训练循环。

支持 PyTorch 训练框架，包括混合精度训练、梯度累积、
学习率调度、断点续训和实时监控等功能。
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from loguru import logger

from models.base_model import BaseMusicModel
from training.monitor import TrainingMonitor
from training.callbacks import TrainingCallback


class MusicTrainer:
    """音乐模型训练器。

    管理完整的训练流程，包括数据加载、前向/反向传播、
    优化器管理、学习率调度、评估和回调处理。

    Attributes:
        model: 待训练的模型。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        monitor: 训练监控器。
        callbacks: 回调列表。
        device: 计算设备。
    """

    def __init__(
        self,
        model: BaseMusicModel,
        device: torch.device,
        learning_rate: float = 5e-5,
        optimizer_type: str = "adamw",
        optimizer_params: Optional[Dict] = None,
        scheduler_type: str = "warmup_cosine",
        warmup_steps: int = 500,
        max_epochs: int = 100,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 4,
        gradient_clip_norm: float = 1.0,
        mixed_precision: bool = True,
        save_every_n_steps: int = 1000,
        eval_every_n_steps: int = 500,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        log_dir: Union[str, Path] = "./logs",
        seed: int = 42,
    ):
        """初始化训练器。

        Args:
            model: 待训练的音乐模型。
            device: 计算设备。
            learning_rate: 学习率。
            optimizer_type: 优化器类型 ("adamw", "adam", "sgd")。
            optimizer_params: 优化器额外参数。
            scheduler_type: 学习率调度器类型。
            warmup_steps: 预热步数。
            max_epochs: 最大训练轮数。
            max_steps: 最大训练步数（-1 表示不限制）。
            gradient_accumulation_steps: 梯度累积步数。
            gradient_clip_norm: 梯度裁剪范数。
            mixed_precision: 是否使用混合精度训练。
            save_every_n_steps: 检查点保存间隔。
            eval_every_n_steps: 评估间隔。
            checkpoint_dir: 检查点目录。
            log_dir: 日志目录。
            seed: 随机种子。
        """
        self.model = model
        self.device = device
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision and device.type == "cuda"
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        self._set_seed(seed)

        # 初始化优化器
        self.optimizer = self._create_optimizer(
            optimizer_type, learning_rate, optimizer_params or {}
        )

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler(scheduler_type, warmup_steps)

        # 初始化混合精度
        self.scaler = GradScaler() if self.mixed_precision else None

        # 初始化监控器
        self.monitor = TrainingMonitor(
            log_dir=log_dir,
            tensorboard_enabled=True,
            log_file=Path(log_dir) / "training.log",
            sample_dir=Path(log_dir) / "samples",
        )

        # 回调列表
        self.callbacks: List[TrainingCallback] = []

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self._should_stop = False

    def _set_seed(self, seed: int) -> None:
        """设置随机种子以确保可重复性。"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"随机种子已设置: {seed}")

    def _create_optimizer(
        self,
        optimizer_type: str,
        learning_rate: float,
        params: Dict,
    ) -> torch.optim.Optimizer:
        """创建优化器。

        Args:
            optimizer_type: 优化器类型。
            learning_rate: 学习率。
            params: 优化器参数。

        Returns:
            优化器实例。
        """
        trainable_params = self.model.get_trainable_parameters()

        if not trainable_params:
            raise RuntimeError("没有可训练的参数")

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=params.get("weight_decay", 0.01),
                betas=tuple(params.get("betas", [0.9, 0.999])),
                eps=params.get("eps", 1e-8),
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                trainable_params,
                lr=learning_rate,
                weight_decay=params.get("weight_decay", 0),
                betas=tuple(params.get("betas", [0.9, 0.999])),
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                trainable_params,
                lr=learning_rate,
                momentum=params.get("momentum", 0.9),
                weight_decay=params.get("weight_decay", 0),
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    def _create_scheduler(
        self,
        scheduler_type: str,
        warmup_steps: int,
    ):
        """创建学习率调度器。

        Args:
            scheduler_type: 调度器类型。
            warmup_steps: 预热步数。

        Returns:
            学习率调度器实例。
        """
        if scheduler_type == "constant":
            return None
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, total_iters=10000
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=1e-7
            )
        elif scheduler_type == "warmup_cosine":
            # 自定义 warmup + cosine 调度
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(10000 - warmup_steps, 1)
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            logger.warning(f"未知的调度器类型: {scheduler_type}，使用默认")
            return None

    def add_callback(self, callback: TrainingCallback) -> None:
        """添加训练回调。

        Args:
            callback: 回调实例。
        """
        self.callbacks.append(callback)
        logger.debug(f"已添加回调: {callback.__class__.__name__}")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """执行完整的训练循环。

        Args:
            train_dataloader: 训练数据加载器。
            eval_dataloader: 验证数据加载器（可选）。

        Returns:
            训练总结字典。
        """
        logger.info(
            f"开始训练: max_epochs={self.max_epochs}, "
            f"max_steps={self.max_steps}, "
            f"device={self.device}"
        )

        # 通知回调和监控器
        config = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": train_dataloader.batch_size,
            "max_epochs": self.max_epochs,
            "optimizer": type(self.optimizer).__name__,
        }
        self.monitor.on_train_start(config)
        for cb in self.callbacks:
            cb.on_train_start(config=config)

        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                epoch_loss = self._train_epoch(train_dataloader, epoch)

                # 轮次结束回调
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss)

                if self._should_stop:
                    logger.info("早停法触发，训练结束")
                    break

                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    logger.info(f"达到最大步数 {self.max_steps}，训练结束")
                    break

        except KeyboardInterrupt:
            logger.info("用户中断训练")
            self._save_emergency_checkpoint()

        # 训练结束
        summary = self.monitor.on_train_end()
        for cb in self.callbacks:
            cb.on_train_end()

        return summary

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """执行一个训练轮次。

        Args:
            dataloader: 训练数据加载器。
            epoch: 当前轮次编号。

        Returns:
            本轮平均损失。
        """
        total_loss = 0.0
        num_batches = 0

        for cb in self.callbacks:
            cb.on_epoch_start(epoch=epoch)

        for batch_idx, batch in enumerate(dataloader):
            # 前向传播与反向传播
            loss, metrics = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            # 全局步数更新
            self.global_step += 1

            # 监控
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.monitor.on_step_end(
                step=self.global_step,
                epoch=epoch,
                metrics=metrics,
                lr=current_lr,
            )

            # 回调
            for cb in self.callbacks:
                cb.on_step_end(
                    step=self.global_step,
                    epoch=epoch,
                    metrics=metrics,
                    model=self.model,
                    monitor=self.monitor,
                )
                # 检查早停法
                if hasattr(cb, "should_stop") and cb.should_stop:
                    self._should_stop = True

            # 评估
            if (
                hasattr(self, "_eval_dataloader")
                and self.global_step % self.eval_every_n_steps == 0
            ):
                self._evaluate()

            # 步数限制检查
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                break

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """执行单个训练步骤。

        Args:
            batch: 数据批次。

        Returns:
            (loss_value, metrics_dict) 元组。
        """
        # 混合精度前向传播
        if self.mixed_precision and self.scaler:
            with autocast(dtype=torch.float16):
                loss, metrics = self.model.forward_training_step(batch)
                loss = loss / self.gradient_accumulation_steps

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度累积步骤
            if self.global_step % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.gradient_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()
        else:
            loss, metrics = self.model.forward_training_step(batch)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if self.global_step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    self.gradient_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps, metrics

    def _evaluate(self) -> Dict[str, float]:
        """在验证集上评估模型。

        Returns:
            评估指标字典。
        """
        if not hasattr(self, "_eval_dataloader") or self._eval_dataloader is None:
            return {}

        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self._eval_dataloader:
                _, metrics = self.model.forward_training_step(batch)
                total_loss += metrics.get("loss", 0)
                num_batches += 1

        eval_metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
        }

        self.monitor.on_eval_end(self.global_step, eval_metrics)

        for cb in self.callbacks:
            cb.on_eval_end(
                step=self.global_step,
                metrics=eval_metrics,
                model=self.model,
                monitor=self.monitor,
            )

        self.model.model.train()
        return eval_metrics

    def _save_emergency_checkpoint(self) -> None:
        """紧急保存检查点（用于中断恢复）。"""
        emergency_path = self.checkpoint_dir / "emergency_checkpoint"
        metadata = {
            "step": self.global_step,
            "epoch": self.current_epoch,
            "reason": "interrupted",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            self.model.save_checkpoint(emergency_path, metadata=metadata)

            # 保存优化器状态
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "current_epoch": self.current_epoch,
                },
                emergency_path / "optimizer_state.pt",
            )

            logger.info(f"紧急检查点已保存: {emergency_path}")
        except Exception as e:
            logger.error(f"紧急检查点保存失败: {e}")

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """从检查点恢复训练。

        恢复模型权重、优化器状态和训练进度。

        Args:
            checkpoint_path: 检查点目录路径。
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        # 恢复模型
        metadata = self.model.load_checkpoint(checkpoint_path)
        self.global_step = metadata.get("step", 0)
        self.current_epoch = metadata.get("epoch", 0)

        # 恢复优化器状态
        optimizer_state_path = checkpoint_path / "optimizer_state.pt"
        if optimizer_state_path.exists():
            state = torch.load(str(optimizer_state_path), map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.global_step = state.get("global_step", self.global_step)
            self.current_epoch = state.get("current_epoch", self.current_epoch)

        logger.info(
            f"从检查点恢复训练: step={self.global_step}, epoch={self.current_epoch}"
        )

    def save_training_state(self, path: Union[str, Path]) -> None:
        """保存完整的训练状态（模型 + 优化器 + 调度器）。

        Args:
            path: 保存目录。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_checkpoint(
            path,
            metadata={
                "step": self.global_step,
                "epoch": self.current_epoch,
                "best_loss": self.monitor.best_loss,
            },
        )

        # 保存优化器
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            path / "optimizer_state.pt",
        )

        logger.info(f"训练状态已保存: {path}")
