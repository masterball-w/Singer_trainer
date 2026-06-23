"""测试：训练模块"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
)
from training.monitor import TrainingMonitor


class TestTrainingMonitor:
    """训练监控器测试。"""

    def test_init(self, tmp_path):
        """测试监控器初始化。"""
        monitor = TrainingMonitor(
            log_dir=tmp_path / "logs",
            tensorboard_enabled=False,
        )
        assert monitor.log_dir.exists()

    def test_on_step_end(self, tmp_path):
        """测试步骤结束回调。"""
        monitor = TrainingMonitor(
            log_dir=tmp_path / "logs",
            tensorboard_enabled=False,
        )
        monitor.on_train_start()

        for step in range(100):
            monitor.on_step_end(
                step=step,
                epoch=0,
                metrics={"loss": 1.0 - step * 0.01},
            )

        assert monitor.current_step == 99
        assert len(monitor.metrics_history["loss"]) == 100

    def test_best_loss_tracking(self, tmp_path):
        """测试最佳损失追踪。"""
        monitor = TrainingMonitor(
            log_dir=tmp_path / "logs",
            tensorboard_enabled=False,
        )
        monitor.on_train_start()

        losses = [5.0, 3.0, 4.0, 2.0, 2.5]
        for i, loss in enumerate(losses):
            monitor.on_step_end(step=i, epoch=0, metrics={"loss": loss})

        assert monitor.best_loss == 2.0
        assert monitor.best_step == 3

    def test_metrics_summary(self, tmp_path):
        """测试指标摘要。"""
        monitor = TrainingMonitor(
            log_dir=tmp_path / "logs",
            tensorboard_enabled=False,
        )
        monitor.on_train_start()

        for step in range(10):
            monitor.on_step_end(
                step=step,
                epoch=0,
                metrics={"loss": float(step)},
            )

        summary = monitor.get_metrics_summary()
        assert "loss" in summary
        assert summary["loss"]["min"] == 0.0
        assert summary["loss"]["max"] == 9.0


class TestEarlyStoppingCallback:
    """早停法回调测试。"""

    def test_no_improvement_triggers_stop(self):
        """测试无改善时触发停止。"""
        cb = EarlyStoppingCallback(
            monitor_metric="eval_loss",
            patience=3,
            mode="min",
        )

        # 持续改善
        cb.on_eval_end(step=1, metrics={"eval_loss": 5.0})
        assert not cb.should_stop

        cb.on_eval_end(step=2, metrics={"eval_loss": 4.0})
        assert not cb.should_stop

        cb.on_eval_end(step=3, metrics={"eval_loss": 3.0})
        assert not cb.should_stop

        # 不再改善
        cb.on_eval_end(step=4, metrics={"eval_loss": 3.5})
        assert not cb.should_stop

        cb.on_eval_end(step=5, metrics={"eval_loss": 3.2})
        assert not cb.should_stop

        # 达到耐心极限
        cb.on_eval_end(step=6, metrics={"eval_loss": 3.1})
        assert cb.should_stop

    def test_improvement_resets_counter(self):
        """测试改善重置计数器。"""
        cb = EarlyStoppingCallback(
            monitor_metric="eval_loss",
            patience=3,
        )

        cb.on_eval_end(step=1, metrics={"eval_loss": 5.0})
        cb.on_eval_end(step=2, metrics={"eval_loss": 6.0})
        assert cb.counter == 1

        cb.on_eval_end(step=3, metrics={"eval_loss": 4.0})
        assert cb.counter == 0  # 改善，计数器重置
