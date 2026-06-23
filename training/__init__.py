"""训练模块：模型训练、监控和回调"""

from .trainer import MusicTrainer
from .monitor import TrainingMonitor
from .callbacks import (
    TrainingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    SampleGenerationCallback,
)

__all__ = [
    "MusicTrainer",
    "TrainingMonitor",
    "TrainingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "SampleGenerationCallback",
]
