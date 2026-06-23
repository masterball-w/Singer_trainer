"""音乐生成模型基类：定义所有模型架构的统一接口。

所有具体模型实现（MusicGen、MuseCoco 等）都应继承此基类，
并实现其抽象方法，以确保训练和生成管线的统一性。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class BaseMusicModel(ABC):
    """音乐生成模型抽象基类。

    定义了模型加载、训练、推理和保存的统一接口。
    所有具体模型必须继承此类并实现抽象方法。

    Attributes:
        model_name: 模型标识名称。
        architecture: 模型架构名称。
        device: 计算设备。
        model: 底层神经网络模型。
        is_loaded: 模型是否已加载。
    """

    def __init__(
        self,
        model_name: str,
        architecture: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """初始化基类。

        Args:
            model_name: 模型标识名称。
            architecture: 架构名称。
            device: 计算设备。
            dtype: 模型参数精度。
        """
        self.model_name = model_name
        self.architecture = architecture
        self.device = device
        self.dtype = dtype
        self.model: Optional[nn.Module] = None
        self.processor = None
        self.is_loaded = False

    @abstractmethod
    def load_pretrained(self, model_path: Union[str, Path]) -> None:
        """从本地路径加载预训练模型。

        Args:
            model_path: 模型目录或文件路径。
        """
        ...

    @abstractmethod
    def prepare_for_training(
        self,
        learning_rate: float = 5e-5,
        freeze_layers: Optional[List[str]] = None,
    ) -> None:
        """准备模型进行微调训练。

        包括设置梯度、冻结特定层等操作。

        Args:
            learning_rate: 学习率。
            freeze_layers: 需要冻结的层名称列表。
        """
        ...

    @abstractmethod
    def forward_training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """执行一个训练前向步骤。

        Args:
            batch: 训练数据批次字典。

        Returns:
            (loss, metrics) 元组，其中 metrics 包含额外的训练指标。
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        duration: float = 30.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        **kwargs,
    ) -> np.ndarray:
        """生成音乐音频。

        Args:
            prompt: 文本描述提示（可选）。
            duration: 生成时长（秒）。
            temperature: 温度参数，控制随机性。
            top_k: Top-k 采样参数。
            top_p: Top-p (nucleus) 采样参数。
            guidance_scale: Classifier-free guidance 缩放因子。
            **kwargs: 模型特定的额外参数。

        Returns:
            生成的音频 numpy 数组。
        """
        ...

    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练的参数列表。

        Returns:
            需要梯度的参数列表。
        """
        ...

    @abstractmethod
    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存模型检查点。

        Args:
            path: 检查点保存路径。
            metadata: 额外元数据（训练步数、损失等）。
        """
        ...

    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """从检查点加载模型。

        Args:
            path: 检查点文件路径。

        Returns:
            检查点中的元数据字典。
        """
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。

        Returns:
            包含模型架构、参数量等信息的字典。
        """
        ...

    def to_device(self, device: Optional[torch.device] = None) -> None:
        """将模型移到指定设备。

        Args:
            device: 目标设备。若为 None 则使用初始化时指定的设备。
        """
        target_device = device or self.device
        if self.model is not None:
            self.model = self.model.to(target_device)
            logger.debug(f"模型已移至 {target_device}")

    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数数量。

        Returns:
            包含 total、trainable、frozen 参数数量的字典。
        """
        if self.model is None:
            return {"total": 0, "trainable": 0, "frozen": 0}

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"name={self.model_name}, "
            f"arch={self.architecture}, "
            f"params={params['total']:,}, "
            f"device={self.device})"
        )
