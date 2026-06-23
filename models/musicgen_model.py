"""MusicGen 模型实现：基于 Meta 的 MusicGen 架构。

MusicGen 是一个自回归 Transformer 模型，使用 EnCodec 音频 tokenizer
将音频编码为离散 token，然后通过语言模型方式生成音乐。
支持文本条件和旋律条件生成。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .base_model import BaseMusicModel


class MusicGenModel(BaseMusicModel):
    """MusicGen 模型实现。

    支持 facebook/musicgen-small、musicgen-medium、musicgen-large
    和 musicgen-melody 等变体。

    Attributes:
        model: MusicGen 语言模型。
        processor: MusicGen 音频处理器。
        sample_rate: 模型输出采样率。
    """

    SAMPLE_RATE = 32000  # MusicGen 默认采样率

    def __init__(
        self,
        model_name: str = "musicgen-small",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            model_name=model_name,
            architecture="musicgen",
            device=device,
            dtype=dtype,
        )

    def load_pretrained(self, model_path: Union[str, Path]) -> None:
        """加载 MusicGen 预训练模型。

        使用 Hugging Face transformers 库加载 MusicGen 模型和处理器。

        Args:
            model_path: 本地模型目录路径。
        """
        from transformers import MusicgenForConditionalGeneration, AutoProcessor

        model_path = Path(model_path)
        logger.info(f"正在加载 MusicGen 模型: {model_path}")

        try:
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(str(model_path))

            # 加载模型
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
            )

            self.to_device()
            self.model.eval()
            self.is_loaded = True

            params = self.count_parameters()
            logger.info(
                f"MusicGen 模型加载成功: "
                f"总参数 {params['total']:,}, "
                f"可训练参数 {params['trainable']:,}"
            )
        except Exception as e:
            logger.error(f"MusicGen 模型加载失败: {e}")
            raise RuntimeError(f"MusicGen 模型加载失败: {e}") from e

    def prepare_for_training(
        self,
        learning_rate: float = 5e-5,
        freeze_layers: Optional[List[str]] = None,
    ) -> None:
        """准备 MusicGen 进行微调训练。

        默认冻结 EnCodec 音频编码器，仅微调 Transformer 语言模型部分。

        Args:
            learning_rate: 学习率。
            freeze_layers: 需要冻结的层名称前缀列表。
                默认为 ["audio_encoder"]，即冻结 EnCodec。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_pretrained()")

        if freeze_layers is None:
            freeze_layers = ["audio_encoder"]

        # 先冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻需要训练的层
        trainable_prefixes = [
            "decoder",
            "text_encoder",
        ]
        frozen_prefixes = freeze_layers

        for name, param in self.model.named_parameters():
            should_train = any(name.startswith(prefix) for prefix in trainable_prefixes)
            should_freeze = any(name.startswith(prefix) for prefix in frozen_prefixes)

            if should_train and not should_freeze:
                param.requires_grad = True

        # 设置模型为训练模式
        self.model.train()

        params = self.count_parameters()
        logger.info(
            f"MusicGen 训练准备完成: "
            f"可训练参数 {params['trainable']:,} / "
            f"总参数 {params['total']:,}"
        )

    def forward_training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """执行 MusicGen 的一个训练前向步骤。

        Args:
            batch: 训练数据批次，应包含：
                - input_values: 音频输入 (B, T)
                - labels: 目标标签 (B, T)
                - attention_mask: 注意力掩码 (B, T)（可选）

        Returns:
            (loss, metrics) 元组。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")

        self.model.train()

        # 构建模型输入
        model_inputs = {
            "input_values": batch["input_values"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = batch["attention_mask"].to(self.device)
        if "decoder_input_ids" in batch:
            model_inputs["decoder_input_ids"] = batch["decoder_input_ids"].to(self.device)

        # 前向传播
        outputs = self.model(**model_inputs)

        loss = outputs.loss
        metrics = {
            "loss": loss.item(),
            "perplexity": float(torch.exp(loss).item()) if loss.item() < 20 else float("inf"),
        }

        # 记录 logits 统计信息
        if hasattr(outputs, "logits") and outputs.logits is not None:
            metrics["logits_mean"] = outputs.logits.mean().item()
            metrics["logits_std"] = outputs.logits.std().item()

        return loss, metrics

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
        """使用 MusicGen 生成音乐。

        Args:
            prompt: 文本描述，如 "一首欢快的钢琴曲"。
            duration: 生成时长（秒）。
            temperature: 采样温度。
            top_k: Top-k 采样。
            top_p: Nucleus 采样。
            guidance_scale: CFG 缩放因子。

        Returns:
            生成的音频 numpy 数组，采样率为 32000。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")

        self.model.eval()

        # 计算生成的最大 token 数
        # MusicGen 大约每秒 50 个 token
        max_new_tokens = int(duration * 50)

        # 构建输入
        inputs = {}
        if prompt:
            inputs["text"] = [prompt]
        if "melody" in kwargs and kwargs["melody"] is not None:
            inputs["audio"] = [kwargs["melody"]]
            inputs["sampling_rate"] = self.SAMPLE_RATE

        if not inputs:
            inputs["text"] = [""]

        processed = self.processor(**inputs, return_tensors="pt")
        processed = {k: v.to(self.device) for k, v in processed.items()}

        logger.info(f"开始生成音乐: prompt='{prompt}', duration={duration}s")

        with torch.no_grad():
            audio_values = self.model.generate(
                **processed,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
                guidance_scale=guidance_scale,
            )

        # 转换为 numpy 数组
        audio = audio_values[0].cpu().numpy().squeeze()

        logger.info(f"音乐生成完成: {len(audio)/self.SAMPLE_RATE:.2f}s")
        return audio

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取 MusicGen 可训练参数。"""
        if self.model is None:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存 MusicGen 训练检查点。

        Args:
            path: 保存目录路径。
            metadata: 额外元数据。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存模型权重
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "architecture": self.architecture,
            "metadata": metadata or {},
        }

        torch.save(checkpoint, path / "pytorch_model.bin")

        # 保存处理器
        if self.processor is not None:
            self.processor.save_pretrained(str(path))

        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """从检查点恢复 MusicGen 模型。

        Args:
            path: 检查点目录路径。

        Returns:
            检查点元数据。
        """
        path = Path(path)
        checkpoint_file = path / "pytorch_model.bin"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_file}")

        checkpoint = torch.load(str(checkpoint_file), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        metadata = checkpoint.get("metadata", {})
        logger.info(f"从检查点加载成功: {path}")
        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """获取 MusicGen 模型详细信息。"""
        info = {
            "name": self.model_name,
            "architecture": "MusicGen",
            "type": "Autoregressive Transformer",
            "sample_rate": self.SAMPLE_RATE,
            "tokenizer": "EnCodec",
            "is_loaded": self.is_loaded,
            "device": str(self.device),
        }
        info.update(self.count_parameters())
        return info
