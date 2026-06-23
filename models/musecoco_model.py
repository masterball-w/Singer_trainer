"""MuseCoco 模型实现：基于符号音乐生成的 MuseCoco 架构。

MuseCoco 是一个专注于符号音乐（MIDI）生成的模型，
通过学习音乐的结构和风格特征来生成新的音乐作品。
支持文本属性条件生成和风格迁移。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .base_model import BaseMusicModel


class MuseCocoModel(BaseMusicModel):
    """MuseCoco 符号音乐生成模型。

    MuseCoco 采用两阶段生成：先通过属性编码器提取音乐特征，
    再通过解码器生成符号音乐序列。支持风格条件生成。

    Attributes:
        model: MuseCoco 核心模型。
        tokenizer: 音乐 tokenizer。
    """

    SAMPLE_RATE = 32000

    def __init__(
        self,
        model_name: str = "musecoco",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            model_name=model_name,
            architecture="musecoco",
            device=device,
            dtype=dtype,
        )
        self.tokenizer = None

    def load_pretrained(self, model_path: Union[str, Path]) -> None:
        """加载 MuseCoco 预训练模型。

        Args:
            model_path: 本地模型目录路径。
        """
        model_path = Path(model_path)
        logger.info(f"正在加载 MuseCoco 模型: {model_path}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # 尝试使用 transformers 加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            self.to_device()
            self.model.eval()
            self.is_loaded = True

            params = self.count_parameters()
            logger.info(
                f"MuseCoco 模型加载成功: "
                f"总参数 {params['total']:,}, "
                f"可训练 {params['trainable']:,}"
            )
        except Exception as e:
            logger.error(f"MuseCoco 模型加载失败: {e}")
            raise RuntimeError(f"MuseCoco 模型加载失败: {e}") from e

    def prepare_for_training(
        self,
        learning_rate: float = 5e-5,
        freeze_layers: Optional[List[str]] = None,
    ) -> None:
        """准备 MuseCoco 进行微调训练。

        默认冻结编码器层，仅微调解码器和注意力层。

        Args:
            learning_rate: 学习率。
            freeze_layers: 需要冻结的层名称前缀列表。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")

        if freeze_layers is None:
            freeze_layers = ["encoder", "embed_tokens"]

        # 解冻所有层
        for param in self.model.parameters():
            param.requires_grad = True

        # 冻结指定层
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_layers):
                param.requires_grad = False

        self.model.train()

        params = self.count_parameters()
        logger.info(
            f"MuseCoco 训练准备完成: "
            f"可训练参数 {params['trainable']:,} / "
            f"总参数 {params['total']:,}"
        )

    def forward_training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """执行 MuseCoco 的一个训练前向步骤。

        Args:
            batch: 训练数据批次，应包含：
                - input_ids: token 输入 (B, T)
                - labels: 目标标签 (B, T)
                - attention_mask: 注意力掩码 (B, T)

        Returns:
            (loss, metrics) 元组。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")

        self.model.train()

        model_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "labels": batch["labels"].to(self.device),
        }
        if "attention_mask" in batch:
            model_inputs["attention_mask"] = batch["attention_mask"].to(self.device)

        outputs = self.model(**model_inputs)

        loss = outputs.loss
        metrics = {
            "loss": loss.item(),
            "perplexity": float(torch.exp(loss).item()) if loss.item() < 20 else float("inf"),
        }

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
        """使用 MuseCoco 生成音乐。

        MuseCoco 生成符号音乐，随后通过简单的合成器转换为音频波形。

        Args:
            prompt: 文本属性描述，如 "Key=C, Tempo=120, Genre=Classical"。
            duration: 生成时长（秒）。
            temperature: 采样温度。
            top_k: Top-k 采样。
            top_p: Nucleus 采样。
            guidance_scale: CFG 缩放因子。

        Returns:
            生成的音频 numpy 数组。
        """
        if not self.is_loaded:
            raise RuntimeError("模型尚未加载")

        self.model.eval()

        # MuseCoco 的 token 生成
        max_new_tokens = int(duration * 16)  # MuseCoco 约 16 tokens/秒

        if prompt and self.tokenizer:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # 使用默认输入
            inputs = {
                "input_ids": torch.ones((1, 1), dtype=torch.long, device=self.device)
            }

        logger.info(f"MuseCoco 开始生成: prompt='{prompt}', duration={duration}s")

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 0.01),
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
            )

        # 将 token 序列转换为音频
        # 使用简单的正弦波合成器将符号数据转换为音频
        token_ids = generated[0].cpu().numpy()
        audio = self._tokens_to_audio(token_ids, duration)

        logger.info(f"MuseCoco 生成完成: {len(audio)/self.SAMPLE_RATE:.2f}s")
        return audio

    def _tokens_to_audio(self, token_ids: np.ndarray, duration: float) -> np.ndarray:
        """将 token 序列转换为音频波形。

        使用简单的加法合成器将符号 token 映射到音频。
        这是一种基础的映射方法，实际部署时可使用更复杂的音频渲染器。

        Args:
            token_ids: token ID 数组。
            duration: 目标时长（秒）。

        Returns:
            音频 numpy 数组。
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        audio = np.zeros(num_samples, dtype=np.float32)

        # 将 token 映射到频率和时间参数
        if len(token_ids) == 0:
            return audio

        # 简单的加法合成
        num_notes = min(len(token_ids), 200)
        note_duration = num_samples / num_notes

        for i in range(num_notes):
            token = token_ids[i % len(token_ids)]
            # 将 token ID 映射到 MIDI 音符范围 (21-108)
            midi_note = 21 + (token % 88)
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

            start = int(i * note_duration)
            end = int(min(start + note_duration * 0.8, num_samples))
            t = np.arange(end - start) / self.SAMPLE_RATE

            # ADSR 包络
            envelope = np.ones_like(t)
            attack = int(0.02 * self.SAMPLE_RATE)
            release = int(0.05 * self.SAMPLE_RATE)
            if len(envelope) > attack:
                envelope[:attack] = np.linspace(0, 1, attack)
            if len(envelope) > release:
                envelope[-release:] = np.linspace(1, 0, release)

            # 生成正弦波 + 谐波
            wave = (
                0.6 * np.sin(2 * np.pi * freq * t)
                + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
                + 0.1 * np.sin(2 * np.pi * freq * 3 * t)
            )

            if start + len(wave) <= num_samples:
                audio[start : start + len(wave)] += wave * envelope * 0.3

        # 归一化
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9

        return audio

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取 MuseCoco 可训练参数。"""
        if self.model is None:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存 MuseCoco 检查点。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "architecture": self.architecture,
            "metadata": metadata or {},
        }

        torch.save(checkpoint, path / "pytorch_model.bin")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(path))

        logger.info(f"MuseCoco 检查点已保存: {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """从检查点恢复 MuseCoco 模型。"""
        path = Path(path)
        checkpoint_file = path / "pytorch_model.bin"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_file}")

        checkpoint = torch.load(str(checkpoint_file), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        metadata = checkpoint.get("metadata", {})
        logger.info(f"MuseCoco 检查点加载成功: {path}")
        return metadata

    def get_model_info(self) -> Dict[str, Any]:
        """获取 MuseCoco 模型信息。"""
        info = {
            "name": self.model_name,
            "architecture": "MuseCoco",
            "type": "Symbolic Music Generation",
            "sample_rate": self.SAMPLE_RATE,
            "tokenizer": "MuseCoco Tokenizer",
            "is_loaded": self.is_loaded,
            "device": str(self.device),
        }
        info.update(self.count_parameters())
        return info
