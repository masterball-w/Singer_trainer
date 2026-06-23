"""音乐生成器：统一的音乐生成接口。

提供用户友好的音乐生成API，支持文本提示生成、
风格条件生成和参数调节。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import soundfile as sf
from loguru import logger

from models.base_model import BaseMusicModel
from models.musicgen_model import MusicGenModel
from models.musecoco_model import MuseCocoModel
from generation.style_fusion import StyleFusion, StyleEmbedding, StyleAnalyzer
from utils.device import get_device


class MusicGenerator:
    """统一的音乐生成器。

    封装模型加载、风格融合和音频输出的完整生成管线。

    Attributes:
        model: 底层音乐生成模型。
        style_analyzer: 风格分析器。
        style_fusion: 风格融合模块。
        device: 计算设备。
    """

    def __init__(
        self,
        architecture: str = "musicgen",
        model_name: str = "musicgen-small",
        device: Optional[torch.device] = None,
        dtype: str = "float16",
    ):
        """初始化音乐生成器。

        Args:
            architecture: 模型架构 ("musicgen" 或 "musecoco")。
            model_name: 模型名称。
            device: 计算设备。None 则自动选择。
            dtype: 模型精度 ("float32", "float16", "bfloat16")。
        """
        self.architecture = architecture
        self.model_name = model_name
        self.device = device or get_device("auto")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(dtype, torch.float16)

        self.model: Optional[BaseMusicModel] = None
        self.style_analyzer = StyleAnalyzer()
        self.style_fusion: Optional[StyleFusion] = None
        self.style_embedding: Optional[StyleEmbedding] = None
        self._style_profiles: Dict[str, Dict] = {}

    def init_style_fusion(
        self,
        embedding_dim: int = 512,
        strategy: str = "adaptive",
        num_heads: int = 8,
    ) -> None:
        """初始化风格融合模块。

        在调用 generate() 之前调用此方法，可启用基于风格特征的
        条件化生成，而非仅使用文本提示拼接。

        Args:
            embedding_dim: 风格嵌入维度。
            strategy: 融合策略 ("weighted_avg", "attention", "adaptive")。
            num_heads: 注意力头数（仅 attention 策略使用）。
        """
        feature_dim = self.style_analyzer.feature_extractor.get_feature_dimension()
        self.style_embedding = StyleEmbedding(
            input_dim=feature_dim,
            embedding_dim=embedding_dim,
        ).to(self.device)

        self.style_fusion = StyleFusion(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            strategy=strategy,
        ).to(self.device)

        logger.info(
            f"风格融合模块已初始化: strategy={strategy}, "
            f"embedding_dim={embedding_dim}"
        )

    def compute_style_embedding(
        self,
        style_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ) -> Optional[torch.Tensor]:
        """计算融合后的风格嵌入向量。

        基于已分析的风格配置，通过 StyleEmbedding 和 StyleFusion
        生成条件向量。

        Args:
            style_names: 要融合的风格名称列表（来自 analyze_style 的结果）。
                若为 None 则使用所有已分析的风格。
            weights: 各风格的权重。

        Returns:
            融合后的风格嵌入 (1, embedding_dim)，若模块未初始化则返回 None。
        """
        if self.style_embedding is None or self.style_fusion is None:
            return None

        if not self._style_profiles:
            logger.warning("尚未分析风格，请先调用 analyze_style()")
            return None

        if style_names is None:
            style_names = list(self._style_profiles.keys())

        embeddings = []
        for name in style_names:
            if name not in self._style_profiles:
                logger.warning(f"风格 '{name}' 不存在，跳过")
                continue

            profile = self._style_profiles[name]
            summary = profile.get("summary", {})
            # 将 summary 中的数值特征转为向量
            numeric_values = [
                v for v in summary.values() if isinstance(v, (int, float))
            ]
            if not numeric_values:
                continue

            # 填充或截断到 feature_dim
            feature_dim = self.style_analyzer.feature_extractor.get_feature_dimension()
            feature_vec = np.zeros(feature_dim)
            for i, val in enumerate(numeric_values[:feature_dim]):
                feature_vec[i] = val

            tensor = torch.from_numpy(feature_vec).float().unsqueeze(0).to(self.device)
            emb = self.style_embedding(tensor)
            embeddings.append(emb)

        if not embeddings:
            return None

        # 融合
        fused = self.style_fusion(embeddings, weights=weights)
        return fused

    def load_model(self, model_path: Union[str, Path]) -> None:
        """加载预训练模型。

        Args:
            model_path: 模型目录路径。
        """
        if self.architecture == "musicgen":
            self.model = MusicGenModel(
                model_name=self.model_name,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.architecture == "musecoco":
            self.model = MuseCocoModel(
                model_name=self.model_name,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"不支持的模型架构: {self.architecture}")

        self.model.load_pretrained(model_path)
        logger.info(f"模型 {self.model_name} 加载完成")

    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """从训练检查点加载模型。

        Args:
            checkpoint_path: 检查点目录路径。

        Returns:
            检查点元数据。
        """
        if self.model is None:
            raise RuntimeError("请先调用 load_model() 加载基础模型")

        metadata = self.model.load_checkpoint(checkpoint_path)
        logger.info(f"从检查点加载完成: {checkpoint_path}")
        return metadata

    def analyze_style(self, dataset_dir: Union[str, Path]) -> Dict[str, Any]:
        """分析数据集的风格特征。

        分析结果将被缓存，供后续的 compute_style_embedding() 使用。

        Args:
            dataset_dir: 数据集目录（包含音频文件的目录）。

        Returns:
            风格分析结果。
        """
        profiles = self.style_analyzer.analyze_directory(dataset_dir)
        self._style_profiles = profiles  # 缓存用于风格融合
        avg_style = self.style_analyzer.get_average_style()
        logger.info(f"风格分析完成: {len(profiles)} 个风格配置，已缓存供融合使用")
        return {"profiles": profiles, "average_style": avg_style}

    def generate(
        self,
        prompt: Optional[str] = None,
        duration: float = 30.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        guidance_scale: float = 3.0,
        style_description: Optional[str] = None,
        style_names: Optional[List[str]] = None,
        style_weights: Optional[List[float]] = None,
        melody: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """生成音乐。

        支持两种风格条件化方式：
        1. 文本方式：通过 style_description 与 prompt 拼接
        2. 向量方式：通过 style_names 指定要融合的风格，
           使用 StyleFusion 模块计算条件向量

        Args:
            prompt: 文本描述提示。
            duration: 生成时长（秒）。
            temperature: 采样温度，控制随机性。较高值生成更多样化的音乐。
            top_k: Top-k 采样参数。
            top_p: Nucleus 采样参数。
            guidance_scale: Classifier-free guidance 缩放因子。
            style_description: 风格描述文本（会与 prompt 拼接）。
            style_names: 要融合的风格名称列表（来自 analyze_style 的结果）。
                启用向量方式风格条件化。
            style_weights: 各风格的权重（配合 style_names 使用）。
            melody: 参考旋律音频（仅 MusicGen melody 模型支持）。
            output_path: 保存生成音频的文件路径。若为 None 则不保存。

        Returns:
            生成的音频 numpy 数组。
        """
        if self.model is None:
            raise RuntimeError("请先加载模型")

        # 如果有向量方式风格条件，计算风格嵌入
        style_embed = None
        if style_names and self.style_embedding is not None:
            style_embed = self.compute_style_embedding(style_names, style_weights)
            if style_embed is not None:
                logger.info(
                    f"使用向量风格条件化: {len(style_names)} 个风格, "
                    f"嵌入维度={style_embed.shape[-1]}"
                )

        # 构建文本提示（文本方式风格条件化）
        final_prompt = prompt
        if style_description:
            if final_prompt:
                final_prompt = f"{final_prompt}, {style_description}"
            else:
                final_prompt = style_description

        # 如果有风格嵌入向量，将其摘要信息也附加到提示中
        # （当前模型接口使用文本条件，嵌入向量可用于未来的模型扩展）
        if style_embed is not None and not final_prompt:
            final_prompt = "styled music"

        # 调用模型生成
        audio = self.model.generate(
            prompt=final_prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            melody=melody,
        )

        # 保存到文件
        if output_path:
            self.save_audio(audio, output_path)

        return audio

    def save_audio(
        self,
        audio: np.ndarray,
        path: Union[str, Path],
        sample_rate: int = 32000,
    ) -> Path:
        """保存音频到文件。

        Args:
            audio: 音频数据。
            path: 输出文件路径。
            sample_rate: 采样率。

        Returns:
            保存的文件路径。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sample_rate)
        logger.info(f"音频已保存: {path}")
        return path

    def batch_generate(
        self,
        prompts: List[str],
        duration: float = 30.0,
        output_dir: Union[str, Path] = "./outputs",
        **kwargs,
    ) -> List[Path]:
        """批量生成音乐。

        Args:
            prompts: 文本提示列表。
            duration: 每段音乐的时长。
            output_dir: 输出目录。
            **kwargs: 额外的生成参数。

        Returns:
            生成的音频文件路径列表。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, prompt in enumerate(prompts):
            output_path = output_dir / f"generated_{i:04d}.wav"
            self.generate(
                prompt=prompt,
                duration=duration,
                output_path=output_path,
                **kwargs,
            )
            results.append(output_path)

        logger.info(f"批量生成完成: {len(results)} 个文件")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型的信息。"""
        if self.model is None:
            return {"status": "not_loaded"}
        return self.model.get_model_info()
