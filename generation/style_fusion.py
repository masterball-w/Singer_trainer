"""风格融合模块：分析和融合多种音乐风格特征。

实现风格特征的提取、嵌入和融合机制，使模型能够理解
并综合数据集中的多种音乐风格元素来生成新音乐。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from dataset.feature_extractor import StyleFeatureExtractor


class StyleEmbedding(nn.Module):
    """风格嵌入网络，将风格特征映射到嵌入空间。

    将多维风格特征压缩为固定维度的嵌入向量，
    用于条件化音乐生成模型。
    """

    def __init__(self, input_dim: int, embedding_dim: int = 512):
        """初始化风格嵌入网络。

        Args:
            input_dim: 输入特征维度。
            embedding_dim: 嵌入向量维度。
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入风格特征 (batch, input_dim)。

        Returns:
            风格嵌入向量 (batch, embedding_dim)。
        """
        return self.network(x)


class StyleFusion(nn.Module):
    """多风格融合模块。

    支持多种融合策略：
    - weighted_avg: 加权平均融合
    - attention: 基于注意力的融合
    - adaptive: 自适应门控融合
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        strategy: str = "adaptive",
    ):
        """初始化风格融合模块。

        Args:
            embedding_dim: 嵌入维度。
            num_heads: 注意力头数（用于 attention 策略）。
            strategy: 融合策略 ("weighted_avg", "attention", "adaptive")。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.strategy = strategy

        if strategy == "attention":
            # 多头注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(embedding_dim)

        elif strategy == "adaptive":
            # 自适应门控融合
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Sigmoid(),
            )
            self.transform = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
            )
            self.layer_norm = nn.LayerNorm(embedding_dim)

        elif strategy != "weighted_avg":
            raise ValueError(f"不支持的融合策略: {strategy}")

    def forward(
        self,
        embeddings: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """融合多个风格嵌入向量。

        Args:
            embeddings: 风格嵌入向量列表，每个形状为 (batch, embedding_dim)。
            weights: 各风格的权重（仅用于 weighted_avg 策略）。

        Returns:
            融合后的嵌入向量 (batch, embedding_dim)。
        """
        if len(embeddings) == 0:
            raise ValueError("至少需要一个风格嵌入")

        if len(embeddings) == 1:
            return embeddings[0]

        if self.strategy == "weighted_avg":
            return self._weighted_avg(embeddings, weights)
        elif self.strategy == "attention":
            return self._attention_fusion(embeddings)
        elif self.strategy == "adaptive":
            return self._adaptive_fusion(embeddings)
        else:
            raise ValueError(f"不支持的融合策略: {self.strategy}")

    def _weighted_avg(
        self,
        embeddings: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """加权平均融合。

        Args:
            embeddings: 嵌入向量列表。
            weights: 权重列表。

        Returns:
            加权平均后的嵌入向量。
        """
        if weights is None:
            weights = [1.0 / len(embeddings)] * len(embeddings)

        # 归一化权重
        total = sum(weights)
        weights = [w / total for w in weights]

        result = sum(w * e for w, e in zip(weights, embeddings))
        return result

    def _attention_fusion(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """基于注意力的融合。

        使用自注意力机制学习不同风格之间的关系。

        Args:
            embeddings: 嵌入向量列表。

        Returns:
            注意力融合后的嵌入向量。
        """
        # 将嵌入堆叠为序列 (batch, num_styles, embedding_dim)
        stacked = torch.stack(embeddings, dim=1)

        # 自注意力
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.layer_norm(stacked + attended)

        # 对所有风格的表示取平均
        return attended.mean(dim=1)

    def _adaptive_fusion(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """自适应门控融合。

        通过可学习的门控机制自适应地融合不同风格。

        Args:
            embeddings: 嵌入向量列表。

        Returns:
            自适应融合后的嵌入向量。
        """
        # 先计算平均值作为基础
        base = torch.stack(embeddings, dim=0).mean(dim=0)

        # 计算门控值
        gate_values = self.gate(base)
        transformed = self.transform(base)

        # 门控融合
        result = gate_values * transformed + (1 - gate_values) * base
        result = self.layer_norm(result)

        return result


class StyleAnalyzer:
    """音乐风格分析器。

    从数据集中分析并提取风格特征，为风格融合提供输入。
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        feature_method: str = "combined",
    ):
        """初始化风格分析器。

        Args:
            sample_rate: 音频采样率。
            feature_method: 特征提取方法。
        """
        self.feature_extractor = StyleFeatureExtractor(
            sample_rate=sample_rate,
            feature_method=feature_method,
        )
        self.style_profiles: Dict[str, Dict[str, Any]] = {}

    def analyze_directory(self, directory: Union[str, Path]) -> Dict[str, Dict]:
        """分析目录中所有音乐文件的风格特征。

        Args:
            directory: 音频文件目录。

        Returns:
            按文件组织的风格特征字典。
        """
        directory = Path(directory)
        supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        audio_files = [
            f for f in directory.rglob("*") if f.suffix.lower() in supported
        ]

        logger.info(f"开始分析 {len(audio_files)} 个音频文件的风格特征...")

        for audio_file in audio_files:
            try:
                features = self.feature_extractor.extract_from_file(audio_file)
                self.style_profiles[audio_file.stem] = {
                    "path": str(audio_file),
                    "features": features,
                    "summary": features.get("summary", {}),
                }
            except Exception as e:
                logger.error(f"分析失败 [{audio_file.name}]: {e}")

        logger.info(f"风格分析完成: {len(self.style_profiles)} 个文件")
        return self.style_profiles

    def get_average_style(self) -> Dict[str, float]:
        """计算所有已分析文件的平均风格特征。

        Returns:
            平均风格特征字典。
        """
        if not self.style_profiles:
            return {}

        all_summaries = [
            p["summary"] for p in self.style_profiles.values() if "summary" in p
        ]

        if not all_summaries:
            return {}

        # 计算所有数值特征的平均值
        avg_style = {}
        keys = all_summaries[0].keys()
        for key in keys:
            values = [s[key] for s in all_summaries if key in s and isinstance(s[key], (int, float))]
            if values:
                avg_style[key] = float(np.mean(values))

        return avg_style

    def get_style_similarity(
        self, style_a: str, style_b: str
    ) -> float:
        """计算两个风格之间的相似度。

        Args:
            style_a: 第一个风格的名称。
            style_b: 第二个风格的名称。

        Returns:
            相似度分数 [0, 1]。
        """
        if style_a not in self.style_profiles or style_b not in self.style_profiles:
            return 0.0

        summary_a = self.style_profiles[style_a].get("summary", {})
        summary_b = self.style_profiles[style_b].get("summary", {})

        # 提取数值特征
        keys = set(summary_a.keys()) & set(summary_b.keys())
        vec_a = np.array([summary_a[k] for k in keys if isinstance(summary_a.get(k), (int, float))])
        vec_b = np.array([summary_b[k] for k in keys if isinstance(summary_b.get(k), (int, float))])

        if len(vec_a) == 0 or len(vec_b) == 0:
            return 0.0

        # 余弦相似度
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return (similarity + 1) / 2  # 映射到 [0, 1]
