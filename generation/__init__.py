"""生成模块：音乐生成相关的组件"""

from .generator import MusicGenerator
from .style_fusion import StyleEmbedding, StyleFusion, StyleAnalyzer

__all__ = [
    "MusicGenerator",
    "StyleEmbedding",
    "StyleFusion",
    "StyleAnalyzer",
]
