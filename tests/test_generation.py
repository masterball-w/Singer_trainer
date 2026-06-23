"""测试：生成模块"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from generation.style_fusion import StyleEmbedding, StyleFusion, StyleAnalyzer


class TestStyleEmbedding:
    """风格嵌入网络测试。"""

    def test_forward_shape(self):
        """测试前向传播输出形状。"""
        embedding = StyleEmbedding(input_dim=25, embedding_dim=512)
        x = torch.randn(4, 25)
        output = embedding(x)

        assert output.shape == (4, 512)

    def test_different_batch_sizes(self):
        """测试不同批大小。"""
        embedding = StyleEmbedding(input_dim=25, embedding_dim=128)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 25)
            output = embedding(x)
            assert output.shape == (batch_size, 128)


class TestStyleFusion:
    """风格融合模块测试。"""

    def test_weighted_avg_fusion(self):
        """测试加权平均融合。"""
        fusion = StyleFusion(embedding_dim=256, strategy="weighted_avg")
        embeddings = [torch.randn(2, 256) for _ in range(3)]

        result = fusion(embeddings, weights=[0.5, 0.3, 0.2])
        assert result.shape == (2, 256)

    def test_attention_fusion(self):
        """测试注意力融合。"""
        fusion = StyleFusion(embedding_dim=256, strategy="attention", num_heads=4)
        embeddings = [torch.randn(2, 256) for _ in range(3)]

        result = fusion(embeddings)
        assert result.shape == (2, 256)

    def test_adaptive_fusion(self):
        """测试自适应融合。"""
        fusion = StyleFusion(embedding_dim=256, strategy="adaptive")
        embeddings = [torch.randn(2, 256) for _ in range(3)]

        result = fusion(embeddings)
        assert result.shape == (2, 256)

    def test_single_embedding(self):
        """测试单个嵌入直接返回。"""
        fusion = StyleFusion(embedding_dim=256, strategy="weighted_avg")
        emb = torch.randn(2, 256)

        result = fusion([emb])
        assert torch.equal(result, emb)

    def test_empty_raises(self):
        """测试空列表抛出异常。"""
        fusion = StyleFusion(embedding_dim=256)
        with pytest.raises(ValueError):
            fusion([])


class TestStyleAnalyzer:
    """风格分析器测试。"""

    def test_init(self):
        """测试初始化。"""
        analyzer = StyleAnalyzer(sample_rate=16000)
        assert analyzer.feature_extractor.sample_rate == 16000

    def test_empty_average_style(self):
        """测试空数据集的平均风格。"""
        analyzer = StyleAnalyzer()
        avg = analyzer.get_average_style()
        assert avg == {}

    def test_analyze_audio_file(self, tmp_path):
        """测试分析音频文件。"""
        import soundfile as sf

        # 创建测试音频
        audio = np.random.randn(16000 * 5).astype(np.float32) * 0.5
        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, 16000)

        analyzer = StyleAnalyzer(sample_rate=16000)
        profiles = analyzer.analyze_directory(tmp_path)

        assert len(profiles) == 1
        assert "test" in profiles
        assert "summary" in profiles["test"]
