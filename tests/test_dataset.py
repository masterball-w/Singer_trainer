"""测试：数据集模块"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import soundfile as sf

from dataset.preprocessor import AudioPreprocessor
from dataset.feature_extractor import StyleFeatureExtractor


class TestAudioPreprocessor:
    """音频预处理器测试。"""

    def test_init(self):
        """测试预处理器初始化。"""
        pp = AudioPreprocessor(
            target_sample_rate=22050,
            slice_duration=10.0,
        )
        assert pp.target_sample_rate == 22050
        assert pp.slice_duration == 10.0

    def test_slice_audio(self):
        """测试音频切片。"""
        pp = AudioPreprocessor(
            target_sample_rate=16000,
            slice_duration=5.0,
            slice_overlap=1.0,
        )
        # 10 秒的音频
        audio = np.random.randn(160000).astype(np.float32)
        slices = pp._slice_audio(audio, 16000)

        assert len(slices) > 0
        for s in slices:
            assert len(s) == 80000  # 5s * 16000

    def test_normalize_peak(self):
        """测试峰值归一化。"""
        pp = AudioPreprocessor(normalization_method="peak")
        audio = np.random.randn(16000).astype(np.float32) * 5.0
        normalized = pp._normalize(audio)

        assert np.max(np.abs(normalized)) <= 1.0

    def test_process_file(self, tmp_path):
        """测试完整文件处理管线。"""
        # 创建测试音频
        sr = 16000
        duration = 10  # 10 秒
        audio = np.random.randn(sr * duration).astype(np.float32) * 0.5
        input_path = tmp_path / "test.wav"
        sf.write(str(input_path), audio, sr)

        # 处理
        output_dir = tmp_path / "output"
        pp = AudioPreprocessor(
            target_sample_rate=16000,
            slice_duration=5.0,
            slice_overlap=1.0,
            min_duration=2.0,
        )
        slices = pp.process_file(input_path, output_dir, target_sample_rate=16000)

        assert len(slices) > 0
        assert output_dir.exists()

        # 检查输出文件
        for s in slices:
            output_file = output_dir / s["filename"]
            assert output_file.exists()

    def test_short_audio_skipped(self, tmp_path):
        """测试短音频被跳过。"""
        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32)  # 2 秒
        input_path = tmp_path / "short.wav"
        sf.write(str(input_path), audio, sr)

        pp = AudioPreprocessor(min_duration=5.0)
        slices = pp.process_file(input_path, tmp_path / "output")
        assert len(slices) == 0


class TestStyleFeatureExtractor:
    """风格特征提取器测试。"""

    def test_extract_chroma(self):
        """测试 Chroma 特征提取。"""
        extractor = StyleFeatureExtractor(
            sample_rate=16000,
            feature_method="chroma",
        )
        audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 秒
        features = extractor.extract(audio, 16000)

        assert "chroma" in features
        assert features["chroma"].shape[0] == 12  # 12 chroma bins

    def test_extract_mfcc(self):
        """测试 MFCC 特征提取。"""
        extractor = StyleFeatureExtractor(
            sample_rate=16000,
            feature_method="mfcc",
            n_mfcc=13,
        )
        audio = np.random.randn(16000 * 5).astype(np.float32)
        features = extractor.extract(audio, 16000)

        assert "mfcc" in features
        assert features["mfcc"].shape[0] == 13

    def test_extract_combined(self):
        """测试组合特征提取。"""
        extractor = StyleFeatureExtractor(
            sample_rate=16000,
            feature_method="combined",
        )
        audio = np.random.randn(16000 * 5).astype(np.float32)
        features = extractor.extract(audio, 16000)

        assert "chroma" in features
        assert "mfcc" in features
        assert "spectral" in features
        assert "rhythm" in features
        assert "summary" in features

    def test_summary_contains_expected_keys(self):
        """测试摘要包含预期的键。"""
        extractor = StyleFeatureExtractor(
            sample_rate=16000,
            feature_method="combined",
        )
        audio = np.random.randn(16000 * 5).astype(np.float32)
        features = extractor.extract(audio, 16000)
        summary = features["summary"]

        assert "chroma_mean" in summary
        assert "mfcc_mean" in summary
        assert "tempo" in summary
        assert "dominant_pitch" in summary

    def test_feature_dimension(self):
        """测试特征维度计算。"""
        extractor = StyleFeatureExtractor(feature_method="combined")
        dim = extractor.get_feature_dimension()
        assert dim > 0
