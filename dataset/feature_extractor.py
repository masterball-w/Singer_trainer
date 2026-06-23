"""风格特征提取器：从音频中提取音乐风格特征。

支持多种音频特征提取方法，包括 MFCC、Chroma、频谱特征等，
用于后续的风格分析和融合。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
from loguru import logger


class StyleFeatureExtractor:
    """音乐风格特征提取器。

    从音频信号中提取多维度的风格特征，用于风格分析和条件生成。

    支持的特征类型：
    - Chroma: 音级特征（12 维），反映调性和和声
    - MFCC: 梅尔频率倒谱系数，反映音色
    - Spectral: 频谱特征（质心、带宽、滚降点等）
    - Rhythm: 节奏特征（节拍、速度、规律性）
    - Combined: 以上所有特征的组合

    Attributes:
        sample_rate: 音频采样率。
        feature_method: 使用的特征提取方法。
        n_mfcc: MFCC 系数数量。
        n_chroma: Chroma 特征维度。
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        feature_method: str = "combined",
        n_mfcc: int = 13,
        n_chroma: int = 12,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        """初始化特征提取器。

        Args:
            sample_rate: 音频采样率。
            feature_method: 特征提取方法。
                可选 "chroma", "mfcc", "spectral", "rhythm", "combined"。
            n_mfcc: MFCC 系数数量。
            n_chroma: Chroma 维度。
            n_fft: FFT 窗口大小。
            hop_length: FFT 跳跃长度。
            n_mels: Mel 滤波器组数量。
        """
        self.sample_rate = sample_rate
        self.feature_method = feature_method
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """从音频中提取风格特征。

        Args:
            audio: 音频波形数组。
            sample_rate: 采样率（覆盖默认值）。

        Returns:
            特征字典，键为特征名称，值为 numpy 数组。
        """
        sr = sample_rate or self.sample_rate
        features = {}

        if self.feature_method in ("chroma", "combined"):
            features["chroma"] = self._extract_chroma(audio, sr)

        if self.feature_method in ("mfcc", "combined"):
            features["mfcc"] = self._extract_mfcc(audio, sr)

        if self.feature_method in ("spectral", "combined"):
            features["spectral"] = self._extract_spectral(audio, sr)

        if self.feature_method in ("rhythm", "combined"):
            features["rhythm"] = self._extract_rhythm(audio, sr)

        # 计算综合统计信息
        features["summary"] = self._compute_summary(features)

        return features

    def extract_from_file(
        self,
        path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """从音频文件提取特征。

        Args:
            path: 音频文件路径。

        Returns:
            特征字典。
        """
        audio, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return self.extract(audio, sr)

    def _extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """提取 Chroma（音级）特征。

        Chroma 特征反映音乐的调性和和声结构，
        对音高八度不变，非常适合风格分析。

        Args:
            audio: 音频数组。
            sr: 采样率。

        Returns:
            Chroma 特征数组，形状 (n_chroma, time_frames)。
        """
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, n_chroma=self.n_chroma, hop_length=self.hop_length
        )
        return chroma

    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """提取 MFCC（梅尔频率倒谱系数）特征。

        MFCC 反映音色特征，是音乐风格识别的常用特征。

        Args:
            audio: 音频数组。
            sr: 采样率。

        Returns:
            MFCC 特征数组，形状 (n_mfcc, time_frames)。
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return mfcc

    def _extract_spectral(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """提取频谱特征。

        包括频谱质心、带宽、滚降点和平坦度。

        Args:
            audio: 音频数组。
            sr: 采样率。

        Returns:
            频谱特征字典。
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )

        return {
            "centroid": spectral_centroid,
            "bandwidth": spectral_bandwidth,
            "rolloff": spectral_rolloff,
            "flatness": spectral_flatness,
        }

    def _extract_rhythm(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取节奏特征。

        包括节拍追踪、速度估计和节奏规律性。

        Args:
            audio: 音频数组。
            sr: 采样率。

        Returns:
            节奏特征字典。
        """
        # 节拍追踪
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=self.hop_length
        )

        # Onset 强度包络
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sr, hop_length=self.hop_length
        )

        # 节奏规律性（自相关）
        rhythm_regularity = self._compute_rhythm_regularity(onset_env)

        return {
            "tempo": float(np.atleast_1d(tempo)[0]),
            "beat_count": len(beat_frames),
            "beat_times": librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length),
            "onset_envelope": onset_env,
            "rhythm_regularity": rhythm_regularity,
        }

    def _compute_rhythm_regularity(self, onset_env: np.ndarray) -> float:
        """计算节奏规律性分数。

        使用自相关方法评估节奏的规律程度。

        Args:
            onset_env: Onset 强度包络。

        Returns:
            规律性分数 [0, 1]，1 表示完全规律。
        """
        if len(onset_env) < 10:
            return 0.0

        autocorr = np.correlate(onset_env, onset_env, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if autocorr[0] == 0:
            return 0.0

        autocorr = autocorr / autocorr[0]

        # 找到最强周期
        if len(autocorr) > 2:
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                    peaks.append(autocorr[i])

            if peaks:
                return float(max(peaks))

        return 0.0

    def _compute_summary(self, features: Dict) -> Dict[str, float]:
        """计算特征的汇总统计信息。

        Args:
            features: 特征字典。

        Returns:
            统计信息字典。
        """
        summary = {}

        if "chroma" in features:
            chroma = features["chroma"]
            summary["chroma_mean"] = float(np.mean(chroma))
            summary["chroma_std"] = float(np.std(chroma))
            # 主导调性
            dominant_pitch = int(np.argmax(np.mean(chroma, axis=1)))
            pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            summary["dominant_pitch"] = pitch_names[dominant_pitch % 12]

        if "mfcc" in features:
            mfcc = features["mfcc"]
            summary["mfcc_mean"] = float(np.mean(mfcc))
            summary["mfcc_std"] = float(np.std(mfcc))
            # 音色亮度（高频 MFCC 均值）
            summary["timbral_brightness"] = float(np.mean(np.abs(mfcc[5:])))

        if "spectral" in features and isinstance(features["spectral"], dict):
            spectral = features["spectral"]
            if "centroid" in spectral:
                summary["spectral_centroid_mean"] = float(
                    np.mean(spectral["centroid"])
                )
            if "rolloff" in spectral:
                summary["spectral_rolloff_mean"] = float(np.mean(spectral["rolloff"]))

        if "rhythm" in features and isinstance(features["rhythm"], dict):
            rhythm = features["rhythm"]
            summary["tempo"] = rhythm.get("tempo", 0)
            summary["rhythm_regularity"] = rhythm.get("rhythm_regularity", 0)

        return summary

    def get_feature_dimension(self) -> int:
        """获取当前配置下的特征维度总数。"""
        dim = 0
        if self.feature_method in ("chroma", "combined"):
            dim += self.n_chroma
        if self.feature_method in ("mfcc", "combined"):
            dim += self.n_mfcc
        if self.feature_method in ("spectral", "combined"):
            dim += 4  # centroid, bandwidth, rolloff, flatness
        if self.feature_method in ("rhythm", "combined"):
            dim += 1  # tempo
        return dim
