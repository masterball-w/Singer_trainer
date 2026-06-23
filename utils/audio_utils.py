"""音频工具函数：提供音频加载、保存、重采样、归一化和切片等基础操作。

所有音频数据以 numpy 数组表示，采样率以 int 表示。
遵循统一的音频处理管线规范。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
from loguru import logger


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """加载音频文件。

    Args:
        path: 音频文件路径。
        sample_rate: 目标采样率。若为 None 则保持原始采样率。
        mono: 是否转换为单声道。

    Returns:
        (audio_data, sample_rate) 元组。
        audio_data 形状为 (num_samples,) 单声道或 (num_channels, num_samples) 多声道。

    Raises:
        FileNotFoundError: 文件不存在。
        RuntimeError: 音频加载失败。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    try:
        audio, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
        logger.debug(f"已加载音频: {path.name}, 采样率: {sr}, 时长: {len(audio)/sr:.2f}s")
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"音频加载失败 [{path}]: {e}") from e


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int,
    format: Optional[str] = None,
) -> Path:
    """保存音频到文件。

    Args:
        path: 输出文件路径。
        audio: 音频数据数组。
        sample_rate: 采样率。
        format: 输出格式（如 "wav", "flac"）。若为 None 则从文件扩展名推断。

    Returns:
        保存的文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 确保音频数据类型正确
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 限制音频范围到 [-1.0, 1.0]
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(str(path), audio, sample_rate, format=format)
    logger.debug(f"已保存音频: {path}, 采样率: {sample_rate}")
    return path


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """重采样音频到目标采样率。

    Args:
        audio: 原始音频数据。
        orig_sr: 原始采样率。
        target_sr: 目标采样率。

    Returns:
        重采样后的音频数据。
    """
    if orig_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    logger.debug(f"重采样: {orig_sr} -> {target_sr}")
    return resampled


def normalize_audio(
    audio: np.ndarray,
    method: str = "peak",
    target_db: float = -3.0,
) -> np.ndarray:
    """归一化音频。

    Args:
        audio: 音频数据。
        method: 归一化方法。
            - "peak": 峰值归一化
            - "rms": RMS 归一化
            - "loudness": 响度归一化（近似）
        target_db: 目标分贝值（用于 rms 和 loudness 方法）。

    Returns:
        归一化后的音频数据。
    """
    if len(audio) == 0:
        return audio

    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
    elif method == "loudness":
        # 使用 RMS 作为响度的近似
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

    return np.clip(audio, -1.0, 1.0)


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    slice_duration: float = 30.0,
    overlap: float = 2.0,
    min_duration: float = 5.0,
) -> list:
    """将音频切分为固定长度的片段。

    Args:
        audio: 音频数据。
        sample_rate: 采样率。
        slice_duration: 每个片段的时长（秒）。
        overlap: 片段之间的重叠时长（秒）。
        min_duration: 最小片段时长（秒），短于此长度的尾部片段将被丢弃。

    Returns:
        音频片段列表，每个元素为 numpy 数组。
    """
    slice_samples = int(slice_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    min_samples = int(min_duration * sample_rate)
    step = slice_samples - overlap_samples

    if step <= 0:
        raise ValueError("重叠时长必须小于切片时长")

    slices = []
    start = 0

    while start < len(audio):
        end = start + slice_samples
        chunk = audio[start:end]

        if len(chunk) >= min_samples:
            slices.append(chunk)
        elif len(slices) > 0:
            # 将短尾部合并到最后一个片段
            pass

        start += step

    logger.debug(
        f"音频切片完成: {len(audio)/sample_rate:.1f}s -> {len(slices)} 个片段 "
        f"(每段 {slice_duration}s, 重叠 {overlap}s)"
    )
    return slices


def get_audio_duration(path: Union[str, Path]) -> float:
    """获取音频文件时长（秒）。

    Args:
        path: 音频文件路径。

    Returns:
        时长（秒）。
    """
    info = sf.info(str(path))
    return info.duration


def pad_audio(
    audio: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """填充音频到目标长度。

    Args:
        audio: 音频数据。
        target_length: 目标采样点数。
        pad_value: 填充值，默认为静音。

    Returns:
        填充后的音频数据。
    """
    if len(audio) >= target_length:
        return audio[:target_length]

    padding = np.full(target_length - len(audio), pad_value, dtype=audio.dtype)
    return np.concatenate([audio, padding])
