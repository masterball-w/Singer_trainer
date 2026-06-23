"""音频预处理器：实现音频格式统一、采样率标准化、归一化和切片。

提供完整的音频预处理管线，将各种格式的音乐文件转换为
统一的格式，以便后续的训练和特征提取。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from loguru import logger


class AudioPreprocessor:
    """音频预处理器，实现完整的预处理管线。

    管线步骤：
    1. 加载音频（支持多种格式）
    2. 格式转换（统一为 WAV）
    3. 重采样（统一采样率）
    4. 归一化（音量标准化）
    5. 切片（固定长度片段）

    Attributes:
        target_sample_rate: 目标采样率。
        slice_duration: 切片时长（秒）。
        slice_overlap: 切片重叠时长（秒）。
        supported_formats: 支持的音频格式集合。
    """

    DEFAULT_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}

    def __init__(
        self,
        target_sample_rate: int = 32000,
        slice_duration: float = 30.0,
        slice_overlap: float = 2.0,
        min_duration: float = 5.0,
        max_duration: float = 600.0,
        normalization_method: str = "peak",
        target_db: float = -3.0,
        supported_formats: Optional[List[str]] = None,
    ):
        """初始化预处理器。

        Args:
            target_sample_rate: 目标采样率（Hz）。
            slice_duration: 每个切片的时长（秒）。
            slice_overlap: 相邻切片之间的重叠时长（秒）。
            min_duration: 最小音频时长（秒），短于此将被跳过。
            max_duration: 最大音频时长（秒），长于此将被截断。
            normalization_method: 归一化方法 ("peak", "rms", "loudness")。
            target_db: 归一化目标分贝值。
            supported_formats: 支持的格式列表。
        """
        self.target_sample_rate = target_sample_rate
        self.slice_duration = slice_duration
        self.slice_overlap = slice_overlap
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.normalization_method = normalization_method
        self.target_db = target_db

        if supported_formats:
            self.supported_formats = {
                f if f.startswith(".") else f".{f}" for f in supported_formats
            }
        else:
            self.supported_formats = self.DEFAULT_FORMATS

    def process_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_sample_rate: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """处理单个音频文件：加载 → 重采样 → 归一化 → 切片 → 保存。

        Args:
            input_path: 输入音频文件路径。
            output_dir: 输出目录。
            target_sample_rate: 目标采样率（覆盖默认值）。

        Returns:
            切片信息列表，每项包含 filename、duration、start_time 等。
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sr = target_sample_rate or self.target_sample_rate

        # 步骤 1：加载音频
        audio, orig_sr = self._load_audio(input_path)
        duration = len(audio) / orig_sr

        if duration < self.min_duration:
            logger.warning(
                f"音频过短，跳过: {input_path.name} ({duration:.1f}s < {self.min_duration}s)"
            )
            return []

        if duration > self.max_duration:
            max_samples = int(self.max_duration * orig_sr)
            audio = audio[:max_samples]
            logger.info(f"音频过长，截断至 {self.max_duration}s: {input_path.name}")

        # 步骤 2：重采样
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        # 步骤 3：归一化
        audio = self._normalize(audio)

        # 步骤 4：切片
        slices = self._slice_audio(audio, sr)

        # 步骤 5：保存切片
        base_name = input_path.stem
        slice_infos = []

        for i, slice_audio in enumerate(slices):
            filename = f"{base_name}_slice_{i:04d}.wav"
            output_path = output_dir / filename
            sf.write(str(output_path), slice_audio, sr, subtype="PCM_16")

            slice_infos.append(
                {
                    "filename": filename,
                    "source_file": input_path.name,
                    "slice_index": i,
                    "duration": len(slice_audio) / sr,
                    "sample_rate": sr,
                    "num_samples": len(slice_audio),
                    "original_duration": duration,
                }
            )

        logger.debug(
            f"预处理完成: {input_path.name} -> {len(slices)} 个切片"
        )
        return slice_infos

    def _load_audio(self, path: Path) -> tuple:
        """加载音频文件为单声道 numpy 数组。

        Args:
            path: 音频文件路径。

        Returns:
            (audio_array, sample_rate) 元组。
        """
        try:
            audio, sr = librosa.load(str(path), sr=None, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"音频加载失败 [{path}]: {e}") from e

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """归一化音频。

        Args:
            audio: 原始音频数组。

        Returns:
            归一化后的音频数组。
        """
        if len(audio) == 0:
            return audio

        if self.normalization_method == "peak":
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95
        elif self.normalization_method == "rms":
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10 ** (self.target_db / 20)
                audio = audio * (target_rms / rms)
        elif self.normalization_method == "loudness":
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10 ** (self.target_db / 20)
                audio = audio * (target_rms / rms)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _slice_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[np.ndarray]:
        """将音频切分为固定长度的重叠片段。

        Args:
            audio: 完整音频数组。
            sample_rate: 采样率。

        Returns:
            音频片段列表。
        """
        slice_samples = int(self.slice_duration * sample_rate)
        overlap_samples = int(self.slice_overlap * sample_rate)
        step = slice_samples - overlap_samples

        if step <= 0:
            raise ValueError("重叠时长必须小于切片时长")

        slices = []
        min_samples = int(self.min_duration * sample_rate)
        start = 0

        while start + min_samples <= len(audio):
            end = min(start + slice_samples, len(audio))
            chunk = audio[start:end]

            # 如果最后一个片段太短，填充静音
            if len(chunk) < slice_samples:
                padding = np.zeros(slice_samples - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, padding])

            slices.append(chunk)
            start += step

        if not slices:
            # 整个音频不足一个切片，填充后仍返回
            padding = np.zeros(max(0, slice_samples - len(audio)), dtype=np.float32)
            slices.append(np.concatenate([audio, padding]))

        return slices

    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """批量处理目录中的所有音频文件。

        Args:
            input_dir: 输入目录。
            output_dir: 输出目录。

        Returns:
            处理统计信息字典。
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        files = [
            f
            for f in input_dir.rglob("*")
            if f.suffix.lower() in self.supported_formats
        ]

        if not files:
            logger.warning(f"目录中没有找到音频文件: {input_dir}")
            return {"total": 0, "processed": 0, "slices": 0}

        total_slices = 0
        processed = 0
        errors = []

        for i, f in enumerate(files):
            try:
                slices = self.process_file(f, output_dir)
                total_slices += len(slices)
                processed += 1
                logger.info(f"[{i+1}/{len(files)}] {f.name}: {len(slices)} 切片")
            except Exception as e:
                errors.append({"file": str(f), "error": str(e)})
                logger.error(f"处理失败 [{f.name}]: {e}")

        return {
            "total": len(files),
            "processed": processed,
            "slices": total_slices,
            "errors": len(errors),
        }
