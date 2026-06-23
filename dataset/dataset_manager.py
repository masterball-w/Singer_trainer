"""数据集管理器：管理本地音乐数据集的完整生命周期。

负责音乐文件的发现、导入、预处理管线的组织，
以及 PyTorch Dataset 的创建。支持增量导入和元数据追踪。
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from .preprocessor import AudioPreprocessor


class MusicAudioDataset(Dataset):
    """PyTorch 音乐音频数据集。

    将预处理后的音频切片加载为 PyTorch 可训练的 Dataset。

    Attributes:
        audio_dir: 预处理后的音频目录。
        metadata: 数据集元数据列表。
        slice_length: 每个样本的目标采样点数。
    """

    def __init__(
        self,
        audio_dir: Union[str, Path],
        metadata: List[Dict[str, Any]],
        slice_length: int,
        sample_rate: int = 32000,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata = metadata
        self.slice_length = slice_length
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本。

        Returns:
            包含以下键的字典：
            - audio: 音频波形 (num_samples,)
            - input_values: 模型输入（与 audio 相同）
            - labels: 模型标签（与 audio 相同）
            - sample_id: 样本 ID
        """
        import soundfile as sf

        item = self.metadata[idx]
        audio_path = self.audio_dir / item["filename"]

        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        audio, sr = sf.read(str(audio_path), dtype="float32")

        # 确保长度一致
        if len(audio) < self.slice_length:
            padding = np.zeros(self.slice_length - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, padding])
        elif len(audio) > self.slice_length:
            audio = audio[: self.slice_length]

        audio_tensor = torch.from_numpy(audio)

        return {
            "audio": audio_tensor,
            "input_values": audio_tensor,
            "labels": audio_tensor,
            "sample_id": torch.tensor(idx, dtype=torch.long),
        }


class MusicDatasetManager:
    """音乐数据集管理器。

    管理音乐数据集的导入、预处理、分割和加载全流程。

    Attributes:
        dataset_dir: 数据集根目录。
        preprocessor: 音频预处理器。
    """

    METADATA_FILE = "dataset_metadata.json"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "./music_dataset",
        target_sample_rate: int = 32000,
        slice_duration: float = 30.0,
        slice_overlap: float = 2.0,
        supported_formats: Optional[List[str]] = None,
    ):
        """初始化数据集管理器。

        Args:
            dataset_dir: 数据集根目录。
            target_sample_rate: 目标采样率。
            slice_duration: 切片时长（秒）。
            slice_overlap: 切片重叠（秒）。
            supported_formats: 支持的音频格式列表。
        """
        self.dataset_dir = Path(dataset_dir)
        self._setup_directories()

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=target_sample_rate,
            slice_duration=slice_duration,
            slice_overlap=slice_overlap,
            supported_formats=supported_formats,
        )

        self.target_sample_rate = target_sample_rate
        self.slice_duration = slice_duration
        self.slice_length = int(slice_duration * target_sample_rate)
        self._metadata = self._load_metadata()

    def _setup_directories(self) -> None:
        """创建数据集目录结构。"""
        dirs = [
            self.dataset_dir,
            self.dataset_dir / "raw",
            self.dataset_dir / "processed",
            self.dataset_dir / "slices",
            self.dataset_dir / "features",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """加载数据集元数据。"""
        meta_path = self.dataset_dir / self.METADATA_FILE
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": [], "stats": {}}

    def _save_metadata(self) -> None:
        """保存数据集元数据。"""
        meta_path = self.dataset_dir / self.METADATA_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    def import_music(
        self,
        source_path: Union[str, Path],
        copy_files: bool = True,
    ) -> int:
        """从外部导入音乐文件到数据集。

        支持导入单个文件或整个目录。

        Args:
            source_path: 源文件或目录路径。
            copy_files: 是否将文件复制到数据集目录。
                若为 False，则仅创建符号链接。

        Returns:
            成功导入的文件数量。
        """
        source_path = Path(source_path)
        imported = 0

        if source_path.is_file():
            files = [source_path]
        elif source_path.is_dir():
            files = list(source_path.rglob("*"))
            files = [f for f in files if f.suffix.lower() in self.preprocessor.supported_formats]
        else:
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        raw_dir = self.dataset_dir / "raw"

        for file_path in files:
            if file_path.suffix.lower() not in self.preprocessor.supported_formats:
                continue

            target = raw_dir / file_path.name

            # 避免覆盖
            if target.exists():
                logger.warning(f"文件已存在，跳过: {file_path.name}")
                continue

            if copy_files:
                shutil.copy2(file_path, target)
            else:
                target.symlink_to(file_path.resolve())

            # 记录到元数据
            self._metadata["files"].append(
                {
                    "original_path": str(file_path),
                    "raw_filename": file_path.name,
                    "imported": True,
                    "processed": False,
                }
            )
            imported += 1
            logger.debug(f"已导入: {file_path.name}")

        self._save_metadata()
        logger.info(f"导入完成: {imported} 个文件")
        return imported

    def preprocess_dataset(self) -> Dict[str, Any]:
        """对数据集中的所有音频文件执行预处理。

        预处理流程：格式统一 → 重采样 → 归一化 → 切片。

        Returns:
            预处理统计信息字典。
        """
        raw_dir = self.dataset_dir / "raw"
        processed_dir = self.dataset_dir / "processed"
        slices_dir = self.dataset_dir / "slices"

        audio_files = [
            f
            for f in raw_dir.iterdir()
            if f.suffix.lower() in self.preprocessor.supported_formats
        ]

        if not audio_files:
            logger.warning("数据集中没有音频文件可处理")
            return {"total_files": 0, "total_slices": 0}

        total_slices = 0
        processed_files = []
        errors = []

        logger.info(f"开始预处理 {len(audio_files)} 个音频文件...")

        for i, audio_file in enumerate(audio_files):
            try:
                # 预处理单个文件
                slices = self.preprocessor.process_file(
                    input_path=audio_file,
                    output_dir=slices_dir,
                    target_sample_rate=self.target_sample_rate,
                )

                total_slices += len(slices)
                processed_files.append(
                    {
                        "source": audio_file.name,
                        "slices": [s["filename"] for s in slices],
                        "duration": slices[0].get("original_duration", 0) if slices else 0,
                    }
                )

                # 更新元数据
                for entry in self._metadata["files"]:
                    if entry["raw_filename"] == audio_file.name:
                        entry["processed"] = True
                        entry["slice_count"] = len(slices)

                logger.debug(
                    f"[{i+1}/{len(audio_files)}] {audio_file.name}: "
                    f"{len(slices)} 个切片"
                )

            except Exception as e:
                errors.append({"file": audio_file.name, "error": str(e)})
                logger.error(f"处理失败 [{audio_file.name}]: {e}")

        # 更新统计信息
        self._metadata["stats"] = {
            "total_files": len(audio_files),
            "processed_files": len(processed_files),
            "total_slices": total_slices,
            "errors": len(errors),
        }
        self._save_metadata()

        stats = {
            "total_files": len(audio_files),
            "processed_files": len(processed_files),
            "total_slices": total_slices,
            "errors": errors,
        }

        logger.info(
            f"预处理完成: {stats['processed_files']}/{stats['total_files']} 文件, "
            f"{total_slices} 个切片"
        )
        return stats

    def create_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        split_ratio: float = 1.0,
    ) -> Union[DataLoader, tuple]:
        """创建 PyTorch DataLoader。

        Args:
            batch_size: 批处理大小。
            shuffle: 是否打乱数据。
            num_workers: 数据加载工作进程数。
            split_ratio: 训练集比例。若小于 1.0 则返回 (train, val) 两个 DataLoader。

        Returns:
            DataLoader，或 (train_loader, val_loader) 元组。
        """
        slices_dir = self.dataset_dir / "slices"

        # 收集所有切片文件
        slice_files = sorted(slices_dir.glob("*.wav"))

        if not slice_files:
            raise RuntimeError(
                "没有可用的训练数据。请先调用 import_music() 和 preprocess_dataset()。"
            )

        metadata = [
            {"filename": f.name, "index": i} for i, f in enumerate(slice_files)
        ]

        if split_ratio < 1.0:
            # 划分训练集和验证集
            np.random.shuffle(metadata)
            split_idx = int(len(metadata) * split_ratio)
            train_meta = metadata[:split_idx]
            val_meta = metadata[split_idx:]

            train_dataset = MusicAudioDataset(
                slices_dir, train_meta, self.slice_length, self.target_sample_rate
            )
            val_dataset = MusicAudioDataset(
                slices_dir, val_meta, self.slice_length, self.target_sample_rate
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            logger.info(
                f"数据加载器已创建: 训练集 {len(train_meta)} 样本, "
                f"验证集 {len(val_meta)} 样本"
            )
            return train_loader, val_loader
        else:
            dataset = MusicAudioDataset(
                slices_dir, metadata, self.slice_length, self.target_sample_rate
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

            logger.info(f"数据加载器已创建: {len(metadata)} 个样本")
            return loader

    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息。"""
        slices_dir = self.dataset_dir / "slices"
        slice_count = len(list(slices_dir.glob("*.wav")))

        stats = {
            "dataset_dir": str(self.dataset_dir),
            "raw_files": len(list((self.dataset_dir / "raw").iterdir())),
            "slices": slice_count,
            "metadata": self._metadata.get("stats", {}),
        }
        return stats

    def list_files(self) -> List[Dict[str, Any]]:
        """列出数据集中的所有文件及其状态。"""
        return self._metadata.get("files", [])
