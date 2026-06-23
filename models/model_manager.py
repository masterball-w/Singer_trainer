"""模型管理器：从 Hugging Face Hub 检索、下载和管理音乐生成模型。

该模块负责模型的自动发现、版本检查、下载和缓存管理，
确保始终使用最新的稳定版本模型。
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    model_info,
    snapshot_download,
)
from loguru import logger


class ModelManager:
    """模型管理器，负责从 Hugging Face Hub 获取和管理模型。

    支持自动检测最新稳定版、缓存管理、模型版本追踪等功能。

    Attributes:
        cache_dir: 模型缓存目录。
        api: Hugging Face API 客户端。
    """

    # 已注册的支持模型及其 Hugging Face 仓库信息
    SUPPORTED_MODELS = {
        "musicgen-small": {
            "repo_id": "facebook/musicgen-small",
            "architecture": "musicgen",
            "description": "MusicGen Small (300M 参数)",
        },
        "musicgen-medium": {
            "repo_id": "facebook/musicgen-medium",
            "architecture": "musicgen",
            "description": "MusicGen Medium (1.5B 参数)",
        },
        "musicgen-large": {
            "repo_id": "facebook/musicgen-large",
            "architecture": "musicgen",
            "description": "MusicGen Large (3.3B 参数)",
        },
        "musicgen-melody": {
            "repo_id": "facebook/musicgen-melody",
            "architecture": "musicgen",
            "description": "MusicGen Melody（旋律条件生成）",
        },
        "musecoco": {
            "repo_id": "musecoco/MuseCoco",
            "architecture": "musecoco",
            "description": "MuseCoco（符号音乐生成）",
        },
    }

    def __init__(self, cache_dir: Union[str, Path] = "./checkpoints/pretrained"):
        """初始化模型管理器。

        Args:
            cache_dir: 模型缓存目录路径。
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        self._metadata_file = self.cache_dir / "model_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """加载本地模型元数据记录。"""
        if self._metadata_file.exists():
            with open(self._metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """保存模型元数据到本地。"""
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)

    def list_available_models(self) -> List[Dict[str, str]]:
        """列出所有支持的音乐生成模型。

        Returns:
            模型信息列表，每项包含 name、repo_id、architecture 和 description。
        """
        models = []
        for name, info in self.SUPPORTED_MODELS.items():
            model_entry = {
                "name": name,
                "repo_id": info["repo_id"],
                "architecture": info["architecture"],
                "description": info["description"],
                "cached": name in self._metadata,
            }
            models.append(model_entry)
        return models

    def check_model_version(self, model_name: str) -> Dict[str, str]:
        """检查 Hugging Face Hub 上模型的最新版本信息。

        Args:
            model_name: 模型名称（如 "musicgen-small"）。

        Returns:
            包含最新版本信息的字典。

        Raises:
            ValueError: 模型名称不在支持列表中。
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}。"
                f"支持的模型: {list(self.SUPPORTED_MODELS.keys())}"
            )

        repo_id = self.SUPPORTED_MODELS[model_name]["repo_id"]

        try:
            info = model_info(repo_id)
            version_info = {
                "repo_id": repo_id,
                "sha": info.sha or "unknown",
                "last_modified": str(info.last_modified) if info.last_modified else "unknown",
                "pipeline_tag": info.pipeline_tag or "unknown",
                "tags": info.tags or [],
            }
            logger.info(f"模型 {model_name} 最新版本: SHA={version_info['sha'][:8]}")
            return version_info
        except Exception as e:
            logger.warning(f"无法获取模型 {model_name} 的版本信息: {e}")
            return {"repo_id": repo_id, "error": str(e)}

    def download_model(
        self,
        model_name: str,
        force_update: bool = False,
    ) -> Path:
        """从 Hugging Face Hub 下载模型。

        自动检查本地缓存，仅在需要时下载更新。

        Args:
            model_name: 模型名称。
            force_update: 是否强制下载最新版本。

        Returns:
            下载后的本地模型目录路径。

        Raises:
            ValueError: 不支持的模型名称。
            RuntimeError: 下载失败。
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}")

        repo_id = self.SUPPORTED_MODELS[model_name]["repo_id"]
        local_dir = self.cache_dir / model_name

        # 检查是否需要更新
        if local_dir.exists() and not force_update:
            if model_name in self._metadata:
                cached_sha = self._metadata[model_name].get("sha", "")
                remote_info = self.check_model_version(model_name)
                if cached_sha == remote_info.get("sha", ""):
                    logger.info(f"模型 {model_name} 已是最新版本，使用本地缓存")
                    return local_dir

        logger.info(f"正在从 Hugging Face Hub 下载模型: {repo_id}")

        try:
            # 使用 snapshot_download 下载完整模型
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )

            # 更新元数据
            remote_info = self.check_model_version(model_name)
            self._metadata[model_name] = {
                "repo_id": repo_id,
                "sha": remote_info.get("sha", "unknown"),
                "downloaded_at": datetime.now().isoformat(),
                "local_path": str(local_dir),
                "architecture": self.SUPPORTED_MODELS[model_name]["architecture"],
            }
            self._save_metadata()

            logger.info(f"模型 {model_name} 下载完成: {local_dir}")
            return local_dir

        except Exception as e:
            logger.error(f"模型下载失败 [{model_name}]: {e}")
            raise RuntimeError(f"模型下载失败: {e}") from e

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """获取已下载模型的本地路径。

        Args:
            model_name: 模型名称。

        Returns:
            模型本地路径，若未下载则返回 None。
        """
        local_dir = self.cache_dir / model_name
        if local_dir.exists():
            return local_dir
        return None

    def clean_cache(self, model_name: Optional[str] = None) -> None:
        """清理模型缓存。

        Args:
            model_name: 要清理的模型名称。若为 None 则清理所有缓存。
        """
        if model_name:
            target_dir = self.cache_dir / model_name
            if target_dir.exists():
                shutil.rmtree(target_dir)
                self._metadata.pop(model_name, None)
                self._save_metadata()
                logger.info(f"已清理模型缓存: {model_name}")
        else:
            for name in list(self._metadata.keys()):
                target_dir = self.cache_dir / name
                if target_dir.exists():
                    shutil.rmtree(target_dir)
            self._metadata.clear()
            self._save_metadata()
            logger.info("已清理所有模型缓存")

    def register_custom_model(
        self,
        name: str,
        repo_id: str,
        architecture: str,
        description: str = "",
    ) -> None:
        """注册自定义的 Hugging Face 模型。

        允许用户添加不在预置列表中的模型。

        Args:
            name: 自定义模型名称。
            repo_id: Hugging Face 仓库 ID。
            architecture: 模型架构（"musicgen" 或 "musecoco"）。
            description: 模型描述。
        """
        if architecture not in ("musicgen", "musecoco"):
            raise ValueError(f"不支持的架构: {architecture}")

        self.SUPPORTED_MODELS[name] = {
            "repo_id": repo_id,
            "architecture": architecture,
            "description": description,
        }
        logger.info(f"已注册自定义模型: {name} ({repo_id})")
