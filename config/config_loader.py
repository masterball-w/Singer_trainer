"""配置加载器：负责读取、合并和管理 YAML 配置文件。

该模块提供了统一的配置管理接口，支持默认配置与用户自定义配置的合并，
以及运行时参数的动态覆盖。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigLoader:
    """配置加载器，管理项目的所有配置项。

    Attributes:
        config: 合并后的完整配置字典。
        config_path: 当前加载的配置文件路径。
    """

    _DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"
    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """初始化配置加载器。

        Args:
            config_path: 用户自定义配置文件路径。若为 None 则仅加载默认配置。
            overrides: 运行时参数覆盖字典，使用点分隔的键名。
                例如 {"training.batch_size": 8} 将覆盖 training.batch_size。
        """
        self.config_path = config_path
        self._config = self._load_config(config_path, overrides)

    def _load_config(
        self,
        config_path: Optional[Union[str, Path]],
        overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """加载并合并配置文件。

        Args:
            config_path: 用户配置文件路径。
            overrides: 参数覆盖字典。

        Returns:
            合并后的配置字典。
        """
        # 加载默认配置
        config = self._read_yaml(self._DEFAULT_CONFIG_PATH)

        # 合并用户配置
        if config_path is not None:
            user_config = self._read_yaml(Path(config_path))
            config = self._deep_merge(config, user_config)

        # 应用运行时覆盖
        if overrides:
            for key, value in overrides.items():
                self._set_nested(config, key, value)

        return config

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        """读取 YAML 文件。

        Args:
            path: YAML 文件路径。

        Returns:
            解析后的字典。

        Raises:
            FileNotFoundError: 配置文件不存在。
            yaml.YAMLError: YAML 解析错误。
        """
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并两个字典，override 中的值覆盖 base 中的值。

        Args:
            base: 基础字典。
            override: 覆盖字典。

        Returns:
            合并后的新字典。
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _set_nested(config: Dict, key: str, value: Any) -> None:
        """设置嵌套字典中的值。

        Args:
            config: 目标字典。
            key: 点分隔的键名，如 "training.batch_size"。
            value: 要设置的值。
        """
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点分隔的键名。

        Args:
            key: 配置键名，如 "training.batch_size"。
            default: 键不存在时的默认值。

        Returns:
            配置项的值。
        """
        keys = key.split(".")
        current = self._config
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return default
            current = current[k]
        return current

    def set(self, key: str, value: Any) -> None:
        """动态设置配置项。

        Args:
            key: 点分隔的键名。
            value: 要设置的值。
        """
        self._set_nested(self._config, key, value)

    @property
    def config(self) -> Dict[str, Any]:
        """返回完整配置字典。"""
        return self._config

    def save(self, path: Union[str, Path]) -> None:
        """将当前配置保存到 YAML 文件。

        Args:
            path: 输出文件路径。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    def resolve_paths(self, base_dir: Union[str, Path]) -> None:
        """将配置中的相对路径解析为绝对路径。

        Args:
            base_dir: 基础目录路径。
        """
        base_dir = Path(base_dir).resolve()
        path_keys = [
            "project.dataset_dir",
            "project.checkpoint_dir",
            "project.log_dir",
            "project.output_dir",
            "model.cache_dir",
            "monitoring.tensorboard.log_dir",
            "monitoring.log_file",
            "monitoring.sample_dir",
        ]
        for key in path_keys:
            value = self.get(key)
            if value and not os.path.isabs(value):
                self.set(key, str(base_dir / value))


# ===== 全局配置单例 =====

_global_loader: Optional[ConfigLoader] = None


def get_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigLoader:
    """获取全局配置加载器实例（单例模式）。

    Args:
        config_path: 用户自定义配置文件路径。
        overrides: 运行时参数覆盖字典。

    Returns:
        ConfigLoader 实例。
    """
    global _global_loader
    if _global_loader is None or config_path is not None or overrides is not None:
        _global_loader = ConfigLoader(config_path, overrides)
    return _global_loader
