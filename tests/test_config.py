"""测试：配置模块"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

from config.config_loader import ConfigLoader, get_config


class TestConfigLoader:
    """配置加载器测试。"""

    def test_load_default_config(self):
        """测试默认配置加载。"""
        loader = ConfigLoader()
        config = loader.config

        assert config is not None
        assert "project" in config
        assert "model" in config
        assert "training" in config

    def test_get_nested_value(self):
        """测试嵌套配置项获取。"""
        loader = ConfigLoader()

        assert loader.get("training.batch_size") == 4
        assert loader.get("training.learning_rate") == 5e-5
        assert loader.get("model.architecture") == "musicgen"

    def test_get_default_value(self):
        """测试不存在的键返回默认值。"""
        loader = ConfigLoader()
        assert loader.get("nonexistent.key", "default") == "default"

    def test_set_nested_value(self):
        """测试动态设置嵌套配置项。"""
        loader = ConfigLoader()
        loader.set("training.batch_size", 16)
        assert loader.get("training.batch_size") == 16

    def test_overrides(self):
        """测试运行时参数覆盖。"""
        loader = ConfigLoader(overrides={
            "training.batch_size": 32,
            "training.learning_rate": 1e-3,
        })
        assert loader.get("training.batch_size") == 32
        assert loader.get("training.learning_rate") == 1e-3

    def test_save_and_reload(self, tmp_path):
        """测试配置保存和重新加载。"""
        loader = ConfigLoader(overrides={"training.batch_size": 64})
        save_path = tmp_path / "test_config.yaml"
        loader.save(save_path)

        assert save_path.exists()

        # 重新加载
        loader2 = ConfigLoader(config_path=save_path)
        assert loader2.get("training.batch_size") == 64

    def test_deep_merge(self):
        """测试深度合并。"""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10, "e": 5}}

        result = ConfigLoader._deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 10
        assert result["b"]["d"] == 3
        assert result["b"]["e"] == 5


class TestGetConfig:
    """全局配置单例测试。"""

    def test_singleton_pattern(self):
        """测试单例模式。"""
        # 重置全局状态
        import config.config_loader as cl
        cl._global_loader = None

        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_with_overrides(self):
        """测试带覆盖参数的全局配置。"""
        import config.config_loader as cl
        cl._global_loader = None

        config = get_config(overrides={"training.batch_size": 128})
        assert config.get("training.batch_size") == 128

        # 重置
        cl._global_loader = None
