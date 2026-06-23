"""测试：模型管理器和生成器"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from models.model_manager import ModelManager
from models.base_model import BaseMusicModel


class TestModelManager:
    """模型管理器测试。"""

    def test_init(self, tmp_path):
        """测试初始化。"""
        manager = ModelManager(cache_dir=tmp_path / "models")
        assert manager.cache_dir.exists()

    def test_list_available_models(self):
        """测试列出支持的模型。"""
        manager = ModelManager()
        models = manager.list_available_models()

        assert len(models) >= 5
        names = [m["name"] for m in models]
        assert "musicgen-small" in names
        assert "musecoco" in names

    def test_get_model_path_not_cached(self, tmp_path):
        """测试未缓存的模型返回 None。"""
        manager = ModelManager(cache_dir=tmp_path / "models")
        path = manager.get_model_path("musicgen-small")
        assert path is None

    def test_check_invalid_model(self):
        """测试不支持的模型名报错。"""
        manager = ModelManager()
        with pytest.raises(ValueError):
            manager.check_model_version("invalid-model-name")

    def test_download_invalid_model(self):
        """测试下载不支持的模型报错。"""
        manager = ModelManager()
        with pytest.raises(ValueError):
            manager.download_model("invalid-model-name")

    def test_register_custom_model(self):
        """测试注册自定义模型。"""
        manager = ModelManager()
        manager.register_custom_model(
            name="my-custom-model",
            repo_id="user/custom-music-model",
            architecture="musicgen",
            description="自定义测试模型",
        )

        models = manager.list_available_models()
        names = [m["name"] for m in models]
        assert "my-custom-model" in names

    def test_register_invalid_architecture(self):
        """测试注册不支持的架构报错。"""
        manager = ModelManager()
        with pytest.raises(ValueError):
            manager.register_custom_model(
                name="bad-model",
                repo_id="user/model",
                architecture="unsupported_arch",
            )

    def test_clean_cache_empty(self, tmp_path):
        """测试空缓存的清理操作不报错。"""
        manager = ModelManager(cache_dir=tmp_path / "models")
        manager.clean_cache()  # 不应抛出异常

    def test_metadata_persistence(self, tmp_path):
        """测试元数据持久化。"""
        manager = ModelManager(cache_dir=tmp_path / "models")

        # 手动设置元数据
        manager._metadata["test"] = {"sha": "abc123"}
        manager._save_metadata()

        # 重新加载
        manager2 = ModelManager(cache_dir=tmp_path / "models")
        assert "test" in manager2._metadata
        assert manager2._metadata["test"]["sha"] == "abc123"


class TestBaseModelInterface:
    """基类接口测试（确保抽象方法定义正确）。"""

    def test_cannot_instantiate_base(self):
        """测试抽象基类不可直接实例化。"""
        import torch
        with pytest.raises(TypeError):
            BaseMusicModel("test", "test", torch.device("cpu"))


class TestMusicGeneratorInit:
    """音乐生成器初始化测试。"""

    def test_init_default(self):
        """测试默认初始化。"""
        from generation.generator import MusicGenerator

        gen = MusicGenerator(
            architecture="musicgen",
            model_name="musicgen-small",
        )
        assert gen.architecture == "musicgen"
        assert gen.model_name == "musicgen-small"
        assert gen.model is None
        assert gen.style_analyzer is not None

    def test_init_style_fusion(self):
        """测试风格融合模块初始化。"""
        from generation.generator import MusicGenerator

        gen = MusicGenerator(architecture="musicgen")
        gen.init_style_fusion(embedding_dim=256, strategy="adaptive")

        assert gen.style_embedding is not None
        assert gen.style_fusion is not None

    def test_style_embedding_not_initialized_returns_none(self):
        """测试未初始化风格融合时返回 None。"""
        from generation.generator import MusicGenerator

        gen = MusicGenerator()
        result = gen.compute_style_embedding()
        assert result is None

    def test_get_model_info_not_loaded(self):
        """测试未加载模型时的信息返回。"""
        from generation.generator import MusicGenerator

        gen = MusicGenerator()
        info = gen.get_model_info()
        assert info["status"] == "not_loaded"
