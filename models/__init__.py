"""模型模块：音乐生成模型的下载、加载和管理"""

from .model_manager import ModelManager
from .base_model import BaseMusicModel
from .musicgen_model import MusicGenModel
from .musecoco_model import MuseCocoModel

__all__ = [
    "ModelManager",
    "BaseMusicModel",
    "MusicGenModel",
    "MuseCocoModel",
]
