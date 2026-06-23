"""设备检测工具：自动检测并管理计算设备（CPU/GPU）。

支持 NVIDIA CUDA 和 AMD ROCm 显卡的自动检测，
并提供设备信息报告和最佳设备选择功能。
"""

from typing import Dict, Optional

import torch
from loguru import logger


def get_device(preference: str = "auto") -> torch.device:
    """获取最佳可用计算设备。

    根据用户偏好和系统可用资源自动选择计算设备。
    支持 NVIDIA CUDA 和 AMD ROCm。

    Args:
        preference: 设备偏好。
            - "auto": 自动选择（优先 GPU）
            - "cuda": 强制使用 CUDA
            - "cpu": 强制使用 CPU

    Returns:
        torch.device: 选定的 PyTorch 设备。

    Raises:
        RuntimeError: 指定的设备不可用。
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"自动检测到 GPU: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("未检测到 GPU，使用 CPU")
    elif preference in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"请求使用 {preference.upper()}，但未检测到可用的 GPU。"
                "请确认已正确安装 GPU 驱动和 PyTorch CUDA/ROCm 版本。"
            )
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
        logger.info("使用 CPU 进行计算")
    else:
        raise ValueError(f"不支持的设备偏好: {preference}")

    return device


def get_device_info() -> Dict[str, str]:
    """获取当前设备的详细信息。

    Returns:
        包含设备信息的字典，包括：
        - device_type: 设备类型
        - device_name: 设备名称
        - gpu_count: GPU 数量
        - gpu_memory: GPU 显存信息
        - torch_version: PyTorch 版本
        - cuda_version: CUDA 版本
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "device_type": "cuda",
                "cuda_version": torch.version.cuda or "N/A",
                "gpu_count": str(torch.cuda.device_count()),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": (
                    f"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
                ),
                "gpu_memory_allocated": (
                    f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
                ),
                "gpu_memory_cached": (
                    f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
                ),
            }
        )
    else:
        info.update(
            {
                "device_type": "cpu",
                "cpu_threads": str(torch.get_num_threads()),
            }
        )

    return info


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """获取当前 GPU 显存使用情况。

    Returns:
        包含显存使用信息的字典（MB），若无 GPU 则返回 None。
    """
    if not torch.cuda.is_available():
        return None

    return {
        "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
        "free_mb": (
            torch.cuda.get_device_properties(0).total_mem
            - torch.cuda.memory_allocated(0)
        )
        / 1024**2,
    }


def clear_gpu_cache() -> None:
    """清理 GPU 显存缓存。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU 显存缓存已清理")
