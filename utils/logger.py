"""日志工具：基于 loguru 的统一日志管理。

提供项目统一的日志记录接口，支持控制台输出、文件输出、
以及 TensorBoard 集成的日志记录。
"""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# 移除 loguru 默认的 handler
logger.remove()

# 已初始化的标志
_initialized = False


def setup_logger(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """初始化全局日志系统。

    Args:
        log_file: 日志文件路径。若为 None 则不写入文件。
        log_level: 日志级别，可选 DEBUG、INFO、WARNING、ERROR。
        console_output: 是否在控制台输出日志。
        format_string: 自定义日志格式字符串。
    """
    global _initialized

    if _initialized:
        return

    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    if console_output:
        logger.add(
            sys.stderr,
            format=format_string,
            level=log_level,
            colorize=True,
        )

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=format_string,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            encoding="utf-8",
        )

    _initialized = True
    logger.info(f"日志系统已初始化，级别: {log_level}")


def get_logger(name: Optional[str] = None):
    """获取日志记录器实例。

    Args:
        name: 记录器名称，通常使用模块名。

    Returns:
        绑定指定名称的 loguru logger。
    """
    if not _initialized:
        setup_logger()

    if name:
        return logger.bind(name=name)
    return logger
