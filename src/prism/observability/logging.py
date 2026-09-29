"""日志配置模块。

提供统一的日志初始化，从环境变量 PRISM_LOG_LEVEL 读取日志级别。
V1 使用 Python 标准库 logging，阶段 4 接入 Langfuse 后本模块继续保留。
"""

import logging
import os

# 日志格式：时间 | 级别 | 模块 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# 默认日志级别，PRISM_LOG_LEVEL 未设置或取值非法时使用
DEFAULT_LEVEL = "INFO"


def resolve_level(name: str | None) -> int:
    """把日志级别名解析为 logging 级别常量。

    Args:
        name: 级别名，如 "DEBUG"；为 None 或非法值时使用 DEFAULT_LEVEL

    Returns:
        logging 级别常量，如 logging.DEBUG
    """
    level = getattr(logging, (name or DEFAULT_LEVEL).upper(), None)
    return level if isinstance(level, int) else logging.INFO


def setup_logging() -> None:
    """初始化日志配置。

    从环境变量 PRISM_LOG_LEVEL 读取日志级别，默认 INFO。
    重复调用无副作用（logging.basicConfig 在已有 handler 时为空操作）。
    """
    logging.basicConfig(level=resolve_level(os.getenv("PRISM_LOG_LEVEL")), format=LOG_FORMAT)
