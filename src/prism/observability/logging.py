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

# 本项目自己的 logger 走这个命名空间，第三方库不受 PRISM_LOG_LEVEL 影响
PRISM_LOGGER = "prism"


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
    级别只作用于 ``prism`` 命名空间：根 logger 固定在 WARNING，
    否则 PRISM_LOG_LEVEL=DEBUG 会把 httpx / openai 等第三方库的
    内部 DEBUG 日志一并打开，淹没我们自己要定位问题的关键节点日志。

    重复调用无副作用（basicConfig 在已有 handler 时为空操作）。
    """
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logging.getLogger(PRISM_LOGGER).setLevel(resolve_level(os.getenv("PRISM_LOG_LEVEL")))
