"""日志配置模块。

提供统一的日志初始化，从环境变量 PRISM_LOG_LEVEL 读取日志级别。
V1 使用 Python 标准库 `logging` + `rich` 渲染，阶段 4 接入 Langfuse 后本模块继续保留。

## 为什么用 RichHandler

1. **多行消息不会糊在一起。** 一条 `llm_trace` 记录就二十多行，用标准
   `StreamHandler` 时下一条的抬头紧贴着上一条的末尾，看不出记录边界。
   RichHandler 把续行缩进到消息列对齐，时间戳那一栏留空，一眼能看出
   「这条记录到哪结束」。
2. **颜色闸门是现成的**，且与业界共识一致：`NO_COLOR` > `FORCE_COLOR` >
   isatty 判定 > `TERM=dumb`。实测重定向到文件时输出零 ANSI 转义，
   `NO_COLOR=1` 时只剩粗体、颜色被剥掉。
3. **只给级别上色**，消息正文保持原样 —— 正文染色会让 `grep` 和重定向都坏掉。
4. 右侧的 `file.py:line` 还是一个 **OSC-8 超链接**，显示短名、点开是完整路径。

## 两个必须显式设的开关（都是实测踩出来的）

- `console=Console(stderr=True)`：`RichHandler()` **默认写 stdout**，
  而 stdout 是留给审查结果的。不改的话 `prism review > out.txt` 会把日志一起写进去。
- `markup=False`：默认开着 rich 标记解析，日志里的 `[high · 并发安全]`
  会被当成标记**整段吃掉**（实测真的消失了）。
"""

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

# 时间 / 级别 / 路径都由 RichHandler 渲染，模板只负责消息本身
LOG_FORMAT = "%(message)s"

# 时间精确到秒即可：一次审查几十秒，带上日期纯属噪声
LOG_TIME_FORMAT = "%H:%M:%S"

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


def build_handler() -> RichHandler:
    """构造日志 handler。

    Returns:
        配置好的 RichHandler（写 stderr、关 markup、右侧显示 file:line）
    """
    handler = RichHandler(
        # 日志走 stderr，stdout 留给审查结果
        console=Console(stderr=True),
        # 关掉标记解析，否则 `[high · 并发安全]` 这类方括号会被吃掉
        markup=False,
        # 右侧显示 file.py:line，定位用
        show_path=True,
        # 只影响 logger.exception；未捕获的异常不走 logging，另由 Python 打印
        rich_tracebacks=True,
        log_time_format=LOG_TIME_FORMAT,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def setup_logging() -> None:
    """初始化日志配置。

    从环境变量 PRISM_LOG_LEVEL 读取日志级别，默认 INFO。
    级别只作用于 ``prism`` 命名空间：根 logger 固定在 WARNING，
    否则 PRISM_LOG_LEVEL=DEBUG 会把 httpx / openai 等第三方库的
    内部 DEBUG 日志一并打开，淹没我们自己要定位问题的关键节点日志。

    重复调用无副作用（basicConfig 在已有 handler 时为空操作）。
    """
    logging.basicConfig(level=logging.WARNING, handlers=[build_handler()])
    logging.getLogger(PRISM_LOGGER).setLevel(resolve_level(os.getenv("PRISM_LOG_LEVEL")))
