"""日志配置模块。

提供两件事：

- `setup_logging()`：初始化，日志级别从 `PRISM_LOG_LEVEL` 读
- `PrismHandler`：把日志渲染成 ``时间 级别 [短路径:行号] 消息``

## 格式约定

    14:23:45.123 INFO [core/diff/loader.py:135] Loaded 2 files, 6 hunks (+4/-16)

- 时间到**毫秒**；级别与路径都**不做对齐** —— 对齐会让字段之间出现大段空白，
  而且路径长度一变整列就歪，不如单空格紧凑
- **短路径**相对包根（`src/prism/`）计算，比绝对路径短，又保留目录结构。
  包外调用方（测试、脚本）只留文件名
- **每条日志之间空一行** —— 多行日志（LLM 上下文那种）不空行会糊成一片，
  看不出这条记录到哪结束

## 为什么自己写 Handler

不用 rich 的 `RichHandler`：它的版面是硬编码的（时间 + 级别 | 正文 |
右对齐路径），改不了。但**颜色闸门继续用 rich 的 `Console`** ——
`NO_COLOR` / `FORCE_COLOR` / isatty / `TERM=dumb` 是现成的，实测重定向到文件时
输出零 ANSI 转义，`NO_COLOR=1` 时颜色被剥掉、只剩粗体。

## 两个必须显式设的开关（都是实测踩出来的）

- ``Console(stderr=True)``：日志走 stderr，stdout 留给审查结果
- ``Console(soft_wrap=True)``：**关掉折行**。rich 默认在终端宽度处插入真换行，
  而 LLM 上下文的正文是「原生的具体内容」，被折了复制回模型就不是原文了
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

# 默认日志级别，PRISM_LOG_LEVEL 未设置或取值非法时使用
DEFAULT_LEVEL = "INFO"

# 本项目自己的 logger 走这个命名空间，第三方库不受 PRISM_LOG_LEVEL 影响
PRISM_LOGGER = "prism"

# 级别配色，沿用 uvicorn 那套
LEVEL_STYLES = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bright_red",
}

PATH_STYLE = "dim cyan"
TIME_STYLE = "dim"

# 包根：src/prism/observability/logging.py 往上数两级
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def short_path(pathname: str) -> str:
    """把源码绝对路径压成相对包根的短路径。

    Args:
        pathname: ``record.pathname``，通常是绝对路径

    Returns:
        ``core/diff/loader.py`` 这样的短路径；不在包内时退化为文件名
    """
    path = Path(pathname)
    try:
        return str(path.resolve().relative_to(_PACKAGE_ROOT))
    except ValueError:
        return path.name


def _timestamp(created: float) -> str:
    """把 LogRecord 的创建时间格式化成毫秒精度。

    `logging` 的 `datefmt` 走 `time.strftime`，不支持 `%f`，所以自己拼。

    Args:
        created: ``record.created``（Unix 时间戳）

    Returns:
        ``14:23:45.123``
    """
    moment = datetime.fromtimestamp(created)
    return f"{moment:%H:%M:%S}.{moment.microsecond // 1000:03d}"


class PrismHandler(logging.Handler):
    """按本项目的格式渲染日志的 handler。

    版面：``时间 级别 [短路径:行号] 消息``，每条记录之后补一个空行。
    """

    def __init__(self, console: Console | None = None) -> None:
        """初始化。

        Args:
            console: rich 控制台；不传时用「stderr + 不折行」的默认配置
        """
        super().__init__()
        self._console = console or Console(stderr=True, soft_wrap=True)

    def emit(self, record: logging.LogRecord) -> None:
        """渲染并输出一条日志。

        Args:
            record: 日志记录
        """
        try:
            text = Text()
            text.append(f"{_timestamp(record.created)} ", style=TIME_STYLE)
            text.append(f"{record.levelname} ", style=LEVEL_STYLES.get(record.levelname))
            text.append(f"[{short_path(record.pathname)}:{record.lineno}]", style=PATH_STYLE)
            text.append(f" {record.getMessage()}")
            self._console.print(text)
            # 每条日志之间空一行，多行记录才不会糊在一起
            self._console.print()
        except Exception:
            self.handleError(record)


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
    logging.basicConfig(level=logging.WARNING, handlers=[PrismHandler()])
    logging.getLogger(PRISM_LOGGER).setLevel(resolve_level(os.getenv("PRISM_LOG_LEVEL")))
