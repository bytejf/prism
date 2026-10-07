"""OpenTelemetry 追踪：把一次审查的**执行过程**写成 OTLP JSON（JSONL）。

设计：`acr-min/06-可观测性/21-可观测性设计.md`

## 为什么是 OTel 而不是 Langfuse SDK

Langfuse SDK 的离线模式有已知 bug（`tracing_enabled=False` 时内部 `_otel_tracer`
是 `None`，`@observe` 仍去访问它），社区 workaround 是塞假密钥让它别报错 —— 那只是
绕过问题。OTel 官方的 `FileSpanExporter` 直接把 spans 写成**标准 OTLP JSON**，而
Langfuse 的服务端原生收 `/api/public/otel`，将来要上传不用改代码。

## 两个开关，都不设就完全不初始化

    PRISM_TRACE_FILE  写到这个文件（bench 驱动按「一个测试用例一个文件」命名）
    PRISM_TRACE_DIR   写到该目录下的自动命名文件（临时跑用）

**不设时零开销**：OTel 的默认 tracer 是 no-op 代理，业务代码里那些
`with tracer.start_as_current_span(...)` 照常写、照常嵌套，只是什么都没记。
所以埋点处不需要 `if tracing_enabled` 分支 —— 那正是选 OTel 的一个附带好处。

## 与现有 logger 的关系

**并行、互不干扰**：日志是给人读的摘要（`PRISM_LOG_LEVEL`），trace 是给程序读的
结构化过程（每个 LLM 调用、每个工具）。OTel SDK 内部也用标准 `logging`
（`opentelemetry` 命名空间），所以它那些"导出失败"之类的消息不会混进我们的格式。
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.json.file import FileSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode

logger = logging.getLogger(__name__)

# 写指定文件的完整路径（bench 驱动用；按设计 §4.1 命名 `<repo>-<pr>.jsonl`）
TRACE_FILE_ENV = "PRISM_TRACE_FILE"

# 写该目录下的自动命名文件（临时跑用）
TRACE_DIR_ENV = "PRISM_TRACE_DIR"

SERVICE_NAME = "prism"

# 自动命名用的前缀
AUTO_NAME_PREFIX = "prism-"

# 每个模块用 `trace.get_tracer(__name__)` 取自己的 tracer，没配 tracing 时是 no-op
_configured = False
_manager: Path | None = None


def setup_tracing() -> Path | None:
    """按环境变量初始化追踪，返回写入的文件路径（没配则 None）。

    只能生效一次：OTel 的全局 TracerProvider 一个进程只认第一次设置，重复调用
    直接返回上一次的结果（而不是静默丢掉第二次配置 —— 那会让人以为改了环境变量
    却没生效）。

    Returns:
        写入的文件路径；两个环境变量都没设时返回 None
    """
    global _configured, _manager
    if _configured:
        logger.debug("追踪已初始化过，跳过")
        return _manager
    _configured = True

    target = _resolve_target()
    if target is None:
        logger.debug("未设置 %s / %s，追踪关闭", TRACE_FILE_ENV, TRACE_DIR_ENV)
        return None

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: SERVICE_NAME}))
    # 用 Batch 而不是 Simple：一次审查几十上百个 span，逐条 flush 会把主流程拖慢；
    # 代价是进程被 kill 时可能丢最后几个 span（TracerProvider 自己在退出时 shutdown）
    provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(target)))
    trace.set_tracer_provider(provider)
    _manager = target
    logger.info("追踪已开启：%s", target)
    return target


def _resolve_target() -> Path | None:
    """决定 trace 写到哪里。

    `PRISM_TRACE_FILE` 优先（给精确指定文件名的一方用），否则用 `PRISM_TRACE_DIR`
    下的自动命名文件。目录不存在时**直接创建** —— 让调用方先 mkdir 只是多一步。

    Returns:
        目标文件路径；都不设时返回 None
    """
    explicit = os.getenv(TRACE_FILE_ENV, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    directory = os.getenv(TRACE_DIR_ENV, "").strip()
    if not directory:
        return None
    folder = Path(directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    # 自动命名带上时间戳：同一个目录里跑多次不会互相覆盖（a 模式追加，
    # 两条 trace 会混在同一个文件里）
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return folder / f"{AUTO_NAME_PREFIX}{stamp}.jsonl"


def record_error(span: Span, error: BaseException) -> None:
    """把一个异常记进 span（状态置 ERROR + 异常事件）。

    设计上说「trace 是给程序读的过程记录」—— 失败路径同样是要读的过程，
    所以异常不能只出现在日志里。

    Args:
        span: 当前 span
        error: 抛出的异常
    """
    span.record_exception(error)
    span.set_status(StatusCode.ERROR, f"{type(error).__name__}: {error}")


def set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """一次设多个属性，跳过值为 None 的。

    `None` 在 OTel 里不是一个合法属性值（会被当成没设），显式跳过免得依赖
    实现的容错。其余值（str / int / float / bool）原样交给 SDK。

    Args:
        span: 当前 span
        attributes: 属性字典
    """
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def parse_trace_attrs(pairs: list[str]) -> dict[str, str]:
    """解析 `--trace-attr key=value` 那种参数。

    给调用方一个通用出口：`pr_url`、`instance_id` 这些**只有驱动知道**的信息，
    由驱动挂在根 span 上，而不是让 prism 去猜。

    Args:
        pairs: 形如 ``["pr_url=https://…", "instance_id=abc"]``

    Returns:
        属性字典

    Raises:
        ValueError: 某一项不是 ``key=value`` 形式，或 key 为空
    """
    attributes: dict[str, str] = {}
    for item in pairs:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            msg = f"--trace-attr 需要 key=value 形式，收到：{item}"
            raise ValueError(msg)
        attributes[key.strip()] = value.strip()
    return attributes
