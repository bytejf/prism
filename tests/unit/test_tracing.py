"""可观测性（OTel 追踪）的单元测试。

**不依赖网络、不依赖真实 LLM**：只验「配置怎么解析、span 写出来长什么样」。
写完就真跑一次的做法在 `tests/manual/` 里（那需要真调模型）。
"""

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prism.observability import semconv as sc
from prism.observability.tracing import (
    AUTO_NAME_PREFIX,
    TRACE_DIR_ENV,
    TRACE_FILE_ENV,
    parse_trace_attrs,
    record_error,
    set_attributes,
    setup_tracing,
)


@pytest.fixture(scope="session")
def span_exporter() -> InMemorySpanExporter:
    """整个测试进程装**一次**内存 exporter。

    `set_tracer_provider` 一个进程只认第一次（第二次会打
    「Overriding of current TracerProvider is not allowed」并静默不生效）——
    所以它必须是 session 级的，不能每个用例装一次。

    Returns:
        内存 exporter；用例结束时读 `get_finished_spans()`
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def collected(span_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """清空 exporter，让本次用例只看得到自己的 span。

    **返回 exporter 而不是 `get_finished_spans()`**：后者返回的是调用那一刻的
    快照（tuple），而在测试体里拿到它时 span 还没结束 —— 必须先清、跑完再取。

    Args:
        span_exporter: 会话级的内存 exporter

    Returns:
        同一个 exporter（用例末尾自己 `get_finished_spans()`）
    """
    span_exporter.clear()
    return span_exporter


@pytest.fixture(autouse=True)
def reset_tracing_flag(monkeypatch: pytest.MonkeyPatch) -> Any:
    """每个用例都把「初始化过」的标记复位，并拦下全局 provider 的设置。

    `setup_tracing` 里那个一次性闸是生产逻辑（全局 provider 只能设一次），测试里
    得复位才验得了第二次调用。而它真去 `set_tracer_provider` 的话，会和上面那个
    会话级 exporter 打架（第二次设置不生效并告警）—— 所以这里换成记录器：
    既不动全局状态，还能顺手断言「确实设了一次」。

    Yields:
        记录器列表，每被调用一次塞进一个 provider
    """
    installed: list[Any] = []
    monkeypatch.setattr("prism.observability.tracing._configured", False)
    monkeypatch.setattr("prism.observability.tracing._manager", None)
    monkeypatch.setattr("prism.observability.tracing.trace.set_tracer_provider", installed.append)
    monkeypatch.delenv(TRACE_FILE_ENV, raising=False)
    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)
    yield installed


# ---------- 配置解析 ----------


def test_no_env_means_no_tracing(reset_tracing_flag: list[Any]) -> None:
    """两个环境变量都不设 → 不初始化（返回 None，也不设全局 provider）。"""
    assert setup_tracing() is None
    assert reset_tracing_flag == []


def test_trace_file_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`PRISM_TRACE_FILE` 指到还不存在的目录时，目录会被建出来。

    让调用方先 mkdir 只是多一步 —— bench 驱动要按「一个测试用例一个文件」命名，
    不该再管目录。
    """
    target = tmp_path / "traces" / "aacr-bench-go" / "wavetermdev-waveterm-1755.jsonl"
    monkeypatch.setenv(TRACE_FILE_ENV, str(target))

    assert setup_tracing() == target
    assert target.parent.is_dir()


def test_trace_dir_auto_names_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`PRISM_TRACE_DIR` 用自动命名（带时间戳，同目录跑多次不互相覆盖）。"""
    monkeypatch.setenv(TRACE_DIR_ENV, str(tmp_path))

    target = setup_tracing()

    assert target is not None
    assert target.parent == tmp_path
    assert target.name.startswith(AUTO_NAME_PREFIX)
    assert target.suffix == ".jsonl"


def test_file_wins_over_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """两个都设时以 FILE 为准（驱动要精确控制文件名时用得上）。"""
    target = tmp_path / "explicit.jsonl"
    monkeypatch.setenv(TRACE_FILE_ENV, str(target))
    monkeypatch.setenv(TRACE_DIR_ENV, str(tmp_path / "auto"))

    assert setup_tracing() == target


def test_setup_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """重复调用只生效一次，且第二次返回同一个路径（不静默丢掉第二次配置）。"""
    first = tmp_path / "a.jsonl"
    monkeypatch.setenv(TRACE_FILE_ENV, str(first))
    assert setup_tracing() == first

    monkeypatch.setenv(TRACE_FILE_ENV, str(tmp_path / "b.jsonl"))

    assert setup_tracing() == first


def test_parse_trace_attrs() -> None:
    """`--trace-attr` 解析成字典。"""
    assert parse_trace_attrs(["pr_url=https://github.com/a/b/pull/1", "instance_id=abc"]) == {
        "pr_url": "https://github.com/a/b/pull/1",
        "instance_id": "abc",
    }


@pytest.mark.parametrize("bad", ["没有等号", "=只有值"])
def test_parse_trace_attrs_rejects_malformed(bad: str) -> None:
    """格式不对直接报错，不静默忽略 —— 驱动把属性写错了得当场知道。"""
    with pytest.raises(ValueError, match="key=value"):
        parse_trace_attrs([bad])


# ---------- 属性与 span ----------


def test_set_attributes_skips_none(collected: InMemorySpanExporter) -> None:
    """值为 None 的属性跳过（OTel 不认 None，靠实现容错不如自己跳）。"""
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("probe") as span:
        set_attributes(span, {"a": 1, "b": None, "c": "x"})

    attributes = collected.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["a"] == 1
    assert attributes["c"] == "x"
    assert "b" not in attributes


def test_record_error_marks_the_span(collected: InMemorySpanExporter) -> None:
    """异常要记进 span 的状态与事件（失败路径也是要读的过程）。"""
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("probe") as span:
        record_error(span, RuntimeError("炸了"))

    finished = collected.get_finished_spans()[0]
    assert finished.status.status_code.name == "ERROR"
    description = finished.status.description or ""
    assert "炸了" in description
    assert any(event.name == "exception" for event in finished.events)


def test_genai_attribute_names_match_semconv() -> None:
    """属性名的**漂移守卫**。

    `semconv.py` 里写字面量是刻意的（那批属性还在 `_incubating` 命名空间下，
    路径在版本间挪过），代价是可能漂 —— 所以这里在**能 import 到 semconv 包时**
    逐条核对：标准改了名，这条用例会红，而不是我们默默记了一份没人认得的属性。
    """
    semconv = pytest.importorskip("opentelemetry.semconv._incubating.attributes.gen_ai_attributes")

    assert sc.GEN_AI_REQUEST_MODEL == semconv.GEN_AI_REQUEST_MODEL
    assert sc.GEN_AI_PROVIDER_NAME == semconv.GEN_AI_PROVIDER_NAME
    assert sc.GEN_AI_USAGE_INPUT_TOKENS == semconv.GEN_AI_USAGE_INPUT_TOKENS
    assert sc.GEN_AI_USAGE_OUTPUT_TOKENS == semconv.GEN_AI_USAGE_OUTPUT_TOKENS
    assert sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS == semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
    assert (
        sc.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        == semconv.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
    )
    assert sc.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS == semconv.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
    assert sc.GEN_AI_RESPONSE_FINISH_REASONS == semconv.GEN_AI_RESPONSE_FINISH_REASONS


def test_span_hierarchy_and_attributes_are_recorded(collected: InMemorySpanExporter) -> None:
    """父子的 `parentSpanId` 与属性都要落到 span 上（这是整条链能不能读的前提）。"""
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("prism_review") as root:
        root.set_attribute(sc.PRISM_HEAD_REV, "HEAD")
        with tracer.start_as_current_span("group_review") as group:
            group.set_attribute(sc.PRISM_GROUP_ID, "group-001")

    by_name = {span.name: span for span in collected.get_finished_spans()}
    assert by_name["group_review"].parent is not None
    assert by_name["group_review"].parent.span_id == by_name["prism_review"].context.span_id
    assert by_name["prism_review"].parent is None
    assert by_name["prism_review"].attributes is not None


def read_span_records(path: Path) -> list[dict[str, Any]]:
    """把 FileSpanExporter 写出的 JSONL 摊平成 span 记录列表。

    **每一行不是一条 span，而是一个 OTLP 导出批次**：

        {"resourceSpans": [{"scopeSpans": [{"spans": [{...}]}]}]}

    设计文档 §4.1 写的是「每行格式：标准 OTLP JSON，包含 traceId、spanId、
    parentSpanId、name、startTime、endTime、attributes」—— **两处都不对**：
    行是批次不是单条 span，时间字段叫 `startTimeUnixNano` / `endTimeUnixNano`
    （纳秒字符串），而且**没有父 span 的键**（父子关系靠子 span 自己的
    `parentSpanId` 表达）。都记在实现反馈里。

    Args:
        path: trace 文件

    Returns:
        摊平后的 span 记录
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        for resource in payload.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                records.extend(scope.get("spans", []))
    return records


def test_exported_json_is_standard_otlp(tmp_path: Path) -> None:
    """写出来的 JSONL 是标准 OTLP JSON，span 记录里有那几个关键字段。

    这条直接对着 `FileSpanExporter` 真写文件 —— 内存 exporter 验不出格式。
    **不碰全局 provider**（那已经被会话级 exporter 占了，第二次设置不生效）：
    用自己这个 provider 的 tracer，走的是同一条导出链路。
    """
    from opentelemetry.exporter.otlp.json.file import FileSpanExporter

    target = tmp_path / "trace.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(FileSpanExporter(target)))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("prism_review") as span:
        span.set_attribute(sc.PRISM_HEAD_REV, "HEAD")
        with tracer.start_as_current_span("group_review"):
            pass
    provider.shutdown()

    records = read_span_records(target)
    by_name = {record["name"]: record for record in records}

    assert set(by_name) == {"prism_review", "group_review"}, "父子两个 span 都要写出来"
    root = by_name["prism_review"]
    for key in ("traceId", "spanId", "name", "startTimeUnixNano", "endTimeUnixNano"):
        assert key in root, f"缺字段 {key}"
    # 属性是 [{"key":…,"value":{"stringValue":…}}] 这种 OTLP 形态，不是扁平 dict
    keys = {item["key"] for item in root["attributes"]}
    assert sc.PRISM_HEAD_REV in keys
    # 子 span 记着父的 spanId（这就是「trace 树」能不能被读出来的前提）
    child = by_name["group_review"]
    assert child["parentSpanId"] == root["spanId"]
    assert child["traceId"] == root["traceId"]
