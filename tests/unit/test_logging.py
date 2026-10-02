"""日志模块的单元测试。"""

import logging
import os
import re
from io import StringIO

import pytest
from rich.console import Console

from prism.observability import logging as prism_logging
from prism.observability.logging import (
    PRISM_LOGGER,
    PrismHandler,
    resolve_level,
    setup_logging,
    short_path,
)


def render(level: int, message: str) -> str:
    """用 PrismHandler 渲染一条日志并取回文本。

    Args:
        level: 日志级别
        message: 日志正文

    Returns:
        渲染结果
    """
    buffer = StringIO()
    handler = PrismHandler(Console(file=buffer, width=200, soft_wrap=True))
    logger = logging.getLogger("prism.test.render")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.log(level, message)
    return buffer.getvalue()


# ---------- 级别解析 ----------


def test_resolve_level_valid() -> None:
    """测试合法级别名解析。"""
    assert resolve_level("DEBUG") == logging.DEBUG
    assert resolve_level("debug") == logging.DEBUG
    assert resolve_level("WARNING") == logging.WARNING


def test_resolve_level_invalid_falls_back_to_info() -> None:
    """测试非法级别名降级为 INFO，不抛异常。"""
    assert resolve_level("VERBOSE") == logging.INFO
    assert resolve_level("") == logging.INFO
    assert resolve_level(None) == logging.INFO


# ---------- 短路径 ----------


def test_short_path_strips_package_root() -> None:
    """测试包内路径被压成相对包根的短路径。"""
    inside = os.path.join(
        os.path.dirname(prism_logging.__file__), "..", "core", "diff", "loader.py"
    )
    assert short_path(inside) == "core/diff/loader.py"


def test_short_path_falls_back_to_basename_outside_package() -> None:
    """测试包外路径退化为文件名（测试、脚本都落在包外）。"""
    assert short_path("/tmp/somewhere/else/probe.py") == "probe.py"


# ---------- 渲染 ----------


def test_handler_renders_time_level_path_message() -> None:
    """测试版面：``时间 级别 [短路径:行号] 消息``。"""
    output = render(logging.INFO, "测试消息")
    line = output.splitlines()[0]

    # 时间到毫秒，紧接着是级别
    assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} INFO ", line), line
    # 路径是包外文件 → 只剩文件名
    assert "[test_logging.py:" in line
    assert line.endswith(" 测试消息")


def test_handler_separates_records_with_blank_line() -> None:
    """测试每条日志之间空一行 —— 多行记录才不会糊在一起。"""
    output = render(logging.INFO, "第一行\n第二行")

    assert output.endswith("\n\n")
    assert "第二行" in output


def test_handler_does_not_wrap_long_content() -> None:
    """测试长正文不被折行 —— 折了就是篡改原文。

    `soft_wrap` 关掉 rich 的自动折行，正文可以安全地复制回模型。
    """
    long_text = "字" * 300
    output = render(logging.INFO, long_text)

    assert long_text in output


# ---------- 初始化 ----------


def test_setup_logging_applies_env_level_to_prism_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 setup_logging 把环境变量级别应用在 prism 命名空间上。"""
    monkeypatch.setenv("PRISM_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])  # 清空已有 handler，让 basicConfig 生效
    setup_logging()

    assert logging.getLogger(PRISM_LOGGER).level == logging.DEBUG
    assert isinstance(root.handlers[0], PrismHandler)


def test_setup_logging_keeps_third_party_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试第三方库日志不被 PRISM_LOG_LEVEL 打开。"""
    monkeypatch.setenv("PRISM_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    setup_logging()

    noqa_logger = logging.getLogger("httpx2")
    assert root.level == logging.WARNING
    assert not noqa_logger.isEnabledFor(logging.DEBUG)
    assert logging.getLogger("prism.core.llm.client").isEnabledFor(logging.DEBUG)
