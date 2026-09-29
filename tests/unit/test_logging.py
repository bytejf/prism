"""日志模块的单元测试。"""

import logging

import pytest

from prism.observability.logging import (
    LOG_FORMAT,
    PRISM_LOGGER,
    resolve_level,
    setup_logging,
)


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


def test_setup_logging_applies_env_level_to_prism_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测试 setup_logging 把环境变量级别应用在 prism 命名空间上。"""
    monkeypatch.setenv("PRISM_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])  # 清空已有 handler，让 basicConfig 生效
    setup_logging()

    assert logging.getLogger(PRISM_LOGGER).level == logging.DEBUG
    assert root.handlers[0].formatter is not None
    assert root.handlers[0].formatter._fmt == LOG_FORMAT


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
