"""日志模块的单元测试。"""

import logging

from prism.observability.logging import LOG_FORMAT, resolve_level, setup_logging


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


def test_setup_logging_applies_env_level(monkeypatch) -> None:
    """测试 setup_logging 从环境变量读取级别并生效。"""
    monkeypatch.setenv("PRISM_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])  # 清空已有 handler，让 basicConfig 生效
    setup_logging()
    assert root.level == logging.DEBUG
    assert root.handlers[0].formatter is not None
    assert root.handlers[0].formatter._fmt == LOG_FORMAT
