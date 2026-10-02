"""pytest 共享夹具。"""

import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from prism.observability.logging import PRISM_LOGGER, PrismHandler, setup_logging


@pytest.fixture(autouse=True)
def prism_logging_to_console() -> None:
    """设了 `PRISM_LOG_LEVEL` 时，把 prism 的日志打到控制台。

    为什么不能只靠 `setup_logging()`：pytest 的 logging 插件会把**根 logger**
    上的 handler 换成自己的捕获 handler，日志只进缓冲区、只在用例失败时以
    「Captured log call」打出来（`-s` 救不了 —— 它关的是 stdout/stderr 捕获，
    与 logging 插件是两回事）。所以这里把 handler 直接挂到 `prism` logger 上，
    它不在 pytest 的替换范围内。

    为什么默认不开：一次 e2e 的 LLM 轨迹近千行，普通跑测试会被淹掉。
    想看得显式设环境变量：

        PRISM_LOG_LEVEL=DEBUG uv run pytest tests/e2e/test_min_walkthrough.py -s

    PyCharm 就填在 Run/Debug Configurations 的 Environment variables 里。
    """
    if not os.getenv("PRISM_LOG_LEVEL"):
        return

    setup_logging()
    logger = logging.getLogger(PRISM_LOGGER)
    if not any(isinstance(handler, PrismHandler) for handler in logger.handlers):
        logger.addHandler(PrismHandler())


@pytest.fixture
def git() -> Callable[..., None]:
    """返回一个在指定仓库内执行 git 命令的辅助函数。

    Returns:
        ``run(repo, *args)``，命令失败时抛 subprocess.CalledProcessError
    """

    def _run(repo: Path, *args: str) -> None:
        """在 repo 内执行 git 命令。

        Args:
            repo: 仓库路径
            args: git 子命令及参数
        """
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    return _run
