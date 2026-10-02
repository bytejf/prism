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
    """设了 `PRISM_LOG_LEVEL` 时，把 prism 的日志按约定格式打到控制台。

    为什么不能只靠 `setup_logging()`：pytest 的 logging 插件会把**根 logger**
    上的 handler 换成自己的捕获 handler，日志只进缓冲区、只在用例失败时以
    「Captured log call」打出来。所以这里把 handler 直接挂到 `prism` logger 上
    —— 它不在 pytest 的替换范围内。

    **必须配合 `-s`**：pytest 的捕获是 **fd 级**的（`dup2` 重定向 fd 2），
    连 `sys.__stderr__` 都一起吞，绕不过去。没加 `-s` 时本夹具等于白挂，
    收尾会由 `pytest_terminal_summary` 提示。

    为什么默认不开：一次 e2e 的 LLM 轨迹近千行，普通跑测试会被淹掉。
    """
    if not os.getenv("PRISM_LOG_LEVEL"):
        return

    setup_logging()
    logger = logging.getLogger(PRISM_LOGGER)
    if not any(isinstance(handler, PrismHandler) for handler in logger.handlers):
        logger.addHandler(PrismHandler())


def pytest_collection_finish(session: pytest.Session) -> None:
    """收集完成后自诊断日志配置 —— 两种错配都会静默地看不到日志。

    放在这个钩子而不是夹具内部，有两个原因：

    - 夹具里 print 会被 pytest 的捕获吞掉，等于没说
    - 也不能放 `pytest_terminal_summary`：IDE 的 runner 常带 `--no-summary`，
      会把钩子的输出一起压掉（实测 PyCharm 的默认参数就是
      `--no-header --no-summary -q`）。收集阶段不受这两个开关影响

    Args:
        session: pytest 会话
    """
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    env_level = os.getenv("PRISM_LOG_LEVEL")
    cli_level = session.config.getoption("log_cli_level", None)
    capture_on = session.config.getoption("capture", "fd") != "no"

    if cli_level and not env_level:
        reporter.write_line(
            "\n[prism] 用了 --log-cli-level —— 日志会走 pytest 自己的格式"
            "（`级别 logger名:文件:行号`），不是项目约定的格式。\n"
            "        想看得换成环境变量 PRISM_LOG_LEVEL=INFO，并去掉 --log-cli-level。"
        )
    elif env_level and capture_on:
        reporter.write_line(
            f"\n[prism] 设了 PRISM_LOG_LEVEL={env_level}，但没加 -s —— "
            "日志被 pytest 的捕获吞掉了（捕获是 fd 级的，绕不过）。\n"
            "        加上 -s 即可。"
        )


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
