"""pytest 共享夹具。"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


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
