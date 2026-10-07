"""端到端 smoke test。

用程序化方式构造临时 git 仓库，跑一次完整审查。
不依赖嵌套 git 仓库，避免主仓库无法跟踪的问题。

V1 只断言「命令跑通 + 有输出」，不断言命中特定 bug
（没有定位、去重、反思，评论质量本身不稳定；质量评估在阶段 3）。
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from prism.models.llm import API_KEY_ENV

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "go-sample"

# 真实 LLM 调用可能跑几十秒
REVIEW_TIMEOUT_SECONDS = 600


@pytest.fixture
def sample_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造一个临时的 git 仓库。

    1. 复制 base/ 到临时目录，作为 main 分支
    2. 复制 buggy/ 覆盖，作为 feature 分支

    Args:
        tmp_path: pytest 临时目录
        git: git 命令辅助函数

    Returns:
        临时仓库路径
    """
    repo_dir = tmp_path / "go-sample"
    repo_dir.mkdir()
    git(repo_dir, "init", "-q", "-b", "main")
    git(repo_dir, "config", "user.name", "test")
    git(repo_dir, "config", "user.email", "test@example.com")

    shutil.copytree(FIXTURE_DIR / "base", repo_dir, dirs_exist_ok=True)
    git(repo_dir, "add", "-A")
    git(repo_dir, "commit", "-q", "-m", "init: base code")

    git(repo_dir, "checkout", "-q", "-b", "feature")
    shutil.copytree(FIXTURE_DIR / "buggy", repo_dir, dirs_exist_ok=True)
    git(repo_dir, "add", "-A")
    git(repo_dir, "commit", "-q", "-m", "feat: introduce bugs")
    return repo_dir


def test_smoke_review(sample_repo: Path) -> None:
    """跑一次完整审查，验证不崩溃且有输出。"""
    result = subprocess.run(
        ["prism", "review", "--path", str(sample_repo), "--base", "main", "--head", "feature"],
        cwd=sample_repo,
        capture_output=True,
        text=True,
        timeout=REVIEW_TIMEOUT_SECONDS,
        check=False,
    )

    # V1 断言：命令跑通
    assert result.returncode == 0, f"命令失败：{result.stderr}"
    # V1 断言：有输出
    assert "Prism Code Review" in result.stdout
    assert "=== 统计 ===" in result.stdout
    # V1 不要求命中特定 bug
    # 阶段 3 再加断言：assert "goroutine" in result.stdout


def test_smoke_review_reports_progress(sample_repo: Path) -> None:
    """验证审查前的三个阶段都有进度输出。"""
    result = subprocess.run(
        ["prism", "review", "--path", str(sample_repo), "--base", "main", "--head", "feature"],
        cwd=sample_repo,
        capture_output=True,
        text=True,
        timeout=REVIEW_TIMEOUT_SECONDS,
        check=False,
    )

    assert result.returncode == 0, f"命令失败：{result.stderr}"
    assert "加载 diff…" in result.stdout
    assert "过滤文件…" in result.stdout
    assert "按包分组…" in result.stdout
    assert "第 1 组：" in result.stdout
    assert "审查完成：" in result.stdout


def test_smoke_review_without_api_key_fails_clearly(sample_repo: Path) -> None:
    """验证缺少 API key 时给出明确错误而非崩溃。"""
    # 继承当前环境（保住 venv 的 PATH），只摘掉 API key
    env = {key: value for key, value in os.environ.items() if key != API_KEY_ENV}
    result = subprocess.run(
        ["prism", "review", "--path", str(sample_repo), "--base", "main", "--head", "feature"],
        cwd=sample_repo,
        capture_output=True,
        text=True,
        timeout=REVIEW_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert API_KEY_ENV in result.stderr


def test_fixture_has_no_nested_git() -> None:
    """验证 fixture 是纯文件，没有嵌套仓库（否则主仓库无法跟踪）。"""
    assert (FIXTURE_DIR / "base").is_dir()
    assert (FIXTURE_DIR / "buggy").is_dir()
    assert not list(FIXTURE_DIR.rglob(".git"))
