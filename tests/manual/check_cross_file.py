"""跨文件场景校准：`file_read_diff` 这条路真的通不通。

**这个脚本回答一个问题**：一个文件改了接口，别的包的实现与调用方**没跟着改**
（它们不在 diff 里、也不在本组里），模型会不会自己去查、能不能查出来。

两个走查仓库都触发不了这个场景（`prism-min-test` 是单审查单元；`prism-tests`
的组内文件同属一个包），所以这里现场搭一个最小的三包仓库：

    pkg/a/iface.go    Runner.Run(ctx) → Run(ctx, opts ...Option)   ← 唯一的改动
    pkg/b/impl.go     实现未同步（不在 diff 里）
    pkg/c/caller.go   调用方未同步（不在 diff 里）

**它只在改过 `guidance/templates.py` 的工作方式段 / 硬约束段、
或改过 `core/tools/file_read_diff.py` 之后才需要跑** —— 这两处改了，
「模型会不会去查」这个前提就要重新验一次。

跑法（真调 LLM，1 组、约 8 轮、20 秒、一组 token）：

    PRISM_LOG_LEVEL=INFO uv run pytest tests/manual/check_cross_file.py -s

看什么（本文件不自己打 print 或 logger，全部走既有日志）：

    code_search(...) / file_read_diff(path=...)   ← 模型查了哪些文件（llm_trace，INFO）
    tool · 轮次=N · file_read_diff · 结果 M 字符   ← M 很小（约 40~50）就是「没改动」
    点 4 的打印                                    ← 最终评论

**期望**：日志里出现 `file_read_diff`，且它的参数指向 pkg/b 或 pkg/c；
最终有一条第 4 步打印出来的评论说「实现 / 调用方未同步更新」。
只有 0 次 `file_read_diff` 调用 = 提示词没把这条路教出去。
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from prism.config.settings import settings
from prism.core.agent.agent import run_review_agent
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import build_diff_index, load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.core.output.locate import locate_group
from prism.models.grouping import Group

BASE = "main"
HEAD = "HEAD"

_GO_MOD = "module xfile\n\ngo 1.22\n"

_IFACE_BEFORE = """\
package a

import "context"

// Runner 描述一个可执行的任务。
type Runner interface {
\tRun(ctx context.Context) error
}

// Option 配置 Runner。
type Option func(*config)

type config struct {
\tname string
}
"""

# 唯一的改动：接口多了一个可变参数
_IFACE_AFTER = _IFACE_BEFORE.replace(
    "\tRun(ctx context.Context) error", "\tRun(ctx context.Context, opts ...Option) error"
)

_IMPL = """\
package b

import (
\t"context"

\t"xfile/pkg/a"
)

// Svc 是 Runner 的一个实现。
type Svc struct {
\tname string
}

// Run 实现 a.Runner。
func (s *Svc) Run(ctx context.Context) error {
\treturn nil
}

// New 返回一个 Runner 实现。
func New(name string) a.Runner {
\treturn &Svc{name: name}
}
"""

_CALLER = """\
package c

import (
\t"context"

\t"xfile/pkg/a"
)

// Start 启动传入的 Runner。
func Start(ctx context.Context, runner a.Runner) error {
\treturn runner.Run(ctx)
}
"""


def _git(repo: Path, *args: str) -> None:
    """在仓库里跑一条 git 命令。

    Args:
        repo: 仓库路径
        args: git 参数
    """
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _write(repo: Path, path: str, text: str) -> None:
    """写入仓库内的一个文件。

    Args:
        repo: 仓库路径
        path: 仓库内相对路径
        text: 文件内容
    """
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def cross_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """搭一个「改了接口、实现与调用方没跟着改」的临时仓库。

    Args:
        tmp_path_factory: pytest 的临时目录工厂

    Yields:
        仓库路径，工作区停在 feature 分支
    """
    repo = tmp_path_factory.mktemp("xfile")
    _git(repo, "init", "-q", "-b", BASE)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")

    _write(repo, "go.mod", _GO_MOD)
    _write(repo, "pkg/a/iface.go", _IFACE_BEFORE)
    _write(repo, "pkg/b/impl.go", _IMPL)
    _write(repo, "pkg/c/caller.go", _CALLER)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "pkg/a/iface.go", _IFACE_AFTER)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat(a): Runner.Run 支持可选项")
    yield repo


@pytest.fixture(scope="module")
def reviewed(cross_repo: Path) -> Iterator[tuple[Group, list[str]]]:
    """跑一次真实审查（模块级缓存，同轮里多个用例复用）。

    Args:
        cross_repo: 临时仓库

    Yields:
        (审查单元, 每条评论的「路径:行号」)，评论正文已经打印出来
    """
    # 与 prism review 同一条链：diff → 过滤 → hunk 映射 → 分组 → 审查
    reviewable = filter_files(load_diffs(cross_repo, BASE, HEAD)).reviewable
    groups = group_by_package(map_hunks_to_symbols(cross_repo, BASE, HEAD, reviewable))
    model: BaseChatModel = LLMClient(settings.llm).model

    # 这里只有一组（另外两个包根本没进 diff），正是本场景的关键：
    # 正确结论依赖**不在本组、也不在 diff 里**的两个文件
    group = groups[0]
    result = run_review_agent(
        group, model, str(cross_repo), HEAD, diff_index=build_diff_index(reviewable)
    )
    located = locate_group(group, result.hunk_comments)

    # 断点最想看的是「模型最后说了什么」，打印出来省得去翻日志
    print(f"\n=== 定位后的评论（{len(located)} 条）===")
    for comment in located:
        print(f"[{comment.severity} · {comment.category}] {comment.path}:{comment.start_line}")
        print(comment.content)
    yield group, [f"{item.path}:{item.start_line}" for item in located]


def test_cross_file_lookup_ran(reviewed: tuple[Group, list[str]]) -> None:
    """占位用例：真正的判定在日志里（见模块 docstring）。

    **不断言**模型一定调了 `file_read_diff` —— 那是提示词的措辞在起作用还是
    模型这一次恰好想到了，单次运行分不出来（同 `calibrate_reflection.py`
    对「删除率」的态度：量出来的是趋势，不是阈值）。这里只断言审查跑完了。

    Args:
        reviewed: 模块级 fixture（审查已经跑完，本用例只是拿着结果看）
    """
    group, _locations = reviewed
    assert group.package_path == "pkg/a", "改动只应落在 pkg/a 这一组"
