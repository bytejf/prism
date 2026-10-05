"""拿 git 当裁判校验定位的行号。

对真实仓库的每个 hunk、每一侧，把该侧的改动行原文当作 existing_code 送去
定位，再按定位结果里的 ``(side, start_line, end_line)`` **从 git 里读回那
几行** —— 文本对不上就说明行号算错了。

**这个校验有单测给不了的东西：期望值是 git 给的，不是我自己写进用例里的。**
单测钉得住「我以为的行号推进规则」，钉不住「我以为的规则就是 git 的规则」；
把两侧起点、改名后的旧路径、切片后的行号这些小算术全串起来之后，只有回到
真实仓库里逐字读一遍才能确认。

需要 prism 之外的真实仓库（见 CLAUDE.md「验证怎么做」），不在就跳过。
"""

import os
import subprocess
from pathlib import Path

import pytest

from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.output.locate import locate, normalize
from prism.models.ast import HunkWithSymbol
from prism.models.comment import LOCATE_HUNK, SIDE_LEFT, SIDE_RIGHT

# (名字, 仓库, base, head)
CASES: tuple[tuple[str, Path, str, str], ...] = (
    (
        "min-test",
        Path(os.getenv("PRISM_MIN_TEST_DIR", Path.home() / "Projects/python/prism-min-test")),
        "main",
        "HEAD",
    ),
    (
        "prism-tests",
        Path(os.getenv("PRISM_TESTS_DIR", Path.home() / "Projects/python/prism-tests")),
        "main",
        "feature",
    ),
)

AVAILABLE = [case for case in CASES if (case[1] / ".git").exists()]

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="两个走查仓库都不在，设 PRISM_MIN_TEST_DIR / PRISM_TESTS_DIR 指定路径",
)


def read_lines(repo: Path, rev: str, path: str, start: int, end: int) -> list[str] | None:
    """从 git 里读某个版本某个文件的若干行。

    空行不参与比对 —— 定位的区间会跨过空行（空行不承载定位信息，这是设计
    如此），所以两边都滤掉。

    Args:
        repo: 仓库路径
        rev: 版本标识
        path: 仓库内相对路径
        start: 起始行（1-based，含）
        end: 结束行（1-based，含）

    Returns:
        归一化后的非空行；文件在该版本不存在时为 None
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    lines = proc.stdout.splitlines()
    return [text for text in (normalize(line) for line in lines[start - 1 : end]) if text]


def side_revision(repo: Path, side: str, base: str, head: str) -> str:
    """读某一侧的内容该看哪个版本。

    RIGHT 看被审查版本；LEFT 看**基准版本** —— 注意是 merge-base，不是 base
    分支的尖端：prism 的 diff 就是按 merge-base 算的。

    Args:
        repo: 仓库路径
        side: LEFT / RIGHT
        base: 基准版本
        head: 被审查版本

    Returns:
        该侧的版本标识
    """
    if side == SIDE_RIGHT:
        return head
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def side_path(item: HunkWithSymbol, side: str) -> str:
    """某一侧的改动行住在哪个路径下。

    文件改过名时两侧路径不同：新侧用新名字，**旧侧用改名前的名字**。
    这个区别只有真的回仓库里读文件才会暴露 —— 定位本身只认 `file_path`。

    Args:
        item: 带符号的 hunk
        side: LEFT / RIGHT

    Returns:
        该侧的路径
    """
    if side == SIDE_LEFT:
        return item.renamed_from or item.file_path
    return item.file_path


def check(repo: Path, base: str, head: str) -> tuple[int, list[str]]:
    """校验一个仓库里所有 hunk 的两侧行号。

    Args:
        repo: 仓库路径
        base: 基准版本
        head: 被审查版本

    Returns:
        (校验的段数, 问题列表)
    """
    diffs = filter_files(load_diffs(repo, base, head)).reviewable
    groups = group_by_package(map_hunks_to_symbols(repo, base, head, diffs))
    revisions = {side: side_revision(repo, side, base, head) for side in (SIDE_LEFT, SIDE_RIGHT)}

    checked = 0
    problems: list[str] = []
    for group in groups:
        for item in group.hunks:
            for side, marker in ((SIDE_RIGHT, "+"), (SIDE_LEFT, "-")):
                raw = [line for line in item.part.content.splitlines() if line.startswith(marker)]
                if not raw:
                    continue
                # 整段拿去定位：片段越长越不容易多命中，也能一次校验整个区间
                found = locate(item.hunk_id, "\n".join(raw), group.hunks)
                where = f"{item.hunk_id} {side_path(item, side)} {side}"
                if found.level != LOCATE_HUNK or found.side != side:
                    # 多命中/空片段会落到这里，那是定位的正常结果，不是行号错
                    problems.append(f"{where} 未按预期定位到本 hunk：{found}")
                    continue
                expected = [normalize(line) for line in raw if normalize(line)]
                actual = read_lines(
                    repo, revisions[side], side_path(item, side), found.start_line, found.end_line
                )
                checked += 1
                if actual is None:
                    problems.append(f"{where}: 该版本里读不到文件")
                elif actual != expected:
                    problems.append(
                        f"{where} {found.start_line}-{found.end_line}\n"
                        f"    期望 {expected}\n    实读 {actual}"
                    )
    return checked, problems


def test_line_numbers_match_git() -> None:
    """测试：报出来的每个行号区间，回 git 里读出来都是那段原文。"""
    checked = 0
    failures: list[str] = []
    for name, repo, base, head in AVAILABLE:
        count, problems = check(repo, base, head)
        checked += count
        failures.extend(f"[{name}] {item}" for item in problems)

    assert checked > 0, "一段都没校验到，先检查仓库与分支参数对不对"
    assert not failures, "定位的行号与 git 里的原文对不上：\n" + "\n".join(failures)
