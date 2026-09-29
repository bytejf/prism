"""hunk → 函数映射的单元测试。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from prism.core.ast.mapper import (
    TOP_LEVEL_LABEL,
    format_location,
    format_symbols,
    hunk_line_range,
    map_hunks_to_symbols,
)
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Diff, Hunk


def make_hunk(old_start: int, old_lines: int, new_start: int, new_lines: int) -> Hunk:
    """构造测试用 Hunk。

    Args:
        old_start: 旧文件起始行
        old_lines: 旧文件行数
        new_start: 新文件起始行
        new_lines: 新文件行数

    Returns:
        Hunk 对象
    """
    return Hunk(
        old_start=old_start,
        old_lines=old_lines,
        new_start=new_start,
        new_lines=new_lines,
        content="",
    )


def make_item(hunk: Hunk, symbols: list[Symbol], hunk_id: str = "H-01") -> HunkWithSymbol:
    """构造测试用 HunkWithSymbol。

    Args:
        hunk: hunk
        symbols: 归属符号
        hunk_id: hunk 编号

    Returns:
        HunkWithSymbol 对象
    """
    return HunkWithSymbol(hunk=hunk, file_path="a.go", symbols=symbols, hunk_id=hunk_id)


def test_hunk_line_range_new_side() -> None:
    """测试取新文件侧行号范围。"""
    assert hunk_line_range(make_hunk(10, 4, 12, 6), use_old_side=False) == (12, 17)


def test_hunk_line_range_old_side() -> None:
    """测试取旧文件侧行号范围。"""
    assert hunk_line_range(make_hunk(10, 4, 12, 6), use_old_side=True) == (10, 13)


def test_hunk_line_range_pure_deletion() -> None:
    """测试纯删除 hunk：new_lines 为 0 时按锚点推算新文件侧范围。"""
    # @@ -10,3 +9,0 @@ → 被删的 3 行在新文件中本应位于 10..12
    assert hunk_line_range(make_hunk(10, 3, 9, 0), use_old_side=False) == (10, 12)
    assert hunk_line_range(make_hunk(10, 3, 9, 0), use_old_side=True) == (10, 12)


def test_format_location_multi_line_and_single_line() -> None:
    """测试位置渲染：多行显示区间，单行只显示一个行号。"""
    assert format_location(make_item(make_hunk(1, 3, 245, 16), [])) == "H-01 a.go:245-260"
    assert format_location(make_item(make_hunk(1, 1, 22, 1), [])) == "H-01 a.go:22"


def test_format_symbols_empty_is_top_level() -> None:
    """测试无归属符号时显示 <顶层>。"""
    assert format_symbols([]) == TOP_LEVEL_LABEL


def test_format_symbols_with_kind_labels() -> None:
    """测试归属符号带中文类型标签。"""
    symbols = [
        Symbol(name="(*Agent).Run", kind="method", start_line=1, end_line=9),
        Symbol(name="Resolve", kind="function", start_line=11, end_line=20),
    ]
    assert format_symbols(symbols) == "(*Agent).Run（方法）, Resolve（函数）"


# --- 真实仓库集成测试 ---

BASELINE_GO = """package main

import "fmt"

func Run() {
\tfmt.Println("start")
}

func Helper() {
\tfmt.Println("help")
}
"""

MODIFIED_GO = """package main

import "fmt"

func Run() {
\tfmt.Println("start")
\tfmt.Println("extra")
}

func Helper() {
\tfmt.Println("help")
}
"""

# 只有包级常量，没有任何函数声明：改动无法归属到函数
CONSTS_GO = """package main

const (
\tMaxRetry = 3
)
"""


@pytest.fixture
def go_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造含 Go 文件的临时仓库：feature 分支在 Run 内加了一行。

    Returns:
        仓库路径
    """
    repo = tmp_path / "go"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "svc.go").write_text(BASELINE_GO, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "svc.go").write_text(MODIFIED_GO, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_map_hunks_attributes_to_all_overlapping_functions(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试 hunk 归属到全部重叠函数。

    diff 带 3 行上下文，改动虽然落在 Run 内，但 hunk 范围会溢出到 Helper，
    按文档「hunk 跨多个函数 → 归属到所有重叠函数」应两者都归属。
    """
    diffs = filter_files(load_diffs(go_repo, "main", "feature")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "feature", diffs)

    assert len(mapped) == 1
    item = mapped[0]
    assert item.hunk_id == "H-01"
    assert item.file_path == "svc.go"
    assert [s.name for s in item.symbols] == ["Run", "Helper"]

    # 真正新增的那一行确实落在 Run 内部
    added_line = MODIFIED_GO.splitlines().index('\tfmt.Println("extra")') + 1
    run = next(s for s in item.symbols if s.name == "Run")
    assert run.start_line <= added_line <= run.end_line


def test_map_hunks_ids_are_globally_sequential(go_repo: Path, git: Callable[..., None]) -> None:
    """测试 hunk 编号全局连续，按文件路径排序分配。"""
    # 基于 feature 再开一个分支并新增 a.go：相对 main 有 svc.go（改）+ a.go（新增）
    git(go_repo, "checkout", "-q", "-b", "more", "feature")
    (go_repo / "a.go").write_text("package main\n\nfunc A() {\n\tprintln(1)\n}\n", encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "more")

    diffs = filter_files(load_diffs(go_repo, "main", "more")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "more", diffs)

    assert [item.hunk_id for item in mapped] == ["H-01", "H-02"]
    assert [item.file_path for item in mapped] == ["a.go", "svc.go"]
    assert [s.name for s in mapped[0].symbols] == ["A"]
    assert "Run" in [s.name for s in mapped[1].symbols]


def test_map_hunks_top_level_change_has_no_symbol(go_repo: Path, git: Callable[..., None]) -> None:
    """测试包级常量的改动无法归属到函数，标记为顶层。"""
    git(go_repo, "checkout", "-q", "-b", "consts", "main")
    (go_repo / "consts.go").write_text(CONSTS_GO, encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "add consts")
    (go_repo / "consts.go").write_text(
        CONSTS_GO.replace("MaxRetry = 3", "MaxRetry = 5"), encoding="utf-8"
    )
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "bump const")

    diffs = filter_files(load_diffs(go_repo, "main", "HEAD")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "HEAD", diffs)
    consts = next(item for item in mapped if item.file_path == "consts.go")

    assert consts.symbols == []
    assert format_symbols(consts.symbols) == TOP_LEVEL_LABEL


def test_map_hunks_deleted_file_uses_old_source(go_repo: Path, git: Callable[..., None]) -> None:
    """测试删除文件时用旧版本源码，仍能归属到函数。"""
    git(go_repo, "checkout", "-q", "-b", "del", "feature")
    git(go_repo, "rm", "-q", "svc.go")
    git(go_repo, "commit", "-q", "-m", "del")

    diffs = filter_files(load_diffs(go_repo, "feature", "del")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "feature", "del", diffs)

    assert mapped
    assert all(item.symbols for item in mapped)
    assert {s.name for item in mapped for s in item.symbols} == {"Run", "Helper"}


def test_map_hunks_empty_diffs(go_repo: Path) -> None:
    """测试空 diff 输入返回空列表。"""
    assert map_hunks_to_symbols(go_repo, "main", "feature", []) == []


def test_map_hunks_missing_file_has_no_symbol(go_repo: Path, git: Callable[..., None]) -> None:
    """测试 diff 里的文件在目标版本中不存在时不抛异常。"""
    diffs = load_diffs(go_repo, "main", "feature")
    fake = Diff(
        old_path="ghost.go",
        new_path="ghost.go",
        hunks=[make_hunk(1, 1, 1, 1)],
        insertions=1,
        deletions=0,
        is_new=False,
        is_deleted=False,
        is_binary=False,
    )
    mapped = map_hunks_to_symbols(go_repo, "main", "feature", [*diffs, fake])
    ghost = next(item for item in mapped if item.file_path == "ghost.go")
    assert ghost.symbols == []
