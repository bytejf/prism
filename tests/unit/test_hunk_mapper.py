"""hunk → 函数映射的单元测试。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from prism.core.ast.mapper import (
    TOP_LEVEL_LABEL,
    changed_lines,
    display_range,
    format_location,
    format_symbols,
    map_hunks_to_symbols,
)
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Diff, Hunk


def make_hunk(
    old_start: int, old_lines: int, new_start: int, new_lines: int, content: str = ""
) -> Hunk:
    """构造测试用 Hunk。

    Args:
        old_start: 旧文件起始行
        old_lines: 旧文件行数
        new_start: 新文件起始行
        new_lines: 新文件行数
        content: 原始 hunk 文本（含 @@ 行）

    Returns:
        Hunk 对象
    """
    return Hunk(
        old_start=old_start,
        old_lines=old_lines,
        new_start=new_start,
        new_lines=new_lines,
        content=content,
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


def test_display_range_prefers_new_side() -> None:
    """测试展示区间优先取新文件侧。"""
    assert display_range(make_hunk(10, 4, 12, 6)) == (12, 17)


def test_display_range_falls_back_to_old_side_for_pure_deletion() -> None:
    """测试纯删除 hunk 展示旧文件侧区间（新文件侧为空）。"""
    assert display_range(make_hunk(10, 3, 9, 0)) == (10, 12)


def test_changed_lines_reads_header_for_modification() -> None:
    """测试改动行直接来自头字段（-U0 下头字段就是改动行）。"""
    content = "\n".join(["@@ -11,1 +13,2 @@", "-old1", "+new1", "+new2"])
    added, removed = changed_lines(make_hunk(11, 1, 13, 2, content))
    assert added == [13, 14]
    assert removed == [11]


def test_changed_lines_pure_addition_has_no_removed() -> None:
    """测试纯新增 hunk（旧文件侧行数为 0）。"""
    content = "\n".join(["@@ -14,0 +16,1 @@", "+ErrInvalid"])
    added, removed = changed_lines(make_hunk(14, 0, 16, 1, content))
    assert added == [16]
    assert removed == []


def test_changed_lines_pure_deletion_has_no_added() -> None:
    """测试纯删除 hunk（新文件侧行数为 0）。"""
    content = "\n".join(["@@ -10,3 +9,0 @@", "-a", "-b", "-c"])
    added, removed = changed_lines(make_hunk(10, 3, 9, 0, content))
    assert added == []
    assert removed == [10, 11, 12]


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

# 只加一行注释在 Run 与 Helper 之间：改动不属于任何函数
MODIFIED_GAP_GO = BASELINE_GO.replace("func Helper() {", "// Helper 只做打印。\nfunc Helper() {")

# 只有包级常量，没有任何函数声明
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


def test_map_hunks_attributes_to_the_changed_function(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试归属到真正包含改动的函数。

    改动是 Run 里新增的一行，hunk（-U0）只含这一行，因此只归属 Run，
    不会牵连后面的 Helper。
    """
    diffs = filter_files(load_diffs(go_repo, "main", "feature")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "feature", diffs)

    assert len(mapped) == 1
    item = mapped[0]
    assert item.hunk_id == "H-01"
    assert item.file_path == "svc.go"
    assert [s.name for s in item.symbols] == ["Run"]

    added, _ = changed_lines(item.hunk)
    assert len(added) == 1
    lines = MODIFIED_GO.splitlines()
    assert lines[added[0] - 1].strip() == 'fmt.Println("extra")'


def test_map_hunks_between_functions_has_no_symbol(go_repo: Path, git: Callable[..., None]) -> None:
    """测试改动落在两个函数之间时归属为空。

    只加一行注释在 Run 的 `}` 与 func Helper 之间 —— 该行不属于任何函数，
    按改动行判定应得到空列表，而不是被挂到前后相邻的函数上。
    从 main 起分支以保证 hunk 里只有这一处改动。
    """
    git(go_repo, "checkout", "-q", "-b", "gap", "main")
    (go_repo / "svc.go").write_text(MODIFIED_GAP_GO, encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "gap")

    diffs = filter_files(load_diffs(go_repo, "main", "gap")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "gap", diffs)

    assert len(mapped) == 1
    item = mapped[0]
    assert item.symbols == []
    assert format_symbols(item.symbols) == TOP_LEVEL_LABEL

    # 反向确认改动确实落在两个函数之间：新增的那一行既不属于 Run 也不属于 Helper
    added, _ = changed_lines(item.hunk)
    lines = MODIFIED_GAP_GO.splitlines()
    assert [lines[ln - 1] for ln in added] == ["// Helper 只做打印。"]


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


def test_map_hunks_top_level_change_attributes_to_const_block(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试包级常量的改动归属到 const 块（而不是笼统的 <顶层>）。"""
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

    assert [s.name for s in consts.symbols] == ["const"]
    assert format_symbols(consts.symbols) == "const（const 块）"


def test_map_hunks_deleted_file_uses_old_source(go_repo: Path, git: Callable[..., None]) -> None:
    """测试删除文件时用旧版本源码，仍能归属到符号。"""
    git(go_repo, "checkout", "-q", "-b", "del", "feature")
    git(go_repo, "rm", "-q", "svc.go")
    git(go_repo, "commit", "-q", "-m", "del")

    diffs = filter_files(load_diffs(go_repo, "feature", "del")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "feature", "del", diffs)

    assert mapped
    assert all(item.symbols for item in mapped)
    names = {s.name for item in mapped for s in item.symbols}
    assert {"Run", "Helper"} <= names


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
