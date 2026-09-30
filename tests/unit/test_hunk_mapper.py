"""hunk → 函数映射的单元测试。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from prism.core.ast.mapper import (
    changed_lines,
    display_range,
    format_location,
    format_symbol,
    map_hunks_to_symbols,
    split_by_symbol,
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
        content: hunk 正文（不含 @@ 行）；changed_lines 不读它，只读前四个数字

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


DUMMY = Symbol(name="X", kind="function", start_line=1, end_line=1)


def make_item(hunk: Hunk, symbol: Symbol = DUMMY, hunk_id: str = "H-01") -> HunkWithSymbol:
    """构造测试用 HunkWithSymbol。

    Args:
        hunk: hunk 切片
        symbol: 归属符号
        hunk_id: hunk 编号

    Returns:
        HunkWithSymbol 对象
    """
    return HunkWithSymbol(part=hunk, file_path="a.go", symbol=symbol, hunk_id=hunk_id)


def test_display_range_prefers_new_side() -> None:
    """测试展示区间优先取新文件侧。"""
    assert display_range(make_hunk(10, 4, 12, 6)) == (12, 17)


def test_display_range_falls_back_to_old_side_for_pure_deletion() -> None:
    """测试纯删除 hunk 展示旧文件侧区间（新文件侧为空）。"""
    assert display_range(make_hunk(10, 3, 9, 0)) == (10, 12)


def test_changed_lines_reads_header_for_modification() -> None:
    """测试改动行直接来自头字段（-U0 下头字段就是改动行）。

    只给四个数字，不传 content —— changed_lines 不读正文。
    """
    added, removed = changed_lines(make_hunk(11, 1, 13, 2))
    assert added == [13, 14]
    assert removed == [11]


def test_changed_lines_pure_addition_has_no_removed() -> None:
    """测试纯新增 hunk（旧文件侧行数为 0）。"""
    added, removed = changed_lines(make_hunk(14, 0, 16, 1))
    assert added == [16]
    assert removed == []


def test_changed_lines_pure_deletion_has_no_added() -> None:
    """测试纯删除 hunk（新文件侧行数为 0）。"""
    added, removed = changed_lines(make_hunk(10, 3, 9, 0))
    assert added == []
    assert removed == [10, 11, 12]


def test_format_location_multi_line_and_single_line() -> None:
    """测试位置渲染：多行显示区间，单行只显示一个行号。"""
    assert format_location(make_item(make_hunk(1, 3, 245, 16))) == "H-01 a.go:245-260"
    assert format_location(make_item(make_hunk(1, 1, 22, 1))) == "H-01 a.go:22"


def test_format_symbol_with_kind_label() -> None:
    """测试归属符号带中文类型标签。"""
    symbol = Symbol(name="(*Agent).Run", kind="method", start_line=1, end_line=9)
    assert format_symbol(symbol) == "(*Agent).Run（方法）"


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

# 注释直接贴在 func Helper 上方 —— 按 Go 约定它是 Helper 的文档注释
MODIFIED_DOC_GO = BASELINE_GO.replace("func Helper() {", "// Helper 只做打印。\nfunc Helper() {")

# 注释与 func Helper 之间隔了空行 —— 不是文档注释，无从归属
MODIFIED_ORPHAN_GO = BASELINE_GO.replace(
    "func Helper() {", "// 与 Helper 无关的独立说明。\n\nfunc Helper() {"
)

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
    assert item.symbol.name == "Run"

    added, _ = changed_lines(item.part)
    assert len(added) == 1
    lines = MODIFIED_GO.splitlines()
    assert lines[added[0] - 1].strip() == 'fmt.Println("extra")'


def test_map_hunks_doc_comment_falls_to_following_declaration(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试函数的文档注释归属到那个函数。

    注释在语法树里是独立节点，不挂在它描述的声明下面，纯语法视角下不归属
    任何符号。补偿策略：改动行是注释时向下「落」到注释块下方第一个非注释行。
    这里新增的注释紧贴 func Helper 上方，是它的文档注释 → 归属 Helper。
    """
    git(go_repo, "checkout", "-q", "-b", "doc", "main")
    (go_repo / "svc.go").write_text(MODIFIED_DOC_GO, encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "doc")

    diffs = filter_files(load_diffs(go_repo, "main", "doc")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "doc", diffs)

    assert len(mapped) == 1
    item = mapped[0]
    assert item.symbol.name == "Helper"

    added, _ = changed_lines(item.part)
    lines = MODIFIED_DOC_GO.splitlines()
    assert [lines[ln - 1] for ln in added] == ["// Helper 只做打印。"]


def test_map_hunks_drops_orphan_comment_hunk(go_repo: Path, git: Callable[..., None]) -> None:
    """测试无归属的 hunk 被丢弃。

    注释块与下一个声明之间有空行 → 按 Go 约定不是文档注释，落不到任何声明上。
    这类改动（独立注释、纯空行）没有可审的实体，送进 LLM 只是噪音，所以直接
    丢掉、不进结果 —— 而不是带着「<顶层>」传给下游。
    """
    git(go_repo, "checkout", "-q", "-b", "orphan", "main")
    (go_repo / "svc.go").write_text(MODIFIED_ORPHAN_GO, encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "orphan")

    diffs = filter_files(load_diffs(go_repo, "main", "orphan")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "orphan", diffs)

    assert mapped == []


def test_map_hunks_ids_are_globally_sequential(go_repo: Path, git: Callable[..., None]) -> None:
    """测试 hunk 编号全局连续，按文件路径排序分配。"""
    # 基于 feature 再开一个分支并新增 a.go：相对 main 有 svc.go（改）+ a.go（新增）
    git(go_repo, "checkout", "-q", "-b", "more", "feature")
    (go_repo / "a.go").write_text("package main\n\nfunc A() {\n\tprintln(1)\n}\n", encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "more")

    diffs = filter_files(load_diffs(go_repo, "main", "more")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "more", diffs)

    # a.go 是新增文件，整个文件都是新增行；按实体边界拆成 2 个子 hunk
    # （package + 空行、A），所以 3 个编号而不是 2 个
    assert [item.hunk_id for item in mapped] == ["H-01", "H-02", "H-03"]
    assert [item.file_path for item in mapped] == ["a.go", "a.go", "svc.go"]
    assert mapped[0].symbol.name == "package"
    assert mapped[1].symbol.name == "A"
    assert mapped[2].symbol.name == "Run"


def test_map_hunks_top_level_change_attributes_to_const_block(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试包级常量的改动归属到 const 块（而不是笼统的 <顶层>）。

    用两个分支比较（consts → bump），保证 hunk 里只有常量值这一处改动 ——
    否则新增文件的 hunk 会覆盖整个文件，把所有符号都命中。
    """
    git(go_repo, "checkout", "-q", "-b", "consts", "main")
    (go_repo / "consts.go").write_text(CONSTS_GO, encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "add consts")

    git(go_repo, "checkout", "-q", "-b", "bump")
    (go_repo / "consts.go").write_text(
        CONSTS_GO.replace("MaxRetry = 3", "MaxRetry = 5"), encoding="utf-8"
    )
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "bump const")

    diffs = filter_files(load_diffs(go_repo, "consts", "bump")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "consts", "bump", diffs)

    assert len(mapped) == 1
    assert mapped[0].symbol.name == "const"
    assert format_symbol(mapped[0].symbol) == "const（const 块）"


def test_map_hunks_deleted_file_uses_old_source(go_repo: Path, git: Callable[..., None]) -> None:
    """测试删除文件时用旧版本源码，仍能归属到符号。"""
    git(go_repo, "checkout", "-q", "-b", "del", "feature")
    git(go_repo, "rm", "-q", "svc.go")
    git(go_repo, "commit", "-q", "-m", "del")

    diffs = filter_files(load_diffs(go_repo, "feature", "del")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "feature", "del", diffs)

    assert mapped
    assert mapped
    names = {item.symbol.name for item in mapped}
    assert {"Run", "Helper"} <= names


def test_map_hunks_empty_diffs(go_repo: Path) -> None:
    """测试空 diff 输入返回空列表。"""
    assert map_hunks_to_symbols(go_repo, "main", "feature", []) == []


def test_map_hunks_missing_file_is_dropped(go_repo: Path, git: Callable[..., None]) -> None:
    """测试 diff 里的文件在目标版本中不存在时，其 hunk 无归属被丢弃、不抛异常。"""
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

    # 该文件取不到源码 → 符号表为空 → hunk 无归属 → 被丢弃
    assert mapped
    assert all(item.file_path != "ghost.go" for item in mapped)


# ---------- 按实体边界拆分 hunk ----------


def test_split_returns_same_hunk_when_single_symbol() -> None:
    """测试只属于一个符号时不拆分（原样返回同一个对象）。"""
    symbols = [Symbol("A", "function", 3, 5)]
    hunk = make_hunk(0, 0, 3, 3, "\n".join(["+\tx := 1", "+\t_ = x", "+}"]))
    assert split_by_symbol(hunk, symbols, symbols) == [hunk]


def test_split_new_file_by_entity_boundary() -> None:
    """测试新增文件的 hunk 按实体边界拆开，空隙行归给相邻子 hunk。"""
    symbols = [Symbol("package", "package", 1, 1), Symbol("A", "function", 3, 5)]
    hunk = make_hunk(
        0,
        0,
        1,
        5,
        "\n".join(["+package main", "+", "+func A() {", "+\tprintln(1)", "+}"]),
    )

    pieces = split_by_symbol(hunk, symbols, symbols)
    assert len(pieces) == 2
    # 子 hunk 1：package + 紧跟的空行（空行不属于任何符号，归给相邻子 hunk）
    assert changed_lines(pieces[0]) == ([1, 2], [])
    # 子 hunk 2：函数体
    assert changed_lines(pieces[1]) == ([3, 4, 5], [])


def test_split_preserves_lines_and_content() -> None:
    """测试拆分不丢内容：行号并集还原、正文拼接还原。"""
    symbols = [Symbol("package", "package", 1, 1), Symbol("A", "function", 3, 5)]
    hunk = make_hunk(
        0,
        0,
        1,
        5,
        "\n".join(["+package main", "+", "+func A() {", "+\tprintln(1)", "+}"]),
    )

    pieces = split_by_symbol(hunk, symbols, symbols)
    added = [ln for piece in pieces for ln in changed_lines(piece)[0]]
    removed = [ln for piece in pieces for ln in changed_lines(piece)[1]]
    assert sorted(added) == changed_lines(hunk)[0]
    assert sorted(removed) == changed_lines(hunk)[1]
    assert "\n".join(piece.content for piece in pieces) == hunk.content


def test_split_mixed_add_and_remove() -> None:
    """测试增删混合的 hunk 也能按实体边界拆开。"""
    # 旧文件：函数 A（1-2）；新文件：函数 A（1-2）+ 函数 B（4-5）
    symbols = [Symbol("A", "function", 1, 2), Symbol("B", "function", 4, 5)]
    hunk = make_hunk(
        1,
        2,
        1,
        4,
        "\n".join(["-line1", "-line2", "+line1", "+line2", "+", "+func B() {}"]),
    )

    pieces = split_by_symbol(hunk, symbols, symbols)
    assert len(pieces) == 2
    # 子 1：A 的删改 + 其后那个空行（空行归给相邻子 hunk）
    assert changed_lines(pieces[0]) == ([1, 2, 3], [1, 2])
    # 子 2：新增的函数 B
    assert changed_lines(pieces[1]) == ([4], [])


def test_split_result_has_at_most_one_symbol(go_repo: Path, git: Callable[..., None]) -> None:
    """测试拆分后每个 hunk 至多归属一个符号（本次改造的目标）。"""
    git(go_repo, "checkout", "-q", "-b", "multi", "main")
    (go_repo / "many.go").write_text(
        "\n".join(
            [
                "package main",
                "",
                'import "fmt"',
                "",
                "func A() {",
                "\tfmt.Println(1)",
                "}",
                "",
                "func B() {",
                "\tfmt.Println(2)",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "multi")

    diffs = filter_files(load_diffs(go_repo, "main", "multi")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "multi", diffs)

    assert mapped
    assert mapped
    # 新增文件被拆成多个子 hunk，整体仍覆盖全部行
    names = [item.symbol.name for item in mapped]
    assert "package" in names
    assert "A" in names
    assert "B" in names


def test_map_hunks_new_function_after_blank_line_is_kept(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """回归用例：切片首个改动行是空行时，整个切片不能被误丢。

    新增「空行 + 文档注释 + 函数」时，切片第一个改动行是空行（无归属），
    但它后面紧跟新函数。归属判定若只看第一个改动行，整个新函数会被判成
    「无归属」而被丢弃 —— 改动的实体凭空消失，且不报错。
    必须取第一个**有归属**的改动行。
    """
    git(go_repo, "checkout", "-q", "-b", "addfn", "main")
    (go_repo / "svc.go").write_text(
        BASELINE_GO + "\n// Extra 是新增的函数。\nfunc Extra() {}\n", encoding="utf-8"
    )
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "addfn")

    diffs = filter_files(load_diffs(go_repo, "main", "addfn")).reviewable
    mapped = map_hunks_to_symbols(go_repo, "main", "addfn", diffs)

    assert [item.symbol.name for item in mapped] == ["Extra"]
    added, _ = changed_lines(mapped[0].part)
    # 空行 + 文档注释 + 函数声明三行都在，没被丢
    assert len(added) == 3
