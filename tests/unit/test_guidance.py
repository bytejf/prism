"""静态指导组装的单元测试。

两块分开测：`build_diff_block`（数据，XML）与 `build_requirements`（流程，md）。
"""

from prism.core.guidance.builder import build_diff_block, build_requirements
from prism.core.guidance.templates import REVIEW_DIMENSIONS, WORKFLOW
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk
from prism.models.grouping import Group


def make_item(
    hunk_id: str,
    file_path: str,
    symbol: Symbol,
    *,
    old: tuple[int, int] = (245, 16),
    new: tuple[int, int] | None = None,
    head_lines: tuple[int, int] | None = None,
    renamed_from: str | None = None,
    deleted: bool = False,
    content: str = "",
) -> HunkWithSymbol:
    """构造测试用 HunkWithSymbol。

    Args:
        hunk_id: hunk 编号
        file_path: 文件路径
        symbol: 归属符号
        old: (旧文件起始行, 行数)
        new: (新文件起始行, 行数)；默认与 old 相同
        head_lines: 归属符号在被审查版本里的行号范围
        renamed_from: 旧路径
        deleted: 是否被整体删除
        content: hunk 正文

    Returns:
        HunkWithSymbol 对象
    """
    new_start, new_lines = new if new is not None else old
    return HunkWithSymbol(
        part=Hunk(
            old_start=old[0],
            old_lines=old[1],
            new_start=new_start,
            new_lines=new_lines,
            content=content,
        ),
        file_path=file_path,
        symbol=symbol,
        hunk_id=hunk_id,
        head_lines=head_lines,
        renamed_from=renamed_from,
        deleted=deleted,
    )


def _group(*hunks: HunkWithSymbol, files: list[str] | None = None) -> Group:
    """把若干 hunk 包成一个审查单元。

    Args:
        hunks: hunk 列表
        files: 文件列表；不传时从 hunk 里推导

    Returns:
        Group 对象
    """
    return Group(
        group_id="group-001",
        package_path="internal/agent",
        files=files if files is not None else sorted({item.file_path for item in hunks}),
        hunks=list(hunks),
    )


def _fixture() -> Group:
    """构造含 3 个 hunk、2 个文件的审查单元。

    Returns:
        Group 对象
    """
    run = Symbol(name="(*Agent).Run", kind="method", start_line=240, end_line=270)
    resolve = Symbol(name="ResolveComment", kind="function", start_line=300, end_line=330)
    helper = Symbol(name="Helper", kind="function", start_line=90, end_line=140)
    return _group(
        make_item("H-01", "internal/agent/agent.go", run, old=(245, 16), head_lines=(240, 270)),
        make_item("H-02", "internal/agent/agent.go", resolve, old=(310, 11), head_lines=(300, 330)),
        make_item("H-03", "internal/diff/resolver.go", helper, old=(100, 41), head_lines=(90, 140)),
    )


def test_scope_summarises_hunk_span_and_file_count() -> None:
    """测试审查范围给出 hunk 区间与文件数。"""
    block = build_diff_block(_fixture())
    assert "### 审查范围" in block
    assert "本组 hunk：H-01 ~ H-03（共 3 个），分布在 2 个文件。" in block


def test_single_hunk_span_shows_one_id() -> None:
    """测试只有一个 hunk 时区间只显示一个编号。"""
    item = make_item(
        "H-07",
        "pkg/a.go",
        Symbol("F", "function", 20, 30),
        old=(22, 1),
        head_lines=(20, 30),
    )
    assert "本组 hunk：H-07（共 1 个），分布在 1 个文件。" in build_diff_block(_group(item))


def test_hunk_carries_id_position_symbol_and_lines() -> None:
    """测试 hunk 标签把编号、位置、归属、可读范围全挂在属性上。"""
    block = build_diff_block(_fixture())
    assert '<hunk id="H-01" old="245,16" new="245,16"' in block
    assert 'symbol="(*Agent).Run" kind="method" lines="240-270"' in block
    assert 'symbol="Helper" kind="function" lines="90-140"' in block


def test_hunk_is_grouped_under_its_file() -> None:
    """测试同一个文件的 hunk 收在同一个 file 标签里。"""
    block = build_diff_block(_fixture())
    assert '<file path="internal/agent/agent.go">' in block
    assert '<file path="internal/diff/resolver.go">' in block
    # H-01 / H-02 同文件，必须落在同一个 file 块内
    agent_block = block.split('<file path="internal/agent/agent.go">')[1].split("</file>")[0]
    assert "H-01" in agent_block
    assert "H-02" in agent_block
    assert "H-03" not in agent_block


def test_no_position_duplication() -> None:
    """测试每个 hunk 只被定义一次 —— 旧版在范围清单与正文里各标了一遍。

    这条是「信息不集中」那个问题的回归守卫：两处标编号会逼模型来回对照。
    开头那行摘要里的 `H-01 ~ H-03` 是计数，不是清单，不算重复。
    """
    block = build_diff_block(_fixture())
    for hunk_id in ("H-01", "H-02", "H-03"):
        assert block.count(f'id="{hunk_id}"') == 1
    assert "归属：" not in block


def test_hunk_body_is_verbatim() -> None:
    """测试 diff 正文原样进提示词，不加任何前缀。"""
    item = make_item(
        "H-01",
        "a.go",
        Symbol("F", "function", 1, 9),
        old=(7, 3),
        new=(6, 0),
        head_lines=(1, 9),
        content='-	if s == "" {\n-		return ""\n-	}',
    )
    assert '-	if s == "" {\n-		return ""\n-	}' in build_diff_block(_group(item))


def test_empty_side_renders_zero_count() -> None:
    """测试纯删除的新文件侧写成 `0` 而不是空区间。"""
    item = make_item(
        "H-01",
        "a.go",
        Symbol("F", "function", 1, 9),
        old=(1, 2),
        new=(0, 0),
        head_lines=(1, 9),
    )
    assert 'old="1,2" new="0,0"' in build_diff_block(_group(item))


def test_lines_omitted_when_symbol_gone_from_reviewed_version() -> None:
    """测试符号不在被审查版本里时不给 `lines` —— 给了就是指向别处代码。"""
    item = make_item(
        "H-01",
        "a.go",
        Symbol("Deprecated", "function", 24, 29),
        old=(24, 7),
        new=(18, 0),
        head_lines=None,
    )
    block = build_diff_block(_group(item))
    assert 'symbol="Deprecated" kind="function">' in block
    assert "lines=" not in block


def test_deleted_file_is_marked() -> None:
    """测试整文件删除时 file 标签带 deleted。"""
    item = make_item(
        "H-01",
        "gone/legacy.go",
        Symbol("Old", "function", 8, 12),
        old=(8, 6),
        new=(0, 0),
        head_lines=None,
        deleted=True,
    )
    assert '<file path="gone/legacy.go" deleted="true">' in build_diff_block(_group(item))


def test_renamed_file_carries_old_path() -> None:
    """测试改名时 file 标签带 renamed-from。"""
    item = make_item(
        "H-01",
        "rename/new.go",
        Symbol("Limit", "function", 3, 8),
        old=(7, 1),
        new=(7, 1),
        head_lines=(3, 8),
        renamed_from="rename/old.go",
    )
    assert '<file path="rename/new.go" renamed-from="rename/old.go">' in build_diff_block(
        _group(item)
    )


def test_plain_file_has_no_extra_attributes() -> None:
    """测试普通文件的 file 标签不带任何可选属性。"""
    block = build_diff_block(_fixture())
    assert '<file path="internal/agent/agent.go">' in block
    assert "renamed-from" not in block
    assert "deleted=" not in block


def test_empty_group() -> None:
    """测试没有 hunk 的组不崩溃。"""
    empty = Group(group_id="group-003", package_path="pkg", files=[], hunks=[])
    assert build_diff_block(empty) == "### 审查范围\n\n本组没有 hunk。"


def test_requirements_lists_all_dimensions() -> None:
    """测试五个审查维度都在文本里。"""
    text = build_requirements()
    for dimension in REVIEW_DIMENSIONS:
        assert dimension in text
    assert "1. **业务逻辑正确性**" in text
    assert "5. **副作用**" in text


def test_requirements_lists_workflow() -> None:
    """测试工作方式的每一条都在文本里。"""
    text = build_requirements()
    assert "### 工作方式" in text
    for step in WORKFLOW:
        assert f"- {step}" in text


def test_requirements_instructs_report_hunks_and_end_done() -> None:
    """测试要求里明确提到 report_hunks 与 end_done。"""
    text = build_requirements()
    assert "report_hunks" in text
    assert "end_done" in text
    assert "### 结束审查" in text


def test_diff_block_and_requirements_do_not_overlap() -> None:
    """测试 diff 块与要求块各管各的，没有重复段落。"""
    diff_block = build_diff_block(_fixture())
    requirements = build_requirements()
    assert "审查要求" not in diff_block
    assert "### 审查范围" not in requirements
    assert "report_hunks" not in diff_block


def test_is_deterministic() -> None:
    """测试同一输入产生同一输出。"""
    group = _fixture()
    assert build_diff_block(group) == build_diff_block(group)
    assert build_requirements() == build_requirements()
