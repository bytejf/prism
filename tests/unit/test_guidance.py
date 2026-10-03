"""静态指导组装的单元测试。"""

import pytest

from prism.core.guidance.builder import build_guidance
from prism.core.guidance.templates import REVIEW_DIMENSIONS
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk
from prism.models.grouping import Group


def make_hunk_with_symbol(
    hunk_id: str, file_path: str, new_start: int, new_lines: int, symbol: Symbol
) -> HunkWithSymbol:
    """构造测试用 HunkWithSymbol。

    Args:
        hunk_id: hunk 编号
        file_path: 文件路径
        new_start: 新文件起始行
        new_lines: 新文件行数
        symbol: 归属符号

    Returns:
        HunkWithSymbol 对象
    """
    return HunkWithSymbol(
        part=Hunk(
            old_start=new_start,
            old_lines=new_lines,
            new_start=new_start,
            new_lines=new_lines,
            content="",
        ),
        file_path=file_path,
        symbol=symbol,
        hunk_id=hunk_id,
        head_lines=(symbol.start_line, symbol.end_line),
        renamed_from=None,
        deleted=False,
    )


@pytest.fixture
def group() -> Group:
    """构造含 3 个 hunk 的审查单元。

    Returns:
        Group 对象
    """
    return Group(
        group_id="group-001",
        package_path="internal/agent",
        files=["internal/agent/agent.go", "internal/diff/resolver.go"],
        hunks=[
            make_hunk_with_symbol(
                "H-01",
                "internal/agent/agent.go",
                245,
                16,
                Symbol(name="(*Agent).Run", kind="method", start_line=240, end_line=270),
            ),
            make_hunk_with_symbol(
                "H-02",
                "internal/agent/agent.go",
                310,
                11,
                Symbol(name="ResolveComment", kind="function", start_line=300, end_line=330),
            ),
            make_hunk_with_symbol(
                "H-03",
                "internal/diff/resolver.go",
                100,
                41,
                Symbol(name="Helper", kind="function", start_line=90, end_line=140),
            ),
        ],
    )


def test_build_guidance_has_all_sections(group: Group) -> None:
    """测试四个段落齐全（文档给的断言）。"""
    guidance = build_guidance(group)
    assert "审查范围" in guidance
    assert "审查要求" in guidance
    assert "报告节奏" in guidance
    assert "结束审查" in guidance
    assert "H-01" in guidance


def test_build_guidance_scope_lists_every_hunk(group: Group) -> None:
    """测试审查范围列出全部 hunk 的位置与归属。"""
    guidance = build_guidance(group)

    assert "本组 hunk：H-01 ~ H-03（共 3 个）" in guidance
    assert "H-01 internal/agent/agent.go:245-260" in guidance
    assert "归属：(*Agent).Run（方法）" in guidance
    assert "H-02 internal/agent/agent.go:310-320" in guidance
    assert "归属：ResolveComment（函数）" in guidance
    assert "H-03 internal/diff/resolver.go:100-140" in guidance
    assert "归属：Helper（函数）" in guidance


def test_build_guidance_single_hunk_span(group: Group) -> None:
    """测试只有一个 hunk 时区间只显示一个编号。"""
    single = Group(
        group_id="group-002",
        package_path="pkg",
        files=["pkg/a.go"],
        hunks=[make_hunk_with_symbol("H-07", "pkg/a.go", 22, 1, Symbol("F", "function", 20, 30))],
    )
    assert "本组 hunk：H-07（共 1 个）" in build_guidance(single)


def test_build_guidance_lists_all_dimensions(group: Group) -> None:
    """测试五个审查维度都在文本里。"""
    guidance = build_guidance(group)
    for dimension in REVIEW_DIMENSIONS:
        assert dimension in guidance
    assert "1. **业务逻辑正确性**" in guidance
    assert "5. **副作用**" in guidance


def test_build_guidance_instructs_report_hunks_and_end_done(group: Group) -> None:
    """测试引导里明确提到 report_hunks 与 end_done。"""
    guidance = build_guidance(group)
    assert "report_hunks" in guidance
    assert "end_done" in guidance


def test_build_guidance_empty_group() -> None:
    """测试没有 hunk 的组不崩溃。"""
    empty = Group(group_id="group-003", package_path="pkg", files=[], hunks=[])
    guidance = build_guidance(empty)
    assert "本组没有 hunk。" in guidance
    assert "审查要求" in guidance


def test_build_guidance_is_deterministic(group: Group) -> None:
    """测试同一输入产生同一输出。"""
    assert build_guidance(group) == build_guidance(group)
