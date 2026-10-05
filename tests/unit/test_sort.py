"""评论排序的单元测试。

整理与定位在 `test_locate.py`，这里只管顺序。
"""

from prism.core.output.sort import (
    UNKNOWN_SEVERITY_RANK,
    sort_comments,
)
from prism.models.comment import LOCATE_HUNK, SIDE_LEFT, SIDE_RIGHT, Comment


def make_comment(
    *,
    severity: str = "medium",
    path: str = "a.go",
    start_line: int = 0,
    hunk_id: str = "H-01",
    side: str | None = SIDE_RIGHT,
) -> Comment:
    """构造测试用 Comment。

    Args:
        severity: 严重程度
        path: 文件路径
        start_line: 起始行号
        hunk_id: hunk 编号
        side: 行号属于哪一侧（None 表示未定位）

    Returns:
        Comment 对象
    """
    return Comment(
        path=path,
        severity=severity,
        category="category",
        content="content",
        existing_code="existing",
        hunk_id=hunk_id,
        locate_level=LOCATE_HUNK,
        side=side,
        start_line=start_line,
    )


def test_sort_by_severity() -> None:
    """测试按 severity 排序，同级按文件路径升序。"""
    comments = [
        make_comment(severity="low", path="b.go"),
        make_comment(severity="high", path="a.go"),
        make_comment(severity="high", path="b.go"),
        make_comment(severity="medium", path="a.go"),
    ]
    sorted_comments = sort_comments(comments)

    assert [c.severity for c in sorted_comments] == ["high", "high", "medium", "low"]
    assert sorted_comments[0].path == "a.go"
    assert sorted_comments[1].path == "b.go"


def test_sort_within_same_file() -> None:
    """测试同文件内按行号排序。"""
    comments = [
        make_comment(severity="high", path="a.go", start_line=100),
        make_comment(severity="high", path="a.go", start_line=50),
    ]
    sorted_comments = sort_comments(comments)

    assert sorted_comments[0].start_line == 50
    assert sorted_comments[1].start_line == 100


def test_sort_unknown_severity_last() -> None:
    """测试未知 severity 排在最后。"""
    comments = [
        make_comment(severity="unknown", path="a.go"),
        make_comment(severity="high", path="b.go"),
        make_comment(severity="low", path="c.go"),
    ]
    sorted_comments = sort_comments(comments)

    assert sorted_comments[-1].severity == "unknown"
    assert UNKNOWN_SEVERITY_RANK == 3


def test_sort_does_not_interleave_the_two_sides() -> None:
    """测试同文件内先新增侧、再删除侧。

    两侧的行号是**两套坐标**（新增侧数新版本、删除侧数旧版本），混着比大小
    排出来的先后纯属巧合。分组展示反而清楚：先看新版上的问题，再看删掉的。
    """
    comments = [
        make_comment(severity="high", path="a.go", start_line=7, side=SIDE_LEFT),
        make_comment(severity="high", path="a.go", start_line=10, side=SIDE_RIGHT),
    ]

    ordered = sort_comments(comments)

    assert [c.side for c in ordered] == [SIDE_RIGHT, SIDE_LEFT]


def test_sort_unlocated_comes_last() -> None:
    """测试未定位的排在两侧之后（它会被渲染层挑出去单独展示）。"""
    comments = [
        make_comment(severity="high", path="a.go", start_line=0, side=None),
        make_comment(severity="high", path="a.go", start_line=7, side=SIDE_LEFT),
        make_comment(severity="high", path="a.go", start_line=10, side=SIDE_RIGHT),
    ]

    ordered = sort_comments(comments)

    assert [c.side for c in ordered] == [SIDE_RIGHT, SIDE_LEFT, None]


def test_sort_does_not_mutate_input() -> None:
    """测试排序不修改原始列表。"""
    comments = [
        make_comment(severity="low", path="b.go"),
        make_comment(severity="high", path="a.go"),
    ]
    original_order = [c.path for c in comments]
    _ = sort_comments(comments)

    assert [c.path for c in comments] == original_order


def test_sort_empty_list() -> None:
    """测试空列表。"""
    assert sort_comments([]) == []


def test_sort_is_stable_for_equal_keys() -> None:
    """测试键相同的评论保持原始收集顺序（可复现）。"""
    comments = [
        make_comment(severity="high", path="a.go", hunk_id="H-01"),
        make_comment(severity="high", path="a.go", hunk_id="H-02"),
        make_comment(severity="high", path="a.go", hunk_id="H-03"),
    ]
    assert [c.hunk_id for c in sort_comments(comments)] == ["H-01", "H-02", "H-03"]


def test_sort_is_deterministic() -> None:
    """测试同一输入两次排序结果一致。"""
    comments = [
        make_comment(severity="low", path="c.go"),
        make_comment(severity="high", path="a.go"),
        make_comment(severity="medium", path="b.go"),
    ]
    assert sort_comments(comments) == sort_comments(comments)
