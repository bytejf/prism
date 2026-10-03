"""评论整理与排序的单元测试。"""

from prism.core.agent.state import MainLoopResult
from prism.core.diff.parser import UNKNOWN_PATH
from prism.core.output.sort import (
    UNKNOWN_SEVERITY_RANK,
    build_hunk_index,
    sort_comments,
    to_comments,
)
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.comment import Comment
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# 排序只关心文件与行号，符号内容无关紧要
DUMMY_SYMBOL = Symbol(name="X", kind="function", start_line=1, end_line=1)


def make_comment(
    *,
    severity: str = "medium",
    path: str = "a.go",
    start_line: int = 0,
    hunk_id: str = "H-01",
) -> Comment:
    """构造测试用 Comment。

    Args:
        severity: 严重程度
        path: 文件路径
        start_line: 起始行号
        hunk_id: hunk 编号

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
        start_line=start_line,
    )


def make_group(hunk_paths: dict[str, str]) -> Group:
    """构造测试用审查单元。

    Args:
        hunk_paths: hunk 编号 → 文件路径

    Returns:
        Group 对象
    """
    hunks = [
        HunkWithSymbol(
            part=Hunk(old_start=1, old_lines=1, new_start=1, new_lines=1, content=""),
            file_path=path,
            symbol=DUMMY_SYMBOL,
            hunk_id=hunk_id,
            head_lines=(DUMMY_SYMBOL.start_line, DUMMY_SYMBOL.end_line),
            renamed_from=None,
            deleted=False,
        )
        for hunk_id, path in hunk_paths.items()
    ]
    files = sorted(set(hunk_paths.values()))
    return Group(group_id="group-001", package_path=".", files=files, hunks=hunks)


# ---------- 排序 ----------


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


# ---------- 整理 ----------


def test_build_hunk_index() -> None:
    """测试 hunk_id → 文件路径 索引。"""
    groups = [make_group({"H-01": "a.go", "H-02": "pkg/b.go"})]
    assert build_hunk_index(groups) == {"H-01": "a.go", "H-02": "pkg/b.go"}


def make_comment_input(content: str, severity: str = "medium") -> CommentInput:
    """构造测试用 CommentInput。

    Args:
        content: 评论正文
        severity: 严重程度

    Returns:
        CommentInput 对象
    """
    return CommentInput(
        severity=severity,
        category="category",
        content=content,
        existing_code="existing",
    )


def test_to_comments_fills_path_from_hunk_id() -> None:
    """测试用 hunk_id 补齐评论的文件路径。"""
    groups = [make_group({"H-01": "internal/agent/agent.go"})]
    result = MainLoopResult(
        group_id="group-001",
        exit_reason="end_done",
        round_count=2,
        hunk_comments={
            "H-01": [
                CommentInput(
                    severity="high",
                    category="concurrency",
                    content="goroutine 缺少退出路径",
                    existing_code="go func() {}()",
                    suggestion_code="加上 stop 通道",
                )
            ]
        },
    )

    comments = to_comments([result], groups)
    assert len(comments) == 1
    comment = comments[0]
    assert comment.path == "internal/agent/agent.go"
    assert comment.hunk_id == "H-01"
    assert comment.suggestion_code == "加上 stop 通道"
    assert comment.start_line == 0


def test_to_comments_unknown_hunk_uses_placeholder() -> None:
    """测试 hunk_id 查不到路径时用占位符且不崩溃。"""
    result = MainLoopResult(
        group_id="group-001",
        exit_reason="end_done",
        round_count=2,
        hunk_comments={"H-99": [make_comment_input("问题")]},
    )
    comments = to_comments([result], [make_group({"H-01": "a.go"})])
    assert comments[0].path == UNKNOWN_PATH


def test_to_comments_preserves_collection_order() -> None:
    """测试整理阶段不排序，保持收集顺序；排序交给 sort_comments。"""
    groups = [make_group({"H-01": "a.go"})]
    result = MainLoopResult(
        group_id="group-001",
        exit_reason="end_done",
        round_count=2,
        hunk_comments={
            "H-01": [
                make_comment_input("第一条", severity="low"),
                make_comment_input("第二条", severity="high"),
            ]
        },
    )

    comments = to_comments([result], groups)
    assert [c.content for c in comments] == ["第一条", "第二条"]

    ordered = sort_comments(comments)
    assert [c.content for c in ordered] == ["第二条", "第一条"]


def test_to_comments_across_multiple_groups() -> None:
    """测试跨多个组的评论一起整理，路径各自正确。"""
    groups = [make_group({"H-01": "pkg/a.go"}), make_group({"H-02": "pkg/b.go"})]
    results = [
        MainLoopResult("group-001", "end_done", 2, {"H-01": [make_comment_input("a")]}),
        MainLoopResult("group-002", "end_done", 2, {"H-02": [make_comment_input("b")]}),
    ]

    comments = to_comments(results, groups)
    assert [(c.hunk_id, c.path) for c in comments] == [("H-01", "pkg/a.go"), ("H-02", "pkg/b.go")]


def test_to_comments_empty() -> None:
    """测试无评论。"""
    assert to_comments([], []) == []
