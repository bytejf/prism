"""终端渲染的单元测试。"""

from prism.core.output.renderer import TerminalRenderer
from prism.models.comment import Comment, ReviewStats


def make_comment(
    *,
    path: str = "agent.go",
    severity: str = "high",
    category: str = "concurrency",
    content: str = "goroutine 缺少退出路径",
    suggestion_code: str | None = None,
) -> Comment:
    """构造测试用 Comment。

    Args:
        path: 文件路径
        severity: 严重程度
        category: 问题类别
        content: 评论正文
        suggestion_code: 建议的修改

    Returns:
        Comment 对象
    """
    return Comment(
        path=path,
        severity=severity,
        category=category,
        content=content,
        existing_code="go func() {}()",
        hunk_id="H-01",
        suggestion_code=suggestion_code,
    )


def make_stats(
    *,
    base_branch: str = "main",
    head_sha: str = "HEAD",
    files_count: int = 3,
    comments_count: int = 5,
) -> ReviewStats:
    """构造测试用 ReviewStats。

    Args:
        base_branch: base 分支名
        head_sha: head 版本
        files_count: 文件数
        comments_count: 评论数

    Returns:
        ReviewStats 对象
    """
    return ReviewStats(
        base_branch=base_branch,
        head_sha=head_sha,
        files_count=files_count,
        comments_count=comments_count,
        duration_seconds=1.5,
    )


def test_render_single_comment() -> None:
    """测试渲染单条评论。"""
    output = TerminalRenderer().render([make_comment()], make_stats())

    assert "─── agent.go ───" in output
    assert "[concurrency · high]" in output
    assert "goroutine 缺少退出路径" in output


def test_render_empty_comments() -> None:
    """测试无评论时仍渲染头部与统计。"""
    output = TerminalRenderer().render([], make_stats(comments_count=0))

    assert "评论数：0" in output
    assert "=== 统计 ===" in output
    assert "high: 0 | medium: 0 | low: 0" in output


def test_render_header() -> None:
    """测试头部渲染。"""
    output = TerminalRenderer().render([], make_stats(files_count=3, comments_count=5))

    assert "=== Prism Code Review ===" in output
    assert "main" in output
    assert "文件数：3" in output
    assert "评论数：5" in output
    assert "耗时：1.5s" in output


def test_render_header_shows_range() -> None:
    """测试审查范围按 merge-base 展开。"""
    output = TerminalRenderer().render([], make_stats(base_branch="main", head_sha="feature"))
    assert "审查范围：merge-base(main, feature)..feature" in output


def test_render_footer_stats() -> None:
    """测试尾部统计按 high → medium → low 展示。"""
    comments = [
        make_comment(severity="high"),
        make_comment(severity="medium"),
        make_comment(severity="low"),
    ]
    output = TerminalRenderer().render(comments, make_stats())

    assert "high: 1" in output
    assert "medium: 1" in output
    assert "low: 1" in output


def test_render_footer_counts_unknown_severity() -> None:
    """测试未知 severity 也计入统计，避免评论数与统计对不上。"""
    comments = [make_comment(severity="high"), make_comment(severity="critical")]
    output = TerminalRenderer().render(comments, make_stats())

    assert "high: 1" in output
    assert "critical: 1" in output


def test_render_multiple_comments_keeps_order() -> None:
    """测试多条评论按传入顺序渲染。"""
    comments = [
        make_comment(path="a.go", severity="high", content="问题 1"),
        make_comment(path="b.go", severity="medium", content="问题 2"),
    ]
    output = TerminalRenderer().render(comments, make_stats())

    assert "问题 1" in output
    assert "问题 2" in output
    assert output.index("问题 1") < output.index("问题 2")


def test_render_simplifies_suggestion_code() -> None:
    """测试 V1 不渲染 suggestion_code 的 diff。"""
    comment = make_comment(suggestion_code="row := db.QueryRow(sql, id)")
    output = TerminalRenderer().render([comment], make_stats())

    assert "row := db.QueryRow(sql, id)" not in output


def test_render_blocks_separated_by_blank_line() -> None:
    """测试各段落之间有空行，格式与文档示例一致。"""
    output = TerminalRenderer().render([make_comment()], make_stats())
    blocks = output.split("\n\n")

    assert blocks[0].startswith("=== Prism Code Review ===")
    assert blocks[1].startswith("─── agent.go ───")
    assert blocks[-1].startswith("=== 统计 ===")
