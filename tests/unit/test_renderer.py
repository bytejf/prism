"""终端渲染的单元测试。"""

from prism.core.output.renderer import TerminalRenderer
from prism.models.comment import (
    LOCATE_FAILED,
    LOCATE_HUNK,
    REASON_EMPTY_EXISTING_CODE,
    REASON_MULTI_MATCH,
    REASON_NO_MATCH,
    SIDE_LEFT,
    SIDE_RIGHT,
    Comment,
    ReviewStats,
)
from prism.models.reflection import PotentialMiss
from prism.models.static import PRIORITY_HIGH, StaticFinding


def make_comment(
    *,
    path: str = "agent.go",
    severity: str = "high",
    category: str = "concurrency",
    content: str = "goroutine 缺少退出路径",
    suggestion_code: str | None = None,
    start_line: int = 10,
    end_line: int = 10,
    locate_level: str = LOCATE_HUNK,
    side: str | None = SIDE_RIGHT,
    locate_reason: str | None = None,
) -> Comment:
    """构造测试用 Comment。

    Args:
        path: 文件路径
        severity: 严重程度
        category: 问题类别
        content: 评论正文
        suggestion_code: 建议的修改
        start_line: 起始行号
        end_line: 结束行号
        locate_level: 定位层级
        side: 行号属于哪一侧
        locate_reason: 未定位原因

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
        locate_level=locate_level,
        side=side,
        locate_reason=locate_reason,
        suggestion_code=suggestion_code,
        start_line=start_line,
        end_line=end_line,
    )


def make_unlocated(
    *,
    path: str = "agent.go",
    content: str = "goroutine 缺少退出路径",
    locate_reason: str = REASON_NO_MATCH,
) -> Comment:
    """构造测试用未定位评论。

    Args:
        path: 文件路径
        content: 评论正文
        locate_reason: 未定位原因

    Returns:
        Comment 对象
    """
    return make_comment(
        path=path,
        content=content,
        locate_level=LOCATE_FAILED,
        side=None,
        locate_reason=locate_reason,
        start_line=0,
        end_line=0,
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
    """测试渲染单条评论：分隔符带行号。"""
    output = TerminalRenderer().render([make_comment()], make_stats())

    assert "─── agent.go:10 ───" in output
    assert "[concurrency · high]" in output
    assert "goroutine 缺少退出路径" in output


def test_render_multiline_span() -> None:
    """测试多行命中渲染成 `起始-结束`，单行只给一个行号。"""
    output = TerminalRenderer().render([make_comment(start_line=245, end_line=248)], make_stats())

    assert "─── agent.go:245-248 ───" in output


def test_render_deleted_side_is_marked() -> None:
    """测试删除侧的行号带标注。

    LEFT 的行号是**基准版本**的坐标，拿到当前文件里找是找不到的。不标的话
    读者会以为定位错了 —— 这类「看起来像 bug 的正确输出」最费时间。
    """
    output = TerminalRenderer().render(
        [make_comment(start_line=7, end_line=9, side=SIDE_LEFT)], make_stats()
    )

    assert "─── agent.go:7-9（已删除） ───" in output


def test_render_added_side_has_no_note() -> None:
    """测试新增侧不加标注 —— 那是默认读法，标了只是噪音。"""
    output = TerminalRenderer().render([make_comment(side=SIDE_RIGHT)], make_stats())

    assert "─── agent.go:10 ───" in output
    assert "已删除" not in output


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
    assert blocks[1].startswith("─── agent.go:10 ───")
    assert blocks[-1].startswith("=== 统计 ===")


# ---------- 未定位聚合 ----------


def test_unlocated_comments_are_listed_separately() -> None:
    """测试未定位的评论单独成段，**不混进正常评论列表**。

    混在一起的话，用户会以为行号已经核实过 —— 它们的位置是「不知道」。
    """
    located = make_comment(path="a.go", content="能定位的")
    failed = make_unlocated(path="b.go", content="定位失败的")
    output = TerminalRenderer().render([located, failed], make_stats())
    blocks = output.split("\n\n")

    assert "─── a.go:10 ───" in blocks[1]
    assert "=== 未定位评论（1 条）===" in output
    # 失败那条只在聚合段里出现，且聚合段在统计之前
    assert output.index("=== 未定位评论") < output.index("=== 统计 ===")
    assert blocks.index(next(b for b in blocks if b.startswith("=== 未定位评论"))) == 2


def test_unlocated_section_is_absent_when_all_located() -> None:
    """测试全部定位成功时不出现聚合段。"""
    output = TerminalRenderer().render([make_comment()], make_stats())

    assert "未定位评论" not in output


def test_unlocated_shows_reason() -> None:
    """测试每条未定位评论带出自己的原因 —— 排障方向取决于它。"""
    comments = [
        make_unlocated(content="没匹配上", locate_reason=REASON_NO_MATCH),
        make_unlocated(content="歧义", locate_reason=REASON_MULTI_MATCH),
        make_unlocated(content="没给锚点", locate_reason=REASON_EMPTY_EXISTING_CODE),
    ]
    output = TerminalRenderer().render(comments, make_stats())

    assert "=== 未定位评论（3 条）===" in output
    assert "existing_code 未在 diff 中匹配到" in output
    assert "existing_code 匹配到多处，无法确定是哪一处" in output
    assert "existing_code 为空" in output
    assert "1. agent.go" in output


def test_unlocated_unknown_reason_still_renders() -> None:
    """测试原因缺失或没见过的取值时不崩，给一句兜底说明。"""
    output = TerminalRenderer().render([make_unlocated(locate_reason="whoknows")], make_stats())

    assert "（原因：未定位）" in output


def test_unlocated_still_counted_in_footer() -> None:
    """测试未定位的评论照样计入统计 —— 评论数与统计不能对不上。"""
    comments = [make_comment(severity="high"), make_unlocated(path="b.go")]
    output = TerminalRenderer().render(comments, make_stats())

    assert "high: 2" in output


# ---------- 疑似漏报 ----------


def test_no_misses_section_when_empty() -> None:
    """测试没有漏报信号时不出现那一段。"""
    output = TerminalRenderer().render([make_comment()], make_stats())

    assert "疑似漏报" not in output


def test_misses_section_lists_hunk_and_description() -> None:
    """测试漏报段列出 hunk 编号、路径与描述。

    它比未定位评论更弱（连评论都不是），所以文案也写得更软 —— 「仅供参考」。
    """
    misses = [PotentialMiss(hunk_id="H-07", path="pkg/b.go", description="新加的解锁路径没人提")]

    output = TerminalRenderer().render([make_comment()], make_stats(), misses)

    assert "=== 疑似漏报（1 条）===" in output
    assert "1. H-07 pkg/b.go" in output
    assert "新加的解锁路径没人提" in output
    assert "仅供参考" in output


def test_misses_section_comes_last() -> None:
    """测试漏报段排在未定位段之后、统计之前 —— 信号越弱越靠后。"""
    comments = [make_comment(), make_unlocated(path="b.go")]
    misses = [PotentialMiss(hunk_id="H-07", path="c.go", description="疑似漏报")]

    output = TerminalRenderer().render(comments, make_stats(), misses)

    assert output.index("=== 未定位评论") < output.index("=== 疑似漏报")
    assert output.index("=== 疑似漏报") < output.index("=== 统计 ===")


def test_misses_do_not_change_the_comment_footer() -> None:
    """测试漏报不进评论统计 —— 它不是评论。"""
    misses = [PotentialMiss(hunk_id="H-07", path="c.go", description="疑似漏报")]

    output = TerminalRenderer().render([make_comment(severity="high")], make_stats(), misses)

    assert "high: 1" in output


def test_no_static_gaps_section_when_empty() -> None:
    """测试没有未处理的静态发现时不出现那一段。"""
    output = TerminalRenderer().render([make_comment()], make_stats())

    assert "静态发现未处理" not in output


def test_static_gaps_lists_id_position_and_text() -> None:
    """测试未处理段给出编号、位置与工具原话 —— 用户要能自己回查。"""
    gaps = [
        StaticFinding(
            finding_id="S-03",
            path="pkg/a/iface.go",
            line=15,
            linter="gosec",
            priority=PRIORITY_HIGH,
            text="G107: Potential HTTP request made with variable url",
        )
    ]

    output = TerminalRenderer().render([make_comment()], make_stats(), static_gaps=gaps)

    assert "=== 静态发现未处理（1 条）===" in output
    assert "S-03 pkg/a/iface.go:15 [gosec · high]" in output
    assert "G107: Potential HTTP request" in output
    assert "人工核查" in output


def test_static_gaps_do_not_enter_the_comment_footer() -> None:
    """静态发现不进评论统计 —— 它不是评论（设计 18 号 §2.1）。"""
    gaps = [
        StaticFinding(
            finding_id="S-03",
            path="pkg/a.go",
            line=1,
            linter="gosec",
            priority=PRIORITY_HIGH,
            text="x",
        )
    ]

    output = TerminalRenderer().render(
        [make_comment(severity="high")], make_stats(), static_gaps=gaps
    )

    assert "high: 1" in output
    assert "【" not in output


def test_static_gaps_come_after_misses() -> None:
    """测试静态发现段排在漏报段之后 —— 同样越弱越靠后。"""
    misses = [PotentialMiss(hunk_id="H-07", path="c.go", description="疑似漏报")]
    gaps = [
        StaticFinding(
            finding_id="S-01",
            path="pkg/a.go",
            line=1,
            linter="gosec",
            priority=PRIORITY_HIGH,
            text="x",
        )
    ]

    output = TerminalRenderer().render([make_comment()], make_stats(), misses, gaps)

    assert output.index("=== 疑似漏报") < output.index("=== 静态发现未处理")
    assert output.index("=== 静态发现未处理") < output.index("=== 统计 ===")
