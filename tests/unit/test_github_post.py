"""GitHub 投递的单元测试。

**不联网**：用一个假的 `GitHubClient` 记录调用、按需抛 `GitHubError`。要验的是
决定与顺序（删旧 → 发内联 → 写摘要），不是 HTTP 本身。
"""

from typing import Any, cast

import pytest

from prism.github.client import GitHubClient, GitHubError
from prism.github.post import (
    MAX_COMMENTS_PER_REVIEW,
    REVIEW_MARKER,
    SUMMARY_MARKER,
    build_inline_comments,
    build_summary,
    post_report,
    render_comment_body,
)


def poster(client: "FakeClient") -> GitHubClient:
    """把假客户端当成真客户端用（只在本文件里，见 `FakeClient` 的注释）。

    Args:
        client: 假客户端

    Returns:
        同一对象，类型标成 `GitHubClient`
    """
    return cast(GitHubClient, client)


def make_comment(
    *,
    path: str = "store.go",
    side: str | None = "RIGHT",
    start: int = 36,
    end: int = 36,
    level: str = "hunk",
    severity: str = "high",
    content: str = "读 map 不加锁，与 Save 并发会 fatal error",
    **extra: Any,
) -> dict[str, Any]:
    """造一条报告里的评论。

    Args:
        path: 文件路径
        side: 行号属于哪一侧（未定位给 None）
        start: 起始行
        end: 结束行
        level: 定位层级
        severity: 严重程度
        content: 正文
        **extra: 其余字段（suggestion_code / rule_ref / finding_ref 等）

    Returns:
        评论 dict
    """
    return {
        "path": path,
        "side": side,
        "start_line": start,
        "end_line": end,
        "locate_level": level,
        "locate_reason": None if level != "failed" else "no_match",
        "severity": severity,
        "category": "边界条件",
        "content": content,
        "suggestion_code": None,
        "rule_ref": None,
        "finding_ref": None,
        **extra,
    }


def make_report(comments: list[dict[str, Any]], **stats: Any) -> dict[str, Any]:
    """造一份最小报告。

    Args:
        comments: 评论列表
        **stats: 覆盖默认的 stats 字段

    Returns:
        报告 dict
    """
    base = {
        "files_count": 3,
        "comments_count": len(comments),
        "duration_seconds": 42.0,
        "pending_hunks": 0,
        "skipped_groups": 0,
        "failed_groups": 0,
    }
    return {
        "stats": {**base, **stats},
        "usage": {"cost_yuan": 0.32, "calls": 6},
        "comments": comments,
        "static_gaps": [],
        "invalid_rule_refs": [],
    }


class FakeClient:
    """记下每次调用，按需制造失败。

    **不继承 `GitHubClient`**：真客户端要令牌与仓库名才能构造，而这里只想验
    「调了哪些端点、什么顺序」。用例里用 `cast` 声明它满足同一套方法（本项目
    测试的既有手法），比造一个 `Protocol` 少一处会漂的声明。
    """

    def __init__(
        self,
        *,
        review_comments: list[dict[str, Any]] | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
        review_error: GitHubError | None = None,
        fail_reviews: int = 0,
    ) -> None:
        """初始化。

        Args:
            review_comments: 已有的行内评论
            issue_comments: 已有的普通评论
            review_error: 建 review 时抛的错
            fail_reviews: 前几次 create_review 抛错（验回退；给个大数就是一直失败）
        """
        self.review_comments = review_comments or []
        self.issue_comments = issue_comments or []
        self.review_error = review_error
        self.fail_reviews = fail_reviews
        self.calls: list[tuple[str, Any]] = []

    def list_review_comments(self, pr: int) -> list[dict[str, Any]]:
        """列出已有的行内评论。"""
        self.calls.append(("list_review_comments", pr))
        return self.review_comments

    def delete_review_comment(self, comment_id: int) -> None:
        """删一条。"""
        self.calls.append(("delete_review_comment", comment_id))

    def create_review(
        self, pr: int, *, commit_id: str, comments: list[dict[str, Any]], body: str = ""
    ) -> dict[str, Any]:
        """建 review（可按需失败）。"""
        self.calls.append(("create_review", comments))
        if self.review_error is not None and self.fail_reviews > 0:
            self.fail_reviews -= 1
            raise self.review_error
        return {}

    def list_issue_comments(self, pr: int) -> list[dict[str, Any]]:
        """列出普通评论。"""
        self.calls.append(("list_issue_comments", pr))
        return self.issue_comments

    def create_issue_comment(self, pr: int, body: str) -> dict[str, Any]:
        """发摘要。"""
        self.calls.append(("create_issue_comment", body))
        return {}

    def update_issue_comment(self, comment_id: int, body: str) -> dict[str, Any]:
        """更新摘要。"""
        self.calls.append(("update_issue_comment", (comment_id, body)))
        return {}


# ---------- 载荷 ----------


def test_inline_payload_carries_the_exact_coordinates() -> None:
    """单行评论只给 `line`，不给 `start_line` —— 给了反而要两边对齐才算数。"""
    payload = build_inline_comments([make_comment(side="RIGHT", start=36, end=36)])

    assert payload[0]["path"] == "store.go"
    assert payload[0]["line"] == 36
    assert payload[0]["side"] == "RIGHT"
    assert "start_line" not in payload[0]


def test_inline_payload_maps_a_deleted_line_to_the_left_side() -> None:
    """讲被删代码的评论走 LEFT + 基准版本的行号（我们与 GitHub 是同一套坐标）。"""
    payload = build_inline_comments([make_comment(side="LEFT", start=25, end=27)])

    assert payload[0]["side"] == "LEFT"
    assert payload[0]["line"] == 27, "end 是 GitHub 的 line，start 才是 start_line"
    assert payload[0]["start_line"] == 25
    assert payload[0]["start_side"] == "LEFT"


def test_unlocated_comments_never_go_inline() -> None:
    """未定位的评论不进内联 —— GitHub 的批量是**原子**的，一条坐标不合法整批 422。

    它们的 `side` 是 None，给不出坐标。硬发会把**已经定位好的那些一起拒掉**。
    """
    comments = [
        make_comment(),
        make_comment(level="failed", side=None, start=0, end=0),
    ]

    payload = build_inline_comments(comments)

    assert len(payload) == 1
    assert payload[0]["path"] == "store.go"


def test_comment_body_carries_the_marker_and_the_refs() -> None:
    """正文里要有标记（认「我们发的」只看它）与引用（与终端同一套写法）。"""
    body = render_comment_body(
        make_comment(rule_ref="CC-G-03", finding_ref="S-02", suggestion_code="return nil")
    )

    assert REVIEW_MARKER in body
    assert "（规则 CC-G-03、静态 S-02）" in body
    assert "[边界条件 · high]" in body
    # 建议改法用**普通**代码块：```suggestion 是一键提交，而我们的片段与行区间
    # 未必严格对齐，一键提交会写出错的代码
    assert "```go" in body and "```suggestion" not in body


# ---------- 投递顺序 ----------


def test_post_deletes_our_old_comments_before_posting() -> None:
    """先删我们上次发的，再发这次的 —— PR 上只留当前结论。"""
    client = FakeClient(
        review_comments=[
            {"id": 11, "body": f"上一次的评论\n{REVIEW_MARKER}"},
            {"id": 12, "body": "别人的评论，不能碰"},
        ]
    )

    outcome = post_report(poster(client), 7, make_report([make_comment()]), head_sha="abc")

    order = [name for name, _ in client.calls]
    assert order.index("delete_review_comment") < order.index("create_review")
    assert ("delete_review_comment", 11) in client.calls
    assert ("delete_review_comment", 12) not in client.calls, "别人的评论一条都不碰"
    assert outcome.deleted == 1


def test_post_updates_the_existing_summary_comment() -> None:
    """摘要**原地更新**，不新发 —— 否则每次 push 在时间线上堆一条。"""
    client = FakeClient(issue_comments=[{"id": 99, "body": f"旧摘要\n{SUMMARY_MARKER}"}])

    outcome = post_report(poster(client), 7, make_report([make_comment()]), head_sha="abc")

    assert outcome.summary_updated is True
    updated = [value for name, value in client.calls if name == "update_issue_comment"]
    assert updated and updated[0][0] == 99
    assert not [name for name, _ in client.calls if name == "create_issue_comment"]


def test_post_creates_a_summary_when_there_is_none() -> None:
    """找不到旧的摘要时必须新发 —— 摘要是唯一能看见缺口的地方。"""
    client = FakeClient()

    outcome = post_report(poster(client), 7, make_report([make_comment()]), head_sha="abc")

    assert outcome.summary_updated is False
    assert [name for name, _ in client.calls].count("create_issue_comment") == 1


def test_batches_are_capped() -> None:
    """超过一批上限就拆成多次 review（上限的来源见 post.py 的注释）。"""
    comments = [make_comment(start=i, end=i) for i in range(1, MAX_COMMENTS_PER_REVIEW + 6)]
    client = FakeClient()

    outcome = post_report(poster(client), 7, make_report(comments), head_sha="abc")

    batches = [value for name, value in client.calls if name == "create_review"]
    assert [len(batch) for batch in batches] == [MAX_COMMENTS_PER_REVIEW, 5]
    assert outcome.posted == MAX_COMMENTS_PER_REVIEW + 5


# ---------- 422 回退 ----------


def test_422_falls_back_to_dropping_left_side_comments() -> None:
    """整批被 422 拒 → 去掉 LEFT 侧重试一次（文件改名那条已知缺口就在这里）。

    回退**不静默**：摘要里要写明有 N 条只进了摘要。
    """
    error = GitHubError("422", status=422, body="line must be part of the diff")
    client = FakeClient(review_error=error, fail_reviews=1)
    comments = [make_comment(side="RIGHT", start=10), make_comment(side="LEFT", start=25)]

    outcome = post_report(poster(client), 7, make_report(comments), head_sha="abc")

    assert outcome.fallback_used is True
    assert outcome.posted == 1
    assert outcome.routed_to_summary == 1
    summary = next(value for name, value in client.calls if name == "create_issue_comment")
    assert "坐标被拒" in summary


def test_non_422_failure_does_not_retry() -> None:
    """不是 422 就不回退 —— 换个参数重试只会把失败重复一遍（同工具失败那条）。"""
    error = GitHubError("500", status=500, body="server error")
    client = FakeClient(review_error=error, fail_reviews=99)

    outcome = post_report(poster(client), 7, make_report([make_comment()]), head_sha="abc")

    assert outcome.fallback_used is False
    assert outcome.posted == 0
    assert len([name for name, _ in client.calls if name == "create_review"]) == 1
    assert outcome.errors, "失败要记下来，让调用方决定退出码"


# ---------- 摘要 ----------


def test_summary_puts_gaps_first() -> None:
    """缺口排在摘要最前面 —— 不报缺口等于假装审完了（本项目的老规矩）。"""
    report = make_report([make_comment()], pending_hunks=5, skipped_groups=2)

    summary = build_summary(report, posted=1, unlocated=[])

    assert "覆盖缺口" in summary
    assert "5 个 hunk 没能审完" in summary
    assert "2 个审查单元没跑" in summary
    assert summary.index("覆盖缺口") < summary.index("评论"), "缺口要在结论之前"


def test_summary_lists_unlocated_comments_in_a_collapsible_block() -> None:
    """未定位的评论正文进折叠块 —— 它们挂不上行，PR 里没别的地方能看到。"""
    unlocated = [make_comment(level="failed", side=None, content="这条挂不上行")]

    summary = build_summary(make_report(unlocated), posted=0, unlocated=unlocated)

    assert "<details>" in summary
    assert "这条挂不上行" in summary
    assert SUMMARY_MARKER in summary


def test_summary_mentions_invalid_rule_refs() -> None:
    """引用了不存在的规则代号要写进摘要 —— 依据存疑这件事得让人看见。"""
    report = make_report([make_comment()])
    report["invalid_rule_refs"] = ["EH-G-99"]

    summary = build_summary(report, posted=1, unlocated=[])

    assert "EH-G-99" in summary and "不存在" in summary


def test_summary_counts_by_severity() -> None:
    """按严重程度分档报数（读的人先看有多少 high）。"""
    report = make_report(
        [make_comment(severity="high"), make_comment(severity="low"), make_comment(severity="low")]
    )

    summary = build_summary(report, posted=3, unlocated=[])

    assert "high 1 / medium 0 / low 2" in summary
    assert "0.32 元" in summary


def test_note_is_rendered_into_the_summary() -> None:
    """额外说明（fork 跳过 / 回退原因）要进摘要 —— 否则那件事没人知道。"""
    summary = build_summary(make_report([]), posted=0, unlocated=[], note="本 PR 来自 fork，未审查")

    assert "本 PR 来自 fork，未审查" in summary


@pytest.mark.parametrize("stats", [{"pending_hunks": 0}, {"skipped_groups": 0}])
def test_summary_has_no_gap_section_when_there_is_no_gap(stats: dict[str, int]) -> None:
    """没有缺口时不写那一段（空标题会让人以为有东西没显示出来）。"""
    summary = build_summary(make_report([], **stats), posted=0, unlocated=[])

    assert "覆盖缺口" not in summary
