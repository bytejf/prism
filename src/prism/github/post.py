"""把一份审查报告发到 PR 上。

**流程**（每一步失败都不会丢掉发现 —— 最差也会落进摘要）：

1. **删掉我们上次发的行内评论**（按标记认，只删自己的）
2. 能锚到行的评论**批量发**成一次 review
3. 更新**一条 sticky 摘要**（找不到就新发一条）

三个决定写在这里，都是取舍：

- **每次 push 全量重审、发之前先删旧的**：PR 上永远只有「当前这次审查」的结论。
  不做增量 —— diff 是 `merge-base(base, head)..head`，**base 一动被审的集合就变了**，
  上一轮的结论对应的是另一批 hunk，复用没有意义（详见 CLAUDE.md 的「GitHub Action」一节）
- **未定位的评论不进内联**：它们的 `side` 是 `None`，给不出 GitHub 要的坐标。
  硬发会被 422 拒掉**整批**（GitHub 的批量是原子的）。我们**静态就知道**哪些
  挂不上，不需要像 OCR 那样在 422 之后再去拉 diff 反推
- **建议改法用普通代码块，不用 ```suggestion**：GitHub 的 suggestion 块是**一键提交**
  的，而它替换的是「这条评论覆盖的整段行」—— 我们的 `suggestion_code` 与行区间
  未必严格对齐，一键提交就写出了错的代码。宁可让人手动复制
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from prism.core.output.renderer import (
    BADGE_TEMPLATE,
    FINDING_REF_LABEL,
    REF_TEMPLATE,
    RULE_REF_LABEL,
)
from prism.github.client import GitHubClient, GitHubError
from prism.models.comment import LOCATE_FAILED, SIDE_LEFT

logger = logging.getLogger(__name__)

# 每条内联评论正文里埋的标记：**认「这是我们发的」只靠它**（作者的登录名会变，
# 机器人的评论也可能有别人发的）。删旧评论、以及以后要做「同一处不重复发」都靠它
REVIEW_MARKER = "<!-- prism-review -->"
# sticky 摘要的标记：用来在 PR 里找到我们**上一次那条**，更新而不是新发
SUMMARY_MARKER = "<!-- prism-summary -->"

# 一次 review 里塞多少条内联评论。**50 是抄来的硬数字**：OCR 那边实测
# 「71 条一批触发过 GitHub 500（部分成功、服务端报错）」，而重试会重复发
MAX_COMMENTS_PER_REVIEW = 50

SEVERITY_ORDER = ("high", "medium", "low")


@dataclass
class PostOutcome:
    """一次投递的结果（给日志、给 CLI 的退出码）。

    Attributes:
        deleted: 删掉的上次评论数
        posted: 成功发出去的内联评论数
        routed_to_summary: 因为锚不上行而进摘要的评论数
        fallback_used: 是否走过 422 回退（真发生过就要能在日志里看见）
        summary_updated: 摘要是更新了旧的那条（False = 新发了一条）
    """

    deleted: int = 0
    posted: int = 0
    routed_to_summary: int = 0
    fallback_used: bool = False
    summary_updated: bool = False
    errors: list[str] = field(default_factory=list)


def render_comment_body(comment: dict[str, Any]) -> str:
    """一条评论的 markdown 正文。

    三段：徽章（类别 · 严重程度 + 引用）、正文、可选的建议改法。
    标记埋在最后 —— 它不该被人看见（HTML 注释在 GitHub 上不渲染）。

    Args:
        comment: JSON 报告里的一条评论

    Returns:
        markdown 正文
    """
    parts = [
        BADGE_TEMPLATE.format(category=comment["category"], severity=comment["severity"])
        + _refs(comment),
        str(comment["content"]).strip(),
    ]
    suggestion = comment.get("suggestion_code")
    if suggestion:
        parts.append(f"**建议改法**（手动复制）：\n\n```go\n{str(suggestion).strip()}\n```")
    parts.append(REVIEW_MARKER)
    return "\n\n".join(parts)


def _refs(comment: dict[str, Any]) -> str:
    """评论的引用后缀（规则代号 / 静态发现编号）。

    **复用终端那份文案模板**（`RULE_REF_LABEL` 等），不另写一套：同一个概念在
    两处用不同写法，读的人会以为它们不是一回事。这里只是数据来源不同 ——
    报告里是 dict，终端那边是 `Comment` 对象。

    Args:
        comment: 评论

    Returns:
        形如 `（规则 EH-G-03、静态 S-02）`；没有引用时是空串
    """
    items = []
    if comment.get("rule_ref"):
        items.append(RULE_REF_LABEL.format(ref=comment["rule_ref"]))
    if comment.get("finding_ref"):
        items.append(FINDING_REF_LABEL.format(ref=comment["finding_ref"]))
    return REF_TEMPLATE.format(items="、".join(items)) if items else ""


def build_inline_comments(comments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把报告里的评论转成 GitHub 的行内评论载荷。

    **只有定位成功的进得来**：未定位的 `side` 是 `None`，硬发会被 422 拒掉整批。

    坐标映射（这一步不需要算，我们的 `side` 就是照 GitHub 抄的）：
    `side="RIGHT"` 数被审查版本的行、`side="LEFT"` 数基准版本的行；
    跨行评论给 `start_line` + `start_side`，单行只给 `line`。

    Args:
        comments: JSON 报告里的评论

    Returns:
        GitHub `pulls.createReview` 的 `comments` 数组
    """
    payload: list[dict[str, Any]] = []
    for comment in comments:
        side = comment.get("side")
        start, end = int(comment.get("start_line") or 0), int(comment.get("end_line") or 0)
        if comment.get("locate_level") == LOCATE_FAILED or not side or start < 1 or end < 1:
            continue
        item: dict[str, Any] = {
            "path": comment["path"],
            "line": end,
            "side": side,
            "body": render_comment_body(comment),
        }
        if start != end:
            item["start_line"] = start
            item["start_side"] = side
        payload.append(item)
    return payload


def build_summary(
    report: dict[str, Any],
    *,
    posted: int,
    unlocated: Sequence[dict[str, Any]],
    note: str = "",
) -> str:
    """sticky 摘要的正文。

    摘要是**唯一能看见缺口的地方** —— PR 上看不到「有多少 hunk 没审完」、
    「有几条挂不上行」。本项目的原则是**不报缺口等于假装审完了**，所以缺口
    永远排在摘要最前面。

    Args:
        report: JSON 报告
        posted: 真正发出去的内联评论数
        unlocated: 未定位的评论
        note: 额外说明（如 422 回退、fork 跳过）

    Returns:
        markdown 正文
    """
    stats = report.get("stats", {})
    usage = report.get("usage", {})
    comments = report.get("comments", [])
    counts = dict.fromkeys(SEVERITY_ORDER, 0)
    for comment in comments:
        counts[comment.get("severity", "")] = counts.get(comment.get("severity", ""), 0) + 1
    severity_line = " / ".join(f"{name} {counts.get(name, 0)}" for name in SEVERITY_ORDER)
    extra = " / ".join(
        f"{name} {count}" for name, count in sorted(counts.items()) if name not in SEVERITY_ORDER
    )
    if extra:
        severity_line = f"{severity_line} / {extra}"

    lines = [SUMMARY_MARKER, "## Prism 审查摘要", ""]
    # **缺口排在最前面**：与终端报告同一条规矩（「在报告头部写『有 N 个 hunk 没能
    # 审完』」）—— 只读前两行的人也要知道这次审查是不完整的
    gaps = _gap_lines(stats)
    if gaps:
        lines.extend(["### ⚠️ 覆盖缺口", "", *gaps, ""])
    lines.extend(
        [
            f"- 评论 **{len(comments)} 条**（{severity_line}）",
            f"- 已发到行内 **{posted} 条**，未定位 **{len(unlocated)} 条**",
            f"- 文件 {stats.get('files_count', 0)} 个 · 花费 **{usage.get('cost_yuan', 0)} 元** · "
            f"用时 {stats.get('duration_seconds', 0)} 秒",
        ]
    )
    if note:
        lines.append(f"- ⚠️ {note}")
    if unlocated:
        lines.extend(
            [
                "",
                f"<details><summary>未定位的评论（{len(unlocated)} 条，挂不上具体行）</summary>",
                "",
                *_render_unlocated(unlocated),
                "",
                "</details>",
            ]
        )
    static_gaps = report.get("static_gaps") or []
    if static_gaps:
        lines.extend(
            [
                "",
                f"<details><summary>未被处理的静态提示（{len(static_gaps)} 条）</summary>",
                "",
                *[
                    f"- `{item.get('path')}:{item.get('line')}` [{item.get('linter')} · "
                    f"{item.get('priority')}] {item.get('text')}"
                    for item in static_gaps
                ],
                "",
                "</details>",
            ]
        )
    if report.get("invalid_rule_refs"):
        lines.append(
            f"\n⚠️ 有评论引用了不存在的规则代号："
            f"{', '.join(report['invalid_rule_refs'])}（依据存疑）"
        )
    return "\n".join(lines)


def _gap_lines(stats: dict[str, Any]) -> list[str]:
    """把三类缺口渲染成摘要里的行（都没有时返回空列表）。

    Args:
        stats: 报告里的 `stats`

    Returns:
        若干行 markdown
    """
    out = []
    if stats.get("pending_hunks"):
        out.append(f"- **{stats['pending_hunks']} 个 hunk 没能审完**（预算或异常中断）")
    if stats.get("skipped_groups"):
        out.append(f"- **{stats['skipped_groups']} 个审查单元没跑**（整条预算用完）")
    if stats.get("failed_groups"):
        out.append(f"- **{stats['failed_groups']} 个审查单元异常退出**（看日志）")
    return out


def _render_unlocated(comments: Sequence[dict[str, Any]]) -> list[str]:
    """未定位评论的 markdown 片段。

    Args:
        comments: 未定位的评论

    Returns:
        若干行 markdown
    """
    out = []
    for comment in comments:
        ref = _refs(comment)
        badge = BADGE_TEMPLATE.format(category=comment["category"], severity=comment["severity"])
        where = f" —— `{comment['locate_reason']}`" if comment.get("locate_reason") else ""
        out.append(f"- `{comment.get('path')}`{where}：{badge}{ref}")
        out.append(f"  {str(comment['content']).strip()}")
    return out


def post_report(
    client: GitHubClient,
    pr: int,
    report: dict[str, Any],
    *,
    head_sha: str,
    note: str = "",
) -> PostOutcome:
    """把一份报告发到 PR 上（删旧 → 发内联 → 更新摘要）。

    每一步的失败都**不静默**：内联发不出去就把发现挪进摘要（摘要里会写明原因），
    摘要也写不了就记 errors 并让调用方决定退出码。

    Args:
        client: GitHub 客户端
        pr: PR 编号
        report: `prism review --json-out` 写出来的报告
        head_sha: 评论锚在哪个提交上
        note: 要写进摘要的额外说明（如「本 PR 来自 fork，未审查」）

    Returns:
        投递结果
    """
    outcome = PostOutcome()
    comments = list(report.get("comments") or [])
    unlocated = [item for item in comments if item.get("locate_level") == LOCATE_FAILED]

    outcome.deleted = _delete_previous(client, pr, outcome)
    payload = build_inline_comments(comments)
    outcome.routed_to_summary = len(comments) - len(payload)

    if payload:
        outcome.posted, fallback_note = _post_batches(client, pr, head_sha, payload, outcome)
        if fallback_note:
            note = f"{note}；{fallback_note}" if note else fallback_note
        outcome.routed_to_summary += len(payload) - outcome.posted

    summary = build_summary(report, posted=outcome.posted, unlocated=unlocated, note=note)
    outcome.summary_updated = _upsert_summary(client, pr, summary, outcome)
    return outcome


def _delete_previous(client: GitHubClient, pr: int, outcome: PostOutcome) -> int:
    """删掉上次我们发的行内评论。

    删之前先扫全量（分页）—— 只读第一页会漏掉后面的，**而且不报错**。
    别人的评论一条都不碰：认的是我们自己的标记。

    Args:
        client: GitHub 客户端
        pr: PR 编号
        outcome: 收集错误

    Returns:
        删掉的条数
    """
    deleted = 0
    try:
        existing = client.list_review_comments(pr)
    except GitHubError as exc:
        outcome.errors.append(f"读旧评论失败：{exc}")
        return 0
    for item in existing:
        if REVIEW_MARKER not in str(item.get("body") or ""):
            continue
        try:
            client.delete_review_comment(int(item["id"]))
            deleted += 1
        except GitHubError as exc:
            # 删不掉不算致命：新的照发，最坏是 PR 上多留一份上一次的结论。
            # 但不静默 —— 水位不对时（比如 token 权限被降）这是唯一的信号
            logger.warning("删旧评论 %s 失败：%s", item.get("id"), exc)
            outcome.errors.append(f"删旧评论失败：{exc}")
    if deleted:
        logger.info("删掉上次发的 %d 条行内评论（本次重审全量重发）", deleted)
    return deleted


def _post_batches(
    client: GitHubClient,
    pr: int,
    head_sha: str,
    payload: list[dict[str, Any]],
    outcome: PostOutcome,
) -> tuple[int, str]:
    """分批发内联评论，失败时按「去掉 LEFT 侧」重试一次。

    **为什么留这个回退**：GitHub 的批量是**原子的** —— 一条评论的行号不被接受，
    整批 422，全都发不出去。而我们已经知道有一个场景可能让行号不被接受：
    **文件改名时 `LEFT` 侧的行住在旧路径下**，而我们的 `path` 只有新路径
    （CLAUDE.md 里记着这条「接 GitHub 时要确认」）。去掉 LEFT 侧再试一次，
    命中的话至少把绝大多数评论发出去，而不是全军覆没。

    回退本身**不静默**：摘要里会写明「本次有 N 条因为行号被拒而只进了摘要」。

    Args:
        client: GitHub 客户端
        pr: PR 编号
        head_sha: 锚定的提交
        payload: 评论载荷
        outcome: 收集结果

    Returns:
        (发出去的条数, 要写进摘要的说明)
    """
    posted = 0
    for start in range(0, len(payload), MAX_COMMENTS_PER_REVIEW):
        batch = payload[start : start + MAX_COMMENTS_PER_REVIEW]
        sent, note = _post_one_batch(client, pr, head_sha, batch, outcome)
        posted += sent
        if sent < len(batch):
            return posted, note
    return posted, ""


def _post_one_batch(
    client: GitHubClient,
    pr: int,
    head_sha: str,
    batch: list[dict[str, Any]],
    outcome: PostOutcome,
) -> tuple[int, str]:
    """发一批，必要时回退。

    Args:
        client: GitHub 客户端
        pr: PR 编号
        head_sha: 锚定的提交
        batch: 这一批的载荷
        outcome: 收集结果

    Returns:
        (发出去的条数, 说明)
    """
    try:
        client.create_review(pr, commit_id=head_sha, comments=batch)
        return len(batch), ""
    except GitHubError as exc:
        if exc.status != 422:
            logger.error("发内联评论失败（HTTP %s）：%s %s", exc.status, exc, exc.body[:200])
            outcome.errors.append(f"发内联评论失败：{exc}")
            return 0, f"内联评论发送失败（HTTP {exc.status}），结论已全部并入本摘要"

    without_left = [item for item in batch if item.get("side") != SIDE_LEFT]
    dropped = len(batch) - len(without_left)
    logger.warning(
        "批量发评论被 422 拒了（GitHub 的批量是原子的），去掉 %d 条 LEFT 侧评论重试",
        dropped,
    )
    outcome.fallback_used = True
    if not without_left:
        return 0, "评论坐标被 GitHub 拒绝（422），结论已全部并入本摘要"
    try:
        client.create_review(pr, commit_id=head_sha, comments=without_left)
    except GitHubError as exc:
        logger.error("去掉 LEFT 侧后仍然被拒（HTTP %s）：%s", exc.status, exc.body[:200])
        outcome.errors.append(f"发内联评论失败（回退后）：{exc}")
        return 0, "评论坐标被 GitHub 拒绝（422），结论已全部并入本摘要"
    return len(without_left), f"有 {dropped} 条（讲被删代码的）因坐标被拒，只进了本摘要"


def _upsert_summary(client: GitHubClient, pr: int, summary: str, outcome: PostOutcome) -> bool:
    """写 sticky 摘要：找到上次那条就更新，否则新发。

    **找不到时必须新发而不是放弃** —— 摘要是唯一能看见缺口的地方。

    Args:
        client: GitHub 客户端
        pr: PR 编号
        summary: 正文
        outcome: 收集错误

    Returns:
        True 表示更新了旧的那条
    """
    try:
        for item in client.list_issue_comments(pr):
            if SUMMARY_MARKER in str(item.get("body") or ""):
                client.update_issue_comment(int(item["id"]), summary)
                logger.info("更新了已有的摘要评论 %s", item["id"])
                return True
        client.create_issue_comment(pr, summary)
        logger.info("新发一条摘要评论")
        return False
    except GitHubError as exc:
        logger.error("写摘要失败：%s %s", exc, exc.body[:200])
        outcome.errors.append(f"写摘要失败：{exc}")
        return False
