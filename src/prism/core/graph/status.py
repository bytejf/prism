"""`<agent_status>` 渲染。

每次 report_hunks 调用后追加一条 user 消息，让 LLM 看到当前进度，
避免重复报告已完成的 hunk 或漏审。
"""

from prism.core.tools.context import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    HunkStatus,
    ToolContext,
)
from prism.models.tool import CommentInput

# 包住进度快照的标签
STATUS_TAG_OPEN = "<agent_status>"
STATUS_TAG_CLOSE = "</agent_status>"

# 进度提示语
STATUS_FOOTER = "请确保所有 hunk 最终都标记为 complete。"


def build_agent_status(ctx: ToolContext) -> str:
    """渲染 hunk 审查进度快照。

    V1 只有 pending / done 两种状态流转，in_progress 分组固定为空，
    但保留该分组以对齐文档约定的输出格式。

    Args:
        ctx: 工具执行上下文

    Returns:
        `<agent_status>...</agent_status>` 文本
    """
    ordered = _sorted(ctx)
    done = [item for item in ordered if item.status == STATUS_DONE]
    in_progress = [item for item in ordered if item.status == STATUS_IN_PROGRESS]
    finished = (STATUS_DONE, STATUS_IN_PROGRESS)
    pending = [item for item in ordered if item.status not in finished]

    lines = [
        STATUS_TAG_OPEN,
        f"### Hunk 审查进度（{len(done)}/{len(ctx.hunk_status)} 完成）",
        "",
        f"✅ 完成审查（{len(done)}）：",
    ]
    lines.extend(f"  {item.hunk_id} — {_describe(item.comments)}" for item in done)

    lines.extend(("", f"🔄 审查中（{len(in_progress)}）："))
    lines.extend(f"  {item.hunk_id}" for item in in_progress)

    lines.extend(("", f"⏳ 待审查（{len(pending)}）："))
    if pending:
        lines.append(f"  {', '.join(item.hunk_id for item in pending)}")

    lines.extend(("", STATUS_FOOTER, STATUS_TAG_CLOSE))
    return "\n".join(lines)


def _sorted(ctx: ToolContext) -> list[HunkStatus]:
    """按编号排序取出 hunk 状态。

    Args:
        ctx: 工具执行上下文

    Returns:
        HunkStatus 列表，按 hunk_id 升序
    """
    return [ctx.hunk_status[hunk_id] for hunk_id in sorted(ctx.hunk_status)]


def _describe(comments: list[CommentInput]) -> str:
    """描述单个 hunk 的评论情况。

    Args:
        comments: 该 hunk 收集到的评论

    Returns:
        ``无问题`` 或 ``2 issues（high:1, medium:1）``
    """
    if not comments:
        return "无问题"

    counts: dict[str, int] = {}
    for comment in comments:
        counts[comment.severity] = counts.get(comment.severity, 0) + 1
    detail = ", ".join(f"{severity}:{count}" for severity, count in sorted(counts.items()))
    return f"{len(comments)} issues（{detail}）"
