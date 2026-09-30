"""hunk 报告工具。

LLM 只报告原始数据（结论 + 评论），进度由代码统计。
返回值只给「剩余未审」而非「已完成」，让返回体随进度递减。
"""

import logging
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.core.graph.state import MainState
from prism.core.tools.context import STATUS_DONE, STATUS_IN_PROGRESS
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)

# 合法严重程度，其余值原样保留（排序时会被排到最后）但打 WARNING
VALID_SEVERITIES = ("high", "medium", "low")


@tool
def report_hunks(
    reports: list[dict[str, Any]],
    runtime: ToolRuntime[None, MainState],
) -> str:
    """报告 hunk 的审查结论。

    Args:
        reports: hunk 报告列表，每项含：
            - hunk_id: str，hunk 编号（如 "H-01"）
            - complete: bool，是否审查完毕
            - comments: list[dict]，评论列表（可为空）
                每条评论含：
                - severity: str，high/medium/low
                - category: str，问题类别
                - content: str，评论正文
                - existing_code: str，用于定位的代码片段
                - suggestion_code: str，建议的修改（可选）
                - rule_ref: str，引用的规则编号（可选，V1 不用）

    Returns:
        进度信息 + 剩余未审 hunk 列表
    """
    ctx = runtime.state["tool_context"]
    logger.debug("report_hunks 调用：%d 个 hunk 报告", len(reports))

    recorded: list[str] = []
    unknown: list[str] = []
    new_comment_count = 0

    for raw in reports:
        hunk_id = str(raw.get("hunk_id", "")).strip()
        status = ctx.hunk_status.get(hunk_id)
        if status is None:
            unknown.append(hunk_id or "<空编号>")
            continue

        comments = [_to_comment(item) for item in raw.get("comments") or []]
        status.comments.extend(comments)
        ctx.all_comments.extend(comments)
        new_comment_count += len(comments)
        # complete=False 表示「报了部分结论但还没审完」，置为审查中
        status.status = STATUS_DONE if raw.get("complete") else STATUS_IN_PROGRESS
        recorded.append(hunk_id)

    done = ctx.done_count()
    total = len(ctx.hunk_status)
    pending = ctx.pending_ids()
    logger.info(
        "report_hunks：记录 %s，新增 %d 条评论，进度 %d/%d",
        ", ".join(recorded) or "无",
        new_comment_count,
        done,
        total,
    )
    if unknown:
        logger.warning("report_hunks 收到未知 hunk 编号：%s", ", ".join(unknown))

    parts: list[str] = []
    if recorded:
        parts.append(f"已记录 {len(recorded)} 个 hunk：{', '.join(recorded)}")
        parts.append(f"新增评论 {new_comment_count} 条")
    else:
        parts.append("未记录任何 hunk")
    if unknown:
        parts.append(f"⚠️ 未知 hunk 编号（已忽略）：{', '.join(unknown)}")
    parts.append(f"进度：{done}/{total} 完成")
    parts.append(f"剩余未审：{', '.join(pending) if pending else '无'}")
    return "\n".join(parts)


def _to_comment(raw: dict[str, Any]) -> CommentInput:
    """把 LLM 报告的评论字典转成 CommentInput。

    字段缺失补空串；severity 不在合法集合内时原样保留并打 WARNING
    （排序阶段会把未知 severity 排到最后，不在这里静默改写）。

    Args:
        raw: LLM 报告的评论字典

    Returns:
        CommentInput 对象
    """
    severity = str(raw.get("severity", "")).strip().lower()
    if severity not in VALID_SEVERITIES:
        logger.warning("评论 severity 非法：%r（将排在最后）", severity)

    return CommentInput(
        severity=severity,
        category=str(raw.get("category", "")).strip(),
        content=str(raw.get("content", "")).strip(),
        existing_code=str(raw.get("existing_code", "")).strip(),
        suggestion_code=_optional_text(raw.get("suggestion_code")),
        rule_ref=_optional_text(raw.get("rule_ref")),
    )


def _optional_text(value: object) -> str | None:
    """把可选字段转成非空字符串或 None。

    Args:
        value: 原始值

    Returns:
        去空白后的文本；为空或 None 时返回 None
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
