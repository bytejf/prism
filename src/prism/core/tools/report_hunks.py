"""hunk 报告工具。

LLM 只报告原始数据（结论 + 评论），进度由代码统计。
返回值只给「剩余未审」而非「已完成」，让返回体随进度递减。

状态回写走框架的 `Command`（工具更新 state 的正路），
不再是「改一个塞在 state 里的共享可变对象」。
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    HunkStatus,
    ReviewState,
    done_count,
    pending_ids,
)
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)

# 合法严重程度，其余值原样保留（排序时会被排到最后）但打 WARNING
VALID_SEVERITIES = ("high", "medium", "low")


@tool
def report_hunks(
    reports: list[dict[str, Any]],
    runtime: ToolRuntime[ReviewContext, ReviewState],
) -> Command[Any]:
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
        状态更新（新 hunk 状态表 + 新评论 + 进度文本的 ToolMessage）

    Notes:
        已标记 complete 的 hunk 会被拒绝重复报告（本次报告整体忽略）。
        一个 hunk 只想补充评论、还没审完时，请用 complete=false 报告。
    """
    state = runtime.state
    hunk_status = dict(state["hunk_status"])
    all_comments = list(state["all_comments"])
    end_done_count = state["end_done_count"]

    logger.debug("report_hunks 调用：%d 个 hunk 报告", len(reports))

    recorded: list[str] = []
    unknown: list[str] = []
    already_done: list[str] = []
    new_comment_count = 0
    newly_done = 0

    for raw in reports:
        hunk_id = str(raw.get("hunk_id", "")).strip()
        current = hunk_status.get(hunk_id)
        if current is None:
            unknown.append(hunk_id or "<空编号>")
            continue

        # 已审查完毕的 hunk 拒绝重复报告。评论是追加不是去重合并，
        # 同一个问题报 N 次就会变成 N 条评论 —— 而 LLM 反复确认同一个 hunk
        # 是自然行为（看到进度提示后想「再确认一下 H-01」）
        if current.status == STATUS_DONE:
            already_done.append(hunk_id)
            continue

        comments = [_to_comment(item) for item in raw.get("comments") or []]
        all_comments.extend(comments)
        new_comment_count += len(comments)

        # complete=False 表示「报了部分结论但还没审完」，置为审查中
        complete = bool(raw.get("complete"))
        if complete:
            newly_done += 1
        # 整体替换成新对象，不做原地修改
        hunk_status[hunk_id] = HunkStatus(
            hunk_id=hunk_id,
            status=STATUS_DONE if complete else STATUS_IN_PROGRESS,
            comments=[*current.comments, *comments],
        )
        recorded.append(hunk_id)

    # 「纠正机会」绑到进展上，不是全局计数：本轮有新 hunk 审完就重置。
    # 否则早先一次误调的 end_done 会把机会提前耗尽，后面真误判时没有第二次机会。
    if newly_done:
        end_done_count = 0

    done = done_count(hunk_status)
    total = len(hunk_status)
    pending = pending_ids(hunk_status)
    logger.info(
        "report_hunks：记录 %s，新增 %d 条评论，进度 %d/%d（重复报告被忽略 %d 个）",
        ", ".join(recorded) or "无",
        new_comment_count,
        done,
        total,
        len(already_done),
    )
    if unknown:
        logger.warning("report_hunks 收到未知 hunk 编号：%s", ", ".join(unknown))
    if already_done:
        logger.warning("report_hunks 重复报告已完成的 hunk，已忽略：%s", ", ".join(already_done))

    parts: list[str] = []
    if recorded:
        parts.append(f"已记录 {len(recorded)} 个 hunk：{', '.join(recorded)}")
        parts.append(f"新增评论 {new_comment_count} 条")
    else:
        parts.append("未记录任何 hunk")
    if already_done:
        parts.append(
            f"⚠️ 以下 hunk 已审查完毕（status=done），本次报告已忽略："
            f"{', '.join(already_done)}。"
            "如需补充评论，请对其他未完成的 hunk 调用 report_hunks。"
        )
    if unknown:
        parts.append(f"⚠️ 未知 hunk 编号（已忽略）：{', '.join(unknown)}")
    parts.append(f"进度：{done}/{total} 完成")
    parts.append(f"剩余未审：{', '.join(pending) if pending else '无'}")

    return Command(
        update={
            "hunk_status": hunk_status,
            "all_comments": all_comments,
            "end_done_count": end_done_count,
            "messages": [
                ToolMessage(content="\n".join(parts), tool_call_id=runtime.tool_call_id)
            ],
        }
    )


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
