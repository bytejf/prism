"""hunk 报告工具。

LLM 只报告原始数据（结论 + 评论），进度由代码统计。
返回值只给「剩余未审」而非「已完成」，让返回体随进度递减。

状态回写走框架的 `Command`（工具更新 state 的正路），
不再是「改一个塞在 state 里的共享可变对象」。
"""

import logging
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

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


class CommentArg(BaseModel):
    """LLM 报告的单条评论。"""

    # 类 docstring 会成为 schema 里的 description，**是模型看得到的**，所以只写
    # 一句给模型看的话；下面这条设计理由用注释写，别写进 docstring。
    #
    # 为什么把字段做成显式模型而不是 list[dict]：前者把字段说明摆到模型面前，
    # 实测 severity 漏填率 55% → 0（渲染成 `[类别 · ]` 的那批评论没有了）。
    severity: Literal["high", "medium", "low"] = Field(
        description=(
            "严重程度，必填。high=会导致错误/崩溃/数据损坏；"
            "medium=有风险或明显的维护性问题；low=风格与可读性"
        )
    )
    category: str = Field(description="问题类别，如 并发安全 / 边界条件 / 跨文件一致性")
    content: str = Field(description="评论正文：说清问题、影响与依据")
    existing_code: str = Field(
        description="用于定位的代码片段：本 hunk diff 里新增行的原文，逐行照抄（写法见系统提示词）"
    )
    suggestion_code: str | None = Field(default=None, description="建议的修改代码（可选）")
    rule_ref: str | None = Field(default=None, description="引用的规则编号（可选，V1 不用）")


class HunkReportArg(BaseModel):
    """单个 hunk 的审查结论。"""

    hunk_id: str = Field(description="hunk 编号，必须来自审查范围，如 H-01")
    complete: bool = Field(
        description="是否审查完毕。true=已审完（之后再报该 hunk 会被忽略）；"
        "false=只报部分结论，仍需继续审查"
    )
    comments: list[CommentArg] = Field(default_factory=list, description="评论列表，可为空")


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py。
#
# 描述里第 2 句是从 docstring 的 `Notes:` 段搬过来的：那条规则模型必须看到，
# 而 `parse_docstring=True` 会把 `Notes:` 剥掉，所以改成显式 `description=`。
@tool(
    parse_docstring=True,
    description=(
        "报告 hunk 的审查结论。\n"
        "已标记 complete 的 hunk 会被拒绝重复报告（本次报告整体忽略）；"
        "一个 hunk 只想补充评论、还没审完时，用 complete=false 报告。"
    ),
)
def report_hunks(
    reports: list[HunkReportArg],
    runtime: ToolRuntime[ReviewContext, ReviewState],
) -> Command[Any]:
    """报告 hunk 的审查结论。

    `reports` 的说明走下面的 `Args:`，**嵌套字段的说明走 `HunkReportArg` /
    `CommentArg` 的 `Field`** —— 嵌套结构只能由 Pydantic 模型表达。
    `runtime` 由框架注入、模型看不到，会被自动排除。

    Args:
        reports: hunk 报告列表
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        状态更新（新 hunk 状态表 + 新评论 + 进度文本的 ToolMessage）
    """
    state = runtime.state
    hunk_status = dict(state["hunk_status"])
    # 只把**本次改动**的 hunk 写回去（增量）。并行调用时两份增量按 key 合并，
    # 各改各的互不覆盖；写全量表会被后一份快照整个盖掉
    changed: dict[str, HunkStatus] = {}

    logger.debug("report_hunks 调用：%d 个 hunk 报告", len(reports))

    recorded: list[str] = []
    unknown: list[str] = []
    already_done: list[str] = []
    new_comment_count = 0

    for raw in reports:
        hunk_id = raw.hunk_id.strip()
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

        comments = [_to_comment(item) for item in raw.comments]
        new_comment_count += len(comments)

        # complete=False 表示「报了部分结论但还没审完」，置为审查中
        # 整体替换成新对象，不做原地修改
        updated = HunkStatus(
            hunk_id=hunk_id,
            status=STATUS_DONE if raw.complete else STATUS_IN_PROGRESS,
            comments=[*current.comments, *comments],
        )
        hunk_status[hunk_id] = updated
        changed[hunk_id] = updated
        recorded.append(hunk_id)

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

    # 写回去的是**增量** `changed`，不是全量 `hunk_status` —— 并行调用要靠
    # reducer 按 key 合并增量。另外 `end_done_count` 的重置挪到了
    # `review_progress`（middleware 是单写者，工具节点可以并行）：两个工具
    # 同时写同一个键会触发 LangGraph 的 InvalidUpdateError
    return Command(
        update={
            "hunk_status": changed,
            "messages": [ToolMessage(content="\n".join(parts), tool_call_id=runtime.tool_call_id)],
        }
    )


def _to_comment(arg: CommentArg) -> CommentInput:
    """把模型报的评论转成 CommentInput。

    severity 由 `Literal` 保证合法 —— 非法值在进到这里之前就被 Pydantic
    拦下、作为错误回灌给模型重试，不会静默变成空字符串。

    Args:
        arg: 模型报的评论（参数名不叫 `raw`，避免与调用点的循环变量同名，
            PyCharm 会把两者混起来推断）

    Returns:
        CommentInput 对象
    """
    return CommentInput(
        severity=arg.severity,
        category=arg.category.strip(),
        content=arg.content.strip(),
        existing_code=arg.existing_code.strip(),
        suggestion_code=_optional_text(arg.suggestion_code),
        rule_ref=_optional_text(arg.rule_ref),
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
