"""审查结束工具。

LLM 需要一个明确的结束信号。有未完成的 hunk 时第一次拒绝（给一次纠正机会），
第二次接受并返回结构化告警（兜底，不让流程卡死）。

「接受与否」的判定与 middleware 共用 `state.end_done_verdict`，
避免规则在两处漂移。
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState, end_done_verdict, pending_ids

logger = logging.getLogger(__name__)


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py
@tool(
    parse_docstring=True,
    description=(
        "结束审查。只有每个 hunk 都至少报告过一次时才能结束；"
        "还有没报告过的 hunk 时会返回拒绝消息并列出它们。"
    ),
)
def end_done(runtime: ToolRuntime[ReviewContext, ReviewState]) -> Command[Any]:
    """结束审查。

    描述走上面的 `description=` —— 以前它把本 docstring 整段（含 `Args:` /
    `Returns:`）发给了模型，是纯噪声。本工具没有模型可见的参数。

    Args:
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        状态更新（累加调用次数 + 结论文本的 ToolMessage）
    """
    state = runtime.state
    count = state["end_done_count"] + 1
    accepted, pending = end_done_verdict(count, pending_ids(state["hunk_status"]))

    if accepted and not pending:
        logger.info("end_done：全部 %d 个 hunk 审查完毕", len(state["hunk_status"]))
        text = "审查完成"
    elif not accepted:
        joined = ", ".join(pending)
        logger.warning("end_done 被拒绝：剩余 %s 未审", joined)
        text = f"⚠️ 剩余 {joined} 未审，请先 report_hunks"
    else:
        joined = ", ".join(pending)
        logger.warning("end_done 第 %d 次调用，接受结束但仍有未审 hunk：%s", count, joined)
        text = f"已接受结束请求。注意：{joined} 未被审查。"

    return Command(
        update={
            "end_done_count": count,
            "messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id)],
        }
    )
