"""审查结束工具。

LLM 需要一个明确的结束信号。有未完成的 hunk 时第一次拒绝（给一次纠正机会），
第二次接受并返回结构化告警（兜底，不让流程卡死）。
"""

import logging

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.core.graph.state import MainState
from prism.core.tools.context import ToolContext

logger = logging.getLogger(__name__)

# 允许强制接受结束前最多拒绝几次
MAX_REJECTIONS = 1


def end_done_verdict(ctx: ToolContext) -> tuple[bool, list[str]]:
    """判定 end_done 是否接受结束。

    Main 循环与工具共用本函数，避免判定规则在两个地方漂移。
    调用前 end_done 已经累加过 ``ctx.end_done_count``。

    Args:
        ctx: 工具执行上下文

    Returns:
        (是否接受结束, 未审的 hunk 编号列表)
    """
    pending = ctx.pending_ids()
    if not pending:
        return True, []
    return ctx.end_done_count > MAX_REJECTIONS, pending


@tool
def end_done(runtime: ToolRuntime[None, MainState]) -> str:
    """结束审查。

    只有所有 hunk 都标记为 complete 时才能结束。
    如果有未完成的 hunk，返回拒绝消息，提示剩余 hunk。

    Args:
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        "审查完成" 或 "⚠️ H-xx 未审，请先 report_hunks"
    """
    ctx = runtime.state["tool_context"]
    ctx.end_done_count += 1
    accepted, pending = end_done_verdict(ctx)

    if accepted and not pending:
        logger.info("end_done：全部 %d 个 hunk 审查完毕", len(ctx.hunk_status))
        return "审查完成"

    joined = ", ".join(pending)
    if not accepted:
        logger.warning("end_done 被拒绝：剩余 %s 未审", joined)
        return f"⚠️ 剩余 {joined} 未审，请先 report_hunks"

    logger.warning("end_done 第 %d 次调用，接受结束但仍有未审 hunk：%s", ctx.end_done_count, joined)
    return f"已接受结束请求。注意：{joined} 未被审查。"
