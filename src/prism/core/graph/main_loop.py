"""Main 循环（StateGraph 手写）。

结构：
    [START] → [call_llm] ── 有 tool_calls ─→ [execute_tools] ─→ [check_exit]
                        └─ 无 tool_calls ─→ [inject_reminder] ─→ [check_exit]
    [check_exit] ── 满足退出条件 ─→ [END]
                 └─ 否则 ────────→ [call_llm]

与文档原图的一处差异：文档画的 `inject_reminder → call_llm` 直连，
但那样纯文本轮次永远不经过 check_exit，§6.1 要求的
「纯文本也累加 no_report_rounds」与「连续 8 轮无产出强制退出」都会失效，
循环可能永不结束。这里改为 `inject_reminder → check_exit`，
行为与文档的退出矩阵一致，退出判断仍只集中在 check_exit 一处。
"""

import logging
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from prism.core.graph.state import (
    EXIT_END_DONE,
    EXIT_END_DONE_WITH_GAPS,
    EXIT_MAX_ROUNDS,
    EXIT_NO_PROGRESS,
    NO_PROGRESS_EXIT_ROUNDS,
    NO_PROGRESS_REMIND_ROUNDS,
    MainLoopResult,
    MainState,
    max_rounds_for,
)
from prism.core.graph.status import build_agent_status
from prism.core.guidance.builder import build_diff_text, build_guidance, build_system_prompt
from prism.core.llm.client import message_text
from prism.core.tools.code_search import make_code_search_tool
from prism.core.tools.context import ToolContext
from prism.core.tools.end_done import end_done_verdict, make_end_done_tool
from prism.core.tools.file_read import make_file_read_tool
from prism.core.tools.report_hunks import make_report_hunks_tool
from prism.models.grouping import Group

logger = logging.getLogger(__name__)

# 触发 report_hunks / end_done 的工具名（命中则重置 no_report_rounds）
PROGRESS_TOOL_NAMES = frozenset({"report_hunks", "end_done"})

# 纯文本轮次的推进提示
PLAIN_TEXT_REMINDER = "\n".join(
    (
        "<agent_reminder>",
        "你没有调用任何工具。请调用 file_read / code_search 继续审查，",
        "或用 report_hunks 报告已完成审查的 hunk。",
        "如果所有 hunk 已审完，请调用 end_done 结束审查。",
        "</agent_reminder>",
    )
)

# 连续无产出到提醒阈值时的提示
NO_PROGRESS_REMINDER = "\n".join(
    (
        "<agent_reminder>",
        "你已经连续 {rounds} 轮未报告审查结论。",
        "请立即调用 report_hunks 报告已完成审查的 hunk，",
        "不要只探索不产出。",
        "如果所有 hunk 已审完，请调用 end_done 结束审查。",
        "</agent_reminder>",
    )
)


class ChatInvoker(Protocol):
    """Main 循环对 LLM 的最小依赖。

    只要求有 invoke，便于单元测试替换为 fake，不强绑 LLMClient。
    """

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool] | None = None) -> AIMessage:
        """调用 LLM。

        Args:
            messages: 消息列表
            tools: 可用的工具列表

        Returns:
            LLM 响应
        """
        ...


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    """按运行时上下文构造 4 个工具。

    Args:
        ctx: 工具执行上下文

    Returns:
        工具列表
    """
    return [
        make_file_read_tool(ctx),
        make_code_search_tool(ctx),
        make_report_hunks_tool(ctx),
        make_end_done_tool(ctx),
    ]


def update_no_report_rounds(current: int, tool_calls: list[dict[str, Any]]) -> int:
    """按本轮工具调用更新 no_report_rounds。

    - 调用了 report_hunks → 0（有审查产出）
    - 调用了 end_done → 0（明确的收尾意图，无论是否被拒绝）
    - 其他（纯 file_read / code_search、纯文本输出）→ +1

    Args:
        current: 当前计数
        tool_calls: 本轮的工具调用

    Returns:
        更新后的计数
    """
    names = {str(call.get("name", "")) for call in tool_calls}
    if names & PROGRESS_TOOL_NAMES:
        return 0
    return current + 1


def call_llm(state: MainState, llm: ChatInvoker) -> dict[str, Any]:
    """调用 LLM。

    Args:
        state: 当前状态
        llm: LLM 客户端

    Returns:
        状态更新（消息 + 轮次）
    """
    ctx = state["tool_context"]
    round_count = state["round_count"] + 1
    logger.info(
        "轮次开始：第 %d/%d 轮，消息 %d 条，连续无产出 %d 轮",
        round_count,
        state["max_rounds"],
        len(state["messages"]),
        state["no_report_rounds"],
    )

    response = llm.invoke(state["messages"], tools=build_tools(ctx))
    logger.debug(
        "LLM 返回：内容 %d 字符，tool_calls %d 个（%s）",
        len(message_text(response)),
        len(response.tool_calls),
        ", ".join(str(call.get("name", "")) for call in response.tool_calls) or "无",
    )
    return {"messages": [response], "round_count": round_count}


def execute_tools(state: MainState) -> dict[str, Any]:
    """串行执行本轮 tool_calls，把结果回灌进消息历史。

    Args:
        state: 当前状态

    Returns:
        状态更新（工具结果消息 + no_report_rounds，report_hunks 时追加进度快照）
    """
    ctx = state["tool_context"]
    last = state["messages"][-1]
    tool_calls = list(getattr(last, "tool_calls", []) or [])
    tool_map = {item.name: item for item in build_tools(ctx)}

    results: list[BaseMessage] = []
    for call in tool_calls:
        name = str(call.get("name", ""))
        args = call.get("args") or {}
        logger.debug("执行工具：%s(%s)", name, args)

        tool = tool_map.get(name)
        if tool is None:
            content = f"错误：未知工具 {name}"
            logger.warning("未知工具调用：%s", name)
        else:
            try:
                content = str(tool.invoke(args))
            except Exception as exc:  # 工具内部已尽量返回文本，这里兜底
                content = f"错误：工具 {name} 执行异常：{exc}"
                logger.warning("工具 %s 执行异常：%s", name, exc)

        logger.debug("工具返回：%s，%d 字符", name, len(content))
        results.append(ToolMessage(content=content, tool_call_id=str(call.get("id", ""))))

    updates: dict[str, Any] = {
        "messages": results,
        "no_report_rounds": update_no_report_rounds(state["no_report_rounds"], tool_calls),
    }

    # report_hunks 之后追加进度快照，让 LLM 知道还剩什么
    if any(str(call.get("name", "")) == "report_hunks" for call in tool_calls):
        updates["messages"] = [*results, HumanMessage(content=build_agent_status(ctx))]
    return updates


def inject_reminder(state: MainState) -> dict[str, Any]:
    """纯文本输出时注入推进消息。

    Args:
        state: 当前状态

    Returns:
        状态更新（提醒消息 + no_report_rounds）
    """
    no_report = state["no_report_rounds"] + 1
    if no_report in NO_PROGRESS_REMIND_ROUNDS:
        logger.warning("连续 %d 轮无产出，注入提醒", no_report)
        content = NO_PROGRESS_REMINDER.format(rounds=no_report)
    else:
        logger.debug("LLM 未调用工具，注入推进提醒（连续无产出 %d 轮）", no_report)
        content = PLAIN_TEXT_REMINDER
    return {"messages": [HumanMessage(content=content)], "no_report_rounds": no_report}


def check_exit(state: MainState) -> dict[str, Any]:
    """判断是否退出。

    优先级：end_done 的裁决 → 轮次耗尽 → 连续无产出。

    Args:
        state: 当前状态

    Returns:
        状态更新（exit_reason，或空 dict 表示继续）
    """
    ctx = state["tool_context"]

    if ctx.end_done_count > 0:
        accepted, pending = end_done_verdict(ctx)
        if accepted and not pending:
            logger.info("退出：end_done，全部 hunk 已审查完毕")
            return {"exit_reason": EXIT_END_DONE}
        if accepted:
            logger.warning("退出：end_done_with_gaps，未审 hunk：%s", ", ".join(pending))
            return {"exit_reason": EXIT_END_DONE_WITH_GAPS}

    if state["round_count"] >= state["max_rounds"]:
        logger.warning("退出：max_rounds，已达 %d 轮", state["max_rounds"])
        return {"exit_reason": EXIT_MAX_ROUNDS}

    if state["no_report_rounds"] >= NO_PROGRESS_EXIT_ROUNDS:
        logger.warning("退出：no_progress，连续 %d 轮无产出", state["no_report_rounds"])
        return {"exit_reason": EXIT_NO_PROGRESS}

    return {}


def route_after_llm(state: MainState) -> str:
    """根据 LLM 响应决定下一步。

    Args:
        state: 当前状态

    Returns:
        "execute_tools" 如果有 tool_calls，否则 "inject_reminder"
    """
    last = state["messages"][-1]
    return "execute_tools" if getattr(last, "tool_calls", None) else "inject_reminder"


def route_after_check(state: MainState) -> str:
    """根据退出条件决定下一步。

    Args:
        state: 当前状态

    Returns:
        "end" 或 "call_llm"
    """
    return "end" if state.get("exit_reason") else "call_llm"


def initial_state(group: Group, project_root: str, head_rev: str = "") -> MainState:
    """构造 Main 循环的初始状态。

    Args:
        group: 审查单元
        project_root: 项目根目录
        head_rev: 被审查的版本，工具据此读取源码而不是读工作区

    Returns:
        初始 MainState
    """
    ctx = ToolContext(project_root=project_root, head_rev=head_rev)
    ctx.init_hunks([item.hunk_id for item in group.hunks])

    user_content = "\n\n".join(
        (
            build_guidance(group),
            "### 本组 diff",
            "",
            build_diff_text(group),
        )
    )
    return MainState(
        messages=[
            SystemMessage(content=build_system_prompt()),
            HumanMessage(content=user_content),
        ],
        tool_context=ctx,
        round_count=0,
        max_rounds=max_rounds_for(len(group.hunks)),
        exit_reason=None,
        group=group,
        no_report_rounds=0,
    )


def build_graph(llm: ChatInvoker) -> Any:
    """组装 Main 循环的 StateGraph。

    Args:
        llm: LLM 客户端

    Returns:
        编译后的图
    """

    def call_llm_node(state: MainState) -> dict[str, Any]:
        """call_llm 节点，绑定 llm。"""
        return call_llm(state, llm)

    graph = StateGraph(MainState)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("inject_reminder", inject_reminder)
    graph.add_node("check_exit", check_exit)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"execute_tools": "execute_tools", "inject_reminder": "inject_reminder"},
    )
    graph.add_edge("execute_tools", "check_exit")
    graph.add_edge("inject_reminder", "check_exit")
    graph.add_conditional_edges(
        "check_exit", route_after_check, {"end": END, "call_llm": "call_llm"}
    )
    return graph.compile()


def run_main_loop(
    group: Group, llm: ChatInvoker, project_root: str, head_rev: str = ""
) -> MainLoopResult:
    """跑一个审查单元的 Main 循环。

    Args:
        group: 审查单元
        llm: LLM 客户端
        project_root: 项目根目录
        head_rev: 被审查的版本，工具据此读取源码而不是读工作区

    Returns:
        循环产出（退出原因、评论、未审 hunk 等）

    Raises:
        ValueError: 图在未设置 exit_reason 的情况下结束
    """
    state = initial_state(group, project_root, head_rev)
    max_rounds = state["max_rounds"]
    logger.info(
        "开始审查 %s（%s，%d 个文件，%d 个 hunk，上限 %d 轮）",
        group.group_id,
        group.package_path,
        len(group.files),
        len(group.hunks),
        max_rounds,
    )

    # 每个节点至少一跳，条件边再分叉，给足余量避免 LangGraph 的递归上限先触发
    final: MainState = build_graph(llm).invoke(
        state, config={"recursion_limit": max_rounds * 4 + 20}
    )

    exit_reason = final.get("exit_reason")
    if exit_reason is None:
        msg = "Main 循环结束时没有 exit_reason，图的路由有误"
        raise ValueError(msg)

    ctx = final["tool_context"]
    # 按 hunk 编号键控返回：CommentInput 不带 hunk_id，
    # 输出阶段要靠这个键补文件路径
    hunk_comments = {
        hunk_id: list(status.comments) for hunk_id, status in sorted(ctx.hunk_status.items())
    }
    total_comments = sum(len(items) for items in hunk_comments.values())
    logger.info(
        "结束审查 %s：exit_reason=%s，轮次 %d，评论 %d 条，未审 %d 个",
        group.group_id,
        exit_reason,
        final["round_count"],
        total_comments,
        len(ctx.pending_ids()),
    )
    return MainLoopResult(
        group_id=group.group_id,
        exit_reason=exit_reason,
        round_count=final["round_count"],
        hunk_comments=hunk_comments,
        pending_hunk_ids=ctx.pending_ids(),
    )
