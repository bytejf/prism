"""审查循环的 middleware。

`create_agent` 的默认退出条件是「模型不再调用工具」（见
`langchain/agents/factory.py` 的 `model_to_tools`）。**prism 的退出条件不同**：
模型返回纯文本时不停机，而是注入提醒让它继续；只有 end_done 被接受、
连续多轮无产出、或轮次耗尽时才结束。

好消息是这条默认条件可以被推翻 —— 路由函数里 `jump_to` 的优先级**高于**
经典退出判定，而 `JumpTo` 含 `"model"`，所以 `after_model` 返回
`jump_to="model"` 就能把控制权拿回来。

职责划分：

- 轮次上限交给框架的 `ModelCallLimitMiddleware`（它就是干这个的）
- 本模块只管**语义退出**：end_done 裁决、连续无产出、纯文本不停机
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents.middleware import after_model, before_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from prism.core.agent.state import (
    NO_PROGRESS_EXIT_ROUNDS,
    NO_PROGRESS_REMIND_ROUNDS,
    ReviewState,
    done_count,
    end_done_verdict,
    pending_ids,
)
from prism.core.agent.status import build_review_status

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


def update_no_report_rounds(current: int, tool_calls: Sequence[Mapping[str, Any]]) -> int:
    """按本轮工具调用更新 no_report_rounds。

    - 调用了 report_hunks / end_done → 0（有产出或有明确收尾意图）
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


def end_done_reset(state: ReviewState) -> dict[str, Any]:
    """有新 hunk 审完时，返回 end_done 计数的重置更新。

    「纠正机会」绑到进展上，不是全局计数：不重置的话，早先一次误调的
    end_done 会把机会提前耗尽，后面真误判时直接被接受并带缺口退出。

    这里而不是 `report_hunks` 里：工具节点一轮可以并行跑多个调用，两个工具
    同时写 `end_done_count` 会撞 LangGraph 的 InvalidUpdateError；middleware
    在模型节点里，是单写者。

    Args:
        state: 当前状态

    Returns:
        需要重置时返回 ``{"end_done_count": 0, "end_done_mark": <已完成数>}``，
        否则返回空字典
    """
    done = done_count(state["hunk_status"])
    if done <= state["end_done_mark"]:
        return {}
    return {"end_done_count": 0, "end_done_mark": done}


def should_exit(state: ReviewState, *, end_done_count: int | None = None) -> bool:
    """语义退出判定。

    轮次上限**不在这里** —— 那是 `ModelCallLimitMiddleware` 的职责。
    本函数只答「审查这件事本身结束了吗」。

    Args:
        state: 当前状态
        end_done_count: 覆盖状态里的 end_done 计数。本步刚重置过时传重置后的值 ——
            否则会拿旧计数把「刚有新进展」误判成「该收尾了」

    Returns:
        是否应当结束循环
    """
    pending = pending_ids(state["hunk_status"])
    count = state["end_done_count"] if end_done_count is None else end_done_count
    if count > 0 and end_done_verdict(count, pending)[0]:
        return True
    return state["no_report_rounds"] >= NO_PROGRESS_EXIT_ROUNDS


def _last_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """取**最近一条 AI 消息**的调用列表。

    工具跑完后消息尾部是 ToolMessage，要知道本轮调了什么得往上找那条 AI 消息
    —— 但**必须停在遇到的第一条 AI 消息上**，不能继续往前翻。

    之前是「找最近一条 tool_calls 非空的消息」，于是纯文本响应（`tool_calls`
    是空列表，假值）会被跳过、翻到上一轮的调用上去。后果：`review_pacing`
    误判「这轮有工具调用」，既不注入推进提醒也不 `jump_to="model"`，
    框架按默认的「没调工具就结束」收尾，而三个退出条件一个都不满足 ——
    整个审查以 `ValueError: 读不出 exit_reason` 崩掉。

    Args:
        messages: 消息历史

    Returns:
        工具调用列表；历史里没有 AI 消息时为空列表
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return [dict(call) for call in message.tool_calls]
    return []


@before_model(state_schema=ReviewState, can_jump_to=["end"], name="ReviewProgress")
def review_progress(state: ReviewState, _runtime: Any) -> dict[str, Any] | None:
    """模型调用前：追加进度快照，并判定语义退出。

    进度快照挂在这里而不是工具里，是为了**保持与旧实现完全相同的时序**：
    旧实现是「工具跑完 → 追加快照 → 判定退出 → 调模型」，这里一一对应。

    Args:
        state: 当前状态
        _runtime: 运行时。框架按位置传参（`func(state, runtime)`），位置必须留着，
            但本钩子用不到，故加下划线前缀表明「占位而非遗漏」

    Returns:
        状态更新；需要退出时带 ``jump_to="end"``
    """
    updates: dict[str, Any] = {}

    calls = _last_tool_calls(state["messages"])
    if any(str(call.get("name", "")) == "report_hunks" for call in calls):
        updates["messages"] = [HumanMessage(content=build_review_status(state["hunk_status"]))]

    reset = end_done_reset(state)
    if reset:
        logger.debug("新审完 hunk，重置 end_done 纠正机会")
        updates.update(reset)

    # 退出判定要看重置之后的值
    end_done_count = 0 if reset else state["end_done_count"]
    if should_exit(state, end_done_count=end_done_count):
        logger.info("退出：end_done 或连续无产出已达阈值")
        return {**updates, "jump_to": "end"}

    return updates or None


@after_model(state_schema=ReviewState, can_jump_to=["model", "end"], name="ReviewPacing")
def review_pacing(state: ReviewState, _runtime: Any) -> dict[str, Any] | None:
    """模型返回后：记轮次、更新无产出计数；纯文本时注入提醒并**继续循环**。

    纯文本返回时若不干预，框架会直接结束循环。这里返回 ``jump_to="model"``
    把控制权拿回来 —— 这是 prism 与框架默认行为唯一的正面冲突点。

    Args:
        state: 当前状态
        _runtime: 运行时，占位（同 `review_progress`）

    Returns:
        状态更新；纯文本时带提醒消息与 ``jump_to``
    """
    calls = _last_tool_calls(state["messages"])
    no_report = update_no_report_rounds(state["no_report_rounds"], calls)

    updates: dict[str, Any] = {
        "round_count": state["round_count"] + 1,
        "no_report_rounds": no_report,
    }

    if calls:
        # 有工具调用 → 交给框架执行工具，不干预路由
        return updates

    if no_report in NO_PROGRESS_REMIND_ROUNDS:
        logger.warning("连续 %d 轮无产出，注入提醒", no_report)
        content = NO_PROGRESS_REMINDER.format(rounds=no_report)
    else:
        logger.debug("模型未调用工具，注入推进提醒（连续无产出 %d 轮）", no_report)
        content = PLAIN_TEXT_REMINDER

    updates["messages"] = [HumanMessage(content=content)]
    # 与旧实现一致：提醒照样追加，到阈值才结束
    updates["jump_to"] = "end" if no_report >= NO_PROGRESS_EXIT_ROUNDS else "model"
    return updates
