"""审查循环的 middleware。

`create_agent` 的默认退出条件是「模型不再调用工具」（见
`langchain/agents/factory.py` 的 `model_to_tools`）。**prism 的退出条件不同**：
模型返回纯文本时不停机，而是注入提醒让它继续；只有 end_done 被接受、
连续多轮无产出、或轮次耗尽时才结束。

好消息是这条默认条件可以被推翻 —— 路由函数里 `jump_to` 的优先级**高于**
经典退出判定，而 `JumpTo` 含 `"model"`，所以 `after_model` 返回
`jump_to="model"` 就能把控制权拿回来。

职责划分：

- 止损交给**预算**（钱 / 时间 / 工具失败），不按轮数数 —— 轮数是代理指标，
  同一轮的成本能差几十倍（见 `budget.py`）
- 本模块只管**语义退出**：end_done 裁决、连续无产出、纯文本不停机
"""

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallRequest,
    after_model,
    hook_config,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from prism.config.settings import settings
from prism.core.agent.budget import (
    STOP_TOOL_FAILURES,
    add_usage,
    describe,
    exceeded,
    usage_of_message,
)
from prism.core.agent.compression import REFETCHABLE_TOOLS, Compressor, is_cleared
from prism.core.agent.context import ReviewContext
from prism.core.agent.state import (
    ReviewState,
    done_count,
    end_done_verdict,
    pending_ids,
)
from prism.core.agent.status import build_review_status
from prism.models.cost import UsageTotals

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


def _round_usage(messages: Sequence[BaseMessage]) -> UsageTotals:
    """取**本轮**那一条 AI 消息的用量。

    增量累加而不是最后扫一遍消息列表：压缩会把历史消息从列表里删掉，扫出来的账
    是漏的（实测大组一次能压 2 次）。这里取最后一条消息 —— `after_model` 就在模型
    刚返回、工具还没跑的时候，尾部必然是本轮那一条。

    Args:
        messages: 消息列表

    Returns:
        本轮用量；尾部不是 AI 消息时返回空用量
    """
    if not messages:
        return UsageTotals()
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return UsageTotals()
    return usage_of_message(last)


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

    资源止损（钱 / 时间 / 工具失败）**不在这里** —— 那是各自中间件的职责。
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
    return state["no_report_rounds"] >= settings.review.no_progress_exit_rounds


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


class ReviewProgress(AgentMiddleware[ReviewState, ReviewContext, Any]):
    """模型调用前：追加进度快照、必要时压缩历史，并判定语义退出。

    进度快照挂在这里而不是工具里，是为了**保持与旧实现完全相同的时序**：
    旧实现是「工具跑完 → 追加快照 → 判定退出 → 调模型」，这里一一对应。
    压缩也挂这里，理由相同 —— `before_model` 是唯一能同时看到「工具跑完的状态」
    与「还没发出去的请求」的位置。

    写成**类式**而不是 `@before_model` 装饰模块级函数：装饰器产出的中间件实例是
    **全局共享**的（框架缺口，见 CLAUDE.md），而压缩持有跨轮次状态（压缩次数、
    连续失败数、熔断标记），共用会把上一组的熔断带到下一组。
    """

    state_schema = ReviewState

    def __init__(self, compressor: Compressor | None = None) -> None:
        """初始化。

        Args:
            compressor: 压缩器；传 None 表示本组不做压缩（测试里常用）
        """
        super().__init__()
        self._compressor = compressor
        # 本组的起点。预算里的「时长」从这里算 —— 实例是每组新建的（见类 docstring），
        # 所以这个时间戳天然是「本组开始」
        self._started = time.monotonic()
        # 「拿不到用量」只警告一次：每轮都喊会把日志淹掉，而根因是同一个
        self._warned_missing_usage = False

    def before_model(
        self, state: ReviewState, runtime: Runtime[ReviewContext]
    ) -> dict[str, Any] | None:
        """压缩 → 追加快照 → 判定退出。

        Args:
            state: 当前状态
            runtime: 运行时。本钩子用不到，但**参数名不能改成 `_runtime`** ——
                手写类时框架靠签名自省决定注入哪些参数（`func_accepts`），
                认不出 `_runtime` 这个键就不注入，直接报
                「missing 1 required positional argument」。装饰器写法没这问题，
                它按位置转发。同理不能删掉这个参数，留着是对齐基类签名

        Returns:
            状态更新；需要退出时带 ``jump_to="end"``
        """
        updates: dict[str, Any] = {}

        # 告警与判定分开：告警每次都要有机会发生（它说的是「成本闸可能失效」），
        # 而判定只在真要停的时候才用得上
        self._warn_if_usage_missing(state)

        reset = end_done_reset(state)
        if reset:
            logger.debug("新审完 hunk，重置 end_done 纠正机会")
            updates.update(reset)

        # 退出判定要看重置之后的值
        end_done_count = 0 if reset else state["end_done_count"]
        if should_exit(state, end_done_count=end_done_count):
            logger.info("退出：end_done 或连续无产出已达阈值")
            return {**updates, "jump_to": "end"}

        # **语义退出优先于预算**：审完了就是审完了 —— 让预算把一个已完成的组标成
        # 「因预算停止」，是给用户一个假的缺口。预算管的是「还没完但不能再花了」
        verdict = self._budget_verdict(state)
        if verdict is not None:
            # 先写者胜：同一轮里若别的停手也写了原因，保留先写的那个
            return {
                **updates,
                "stop_reason": state["stop_reason"] or verdict,
                "jump_to": "end",
            }

        # **退出判定要在压缩之前**：这一轮就要收尾了还去压一遍历史，是白烧一次
        # 完整的摘要请求（实测踩到过 —— end_done 已经调完的那一轮仍然压了一次）
        if self._compressor is not None and self._compressor.should_compress(state["messages"]):
            # 压缩会自己追加一份最新的 <review_status>，所以这里不再重复注入快照
            updates["messages"] = self._compressor.compress(state["messages"], state["hunk_status"])
        else:
            calls = _last_tool_calls(state["messages"])
            if any(str(call.get("name", "")) == "report_hunks" for call in calls):
                updates["messages"] = [
                    HumanMessage(content=build_review_status(state["hunk_status"]))
                ]

        return updates or None

    def _warn_if_usage_missing(self, state: ReviewState) -> None:
        """拿不到用量时警告一次（每组一次）。

        成本闸完全依赖 `usage_metadata`，而它是提供商给的。缺了它，钱那一档就永远
        不响 —— 那是「不报错的错链路」，必须让它在日志里可见。警告放在判定**之外**：
        就算本组马上要因为别的原因退出，这件事也该被记下来。

        Args:
            state: 当前状态
        """
        if not state["usage"].missing or self._warned_missing_usage:
            return
        self._warned_missing_usage = True
        logger.warning(
            "有 %d 次调用拿不到用量，成本只能算下界；本组的花费闸可能失效，"
            "兜底的是时间闸与 recursion_limit",
            state["usage"].missing,
        )

    def _budget_verdict(self, state: ReviewState) -> str | None:
        """判预算：超了返回原因码，没超返回 None。

        **钱和时间都要看**：用量拿不到时钱这条会失效（那时只有时间能兜），
        时间则永远在涨。所以这不是「双保险」，是两条各自会单独失效的判据。

        Args:
            state: 当前状态

        Returns:
            `STOP_COST` / `STOP_TIME` / None
        """
        usage = state["usage"]
        elapsed = time.monotonic() - self._started
        verdict = exceeded(
            usage,
            elapsed,
            prices=settings.cost,
            max_cost=settings.review.max_group_cost,
            max_seconds=settings.review.max_group_seconds,
        )
        if verdict is None:
            return None
        logger.warning(
            "本组超预算（%s）：%s · 已用 %.0f 秒 · 预算 %.3f 元 / %.0f 秒",
            verdict,
            describe(usage, settings.cost),
            elapsed,
            settings.review.max_group_cost,
            settings.review.max_group_seconds,
        )
        return verdict


# `can_jump_to` 的读取路径是 `getattr(middleware.__class__, "before_model").__can_jump_to__`
# （见 `langchain/agents/factory.py` 的 `_get_can_jump_to`）。装饰器写法会把这个属性
# 挂到包装出来的函数上，手写类就得自己挂一次 —— 框架缺口，不是我们的取舍。
ReviewProgress.before_model.__can_jump_to__ = ["end"]  # type: ignore[attr-defined]


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
        "usage": add_usage(state["usage"], _round_usage(state["messages"])),
    }

    if calls:
        # 有工具调用 → 交给框架执行工具，不干预路由
        return updates

    if no_report in settings.review.no_progress_remind_rounds:
        logger.warning("连续 %d 轮无产出，注入提醒", no_report)
        content = NO_PROGRESS_REMINDER.format(rounds=no_report)
    else:
        logger.debug("模型未调用工具，注入推进提醒（连续无产出 %d 轮）", no_report)
        content = PLAIN_TEXT_REMINDER

    updates["messages"] = [HumanMessage(content=content)]
    # 与旧实现一致：提醒照样追加，到阈值才结束
    updates["jump_to"] = "end" if no_report >= settings.review.no_progress_exit_rounds else "model"
    return updates


# 工具内部抛异常时回给模型的话。**要让模型能自己做决定**：换参数 / 换工具 /
# 放弃这条线继续审别的，而不是照着同一套参数重试（重试必然再炸一次）
TOOL_ERROR_MESSAGE = (
    "错误：工具内部执行失败（{kind}：{detail}）。\n"
    "这不是你调用方式的问题，换一种方式获取这条信息，或先报告已有的结论继续审查 —— "
    "不要用同样的参数重试。"
)

# 连续失败到阈值时追加的一句。措辞要比上面更硬 —— 上一句是「建议」，这一句是
# 「你已经照着重试了几次，别再试了」
TOOL_FAILURE_ESCALATION = (
    "\n⚠️ 这个工具在本组已经**连续失败 {streak} 次**：不要再试它了，换成别的工具或直接报告已有结论。"
)


# 重复调用时回给模型的话。**要给出路**：它可能真的想再看一眼那段代码 ——
# 那就换个范围读（不同的参数不会被拦），或者用已有的结论继续
DUPLICATE_CALL_MESSAGE = (
    "这个调用你已经做过，结果就在上文（没有被压缩清理），不再重复执行。\n"
    "要重新看，请换一个行范围或换一个搜索词；否则请基于已有结果继续。"
)


# 模块级函数式中间件（同 review_pacing）。它是**无状态**的，所以共享一个实例没问题；
# 换参数值也一样：它只读不改。
@wrap_tool_call
def tool_call_dedup(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """参数完全相同的**只读**调用不重复执行。

    不是启发式：我们的三个读工具（`file_read` / `code_search` / `file_read_diff`）
    读的都是被审查版本的 blob 与内存里的 diff 索引，**一次审查内不变**，所以
    「同样的参数必然同样的结果」是可以证明的。重复执行纯属浪费 —— 既花一次输出，
    又把一份一模一样的内容再加进上下文。

    三条边界：

    - **只在白名单里生效**（`REFETCHABLE_TOOLS`）：`report_hunks` / `end_done`
      有副作用，重复调用是两回事
    - **结果已被压缩清掉时允许重读**：那时上文里只剩占位符，拦下来就等于让模型
      永远拿不回那段内容
    - 同一条 AI 消息里的**当前**调用不算「已经执行过」（按调用 id 排除），
      否则每个调用都会把自己当成重复

    Args:
        request: 工具调用请求
        handler: 真正的执行链

    Returns:
        工具结果；重复调用时返回一句说明，不执行
    """
    name = str(request.tool_call.get("name") or "")
    if name not in REFETCHABLE_TOOLS:
        return handler(request)

    args = request.tool_call.get("args") or {}
    messages = _messages_of(request.state)
    if not _executed_before(
        messages, name, args, exclude_id=str(request.tool_call.get("id") or "")
    ):
        return handler(request)

    logger.info("跳过重复调用：%s（同样的参数，结果还在上下文里）", name)
    return ToolMessage(
        content=DUPLICATE_CALL_MESSAGE,
        tool_call_id=str(request.tool_call.get("id") or ""),
    )


def _messages_of(state: Any) -> list[BaseMessage]:
    """从工具请求的状态里取消息列表。

    `ToolCallRequest.state` 的静态类型是 `dict | list | BaseModel`，我们的图状态
    是 dict。取不到时返回空列表 —— 宁可不去重，也不要在这里抛异常。

    Args:
        state: 图状态

    Returns:
        消息列表
    """
    if isinstance(state, dict):
        messages = state.get("messages")
        if isinstance(messages, list):
            return list(messages)
    return []


def _executed_before(
    messages: Sequence[BaseMessage], name: str, args: Mapping[str, Any], *, exclude_id: str
) -> bool:
    """历史里有没有一次「同样的调用，且结果还在」？

    Args:
        messages: 消息历史
        name: 工具名
        args: 本次参数
        exclude_id: 要排除的调用 id（当前这一次自己）

    Returns:
        有过时返回 True
    """
    # 结果还在的调用 id（占位符不算 —— 那表示内容已被清掉，允许重新取）
    alive = {
        str(message.tool_call_id)
        for message in messages
        if isinstance(message, ToolMessage) and not is_cleared(str(message.content))
    }
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "")
            if call_id == exclude_id or call_id not in alive:
                continue
            if str(call.get("name") or "") == name and (call.get("args") or {}) == args:
                return True
    return False


class ToolFailureBudget(AgentMiddleware[ReviewState, ReviewContext, Any]):
    """兜住工具异常，并给「失败」本身设一个预算。

    两件事做在一处，因为它们读的是同一份计数：

    1. **兜异常**（原来 `tool_error_guard` 的活，现在并入这里）：框架默认的兜底只认
       `ToolInvocationError`，其余原样 re-raise
       （`langgraph.prebuilt.tool_node._default_handle_tool_errors` 最后一行就是
       `raise e`）。于是工具里任何一个意外异常都会穿过图、穿过 `run_review_agent`，
       把**整条 review 带走**（后面所有组不跑，前面审出来的评论也一起没）。
       实测过：`code_search(file_pattern="a\0b")` 让 `subprocess` 抛 `ValueError`。
       兜底**不是静默**：ERROR + 完整堆栈照记。
    2. **失败预算**：连续失败到 `tool_fail_remind` 就在工具结果里追加一句硬提醒；
       累计到 `tool_fail_exit` 就在下一次 `before_model` 带缺口退出本组。

    为什么需要第 2 件：`no_report_rounds` 只管「这一轮有没有产出」，而一轮里
    **既有 report_hunks 又有失败的工具调用**时它会被清零 —— 于是模型可以一边慢
    慢产出、一边把预算烧在一直失败的调用上。那是「把预算浪费在无效调用上」唯一
    没被别处覆盖的形态（重复调用由 `tool_call_dedup` 管）。

    写成**类式**：计数是跨轮次状态，而模块级的 `@wrap_tool_call` 产出的是全局
    共享实例（框架缺口，见 CLAUDE.md），上一组的失败计数会漏到下一组。
    """

    state_schema = ReviewState

    def __init__(self) -> None:
        """初始化计数。"""
        super().__init__()
        self._streak = 0
        self._total = 0
        # 并行工具调用可能在多个线程里同时进来（同一个 super-step）——
        # 计数用锁护一下，免得漏记（漏记的后果是闸门晚响，方向是安全的）
        self._lock = threading.Lock()

    # @hook_config 是类式钩子声明跳转目标的方式（`__can_jump_to__` 由它挂上），
    # 与 ReviewProgress.before_model 那行手工挂载是同一件事
    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: ReviewState, runtime: Runtime[ReviewContext]
    ) -> dict[str, Any] | None:
        """累计失败到上限就带缺口退出。

        Args:
            state: 当前状态
            runtime: 运行时（不用）

        Returns:
            退出更新；没到上限时 None
        """
        with self._lock:
            total = self._total
        if total < settings.review.tool_fail_exit:
            return None
        logger.error(
            "工具在本组累计失败 %d 次（上限 %d），带缺口退出本组 —— 去看 ERROR 日志找根因",
            total,
            settings.review.tool_fail_exit,
        )
        return {
            "stop_reason": state["stop_reason"] or STOP_TOOL_FAILURES,
            "jump_to": "end",
        }

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """执行工具；抛异常时兜成一条结果并计数。

        Args:
            request: 工具调用请求
            handler: 真正的工具调用

        Returns:
            工具返回；抛异常时返回一条说明性的 ToolMessage
        """
        try:
            result = handler(request)
        # 这里是最后一道网，宽except 是有意的（见类 docstring）
        except Exception as exc:
            streak = self._record_failure()
            logger.exception(
                "工具 %s 抛异常（本组累计失败 %d 次，连续 %d 次），转成工具结果继续",
                request.tool_call["name"],
                self._total,
                streak,
            )
            text = TOOL_ERROR_MESSAGE.format(kind=type(exc).__name__, detail=exc)
            if streak >= settings.review.tool_fail_remind:
                text += TOOL_FAILURE_ESCALATION.format(streak=streak)
            return ToolMessage(
                content=text,
                tool_call_id=str(request.tool_call["id"]),
                # status="error"：工具结果自身标记失败，模型与日志都能一眼看出来
                status="error",
            )
        self._record_success()
        return result

    def _record_failure(self) -> int:
        """记一次失败。

        Returns:
            记完之后的连续失败次数
        """
        with self._lock:
            self._streak += 1
            self._total += 1
            return self._streak

    def _record_success(self) -> None:
        """记一次成功：**连续**计数清零，累计计数不动。

        连续与累计是两个问题：连续问「这个工具是不是坏了」，累计问「这一组烧了
        多少」。中途成功一次说明工具还能用，但那次失败确实已经花掉了。
        """
        with self._lock:
            self._streak = 0
