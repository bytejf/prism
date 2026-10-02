"""LLM 上下文与响应的可观测性。

`create_agent` 造出来的图是个黑盒：提示词写错了、历史膨胀了、缓存没命中、
思考档位开得过高，从外面全看不见。这里把每次模型调用的上下文与响应打出来。

## 一条消息 = 一条日志

    14:23:45.433 INFO [observability/llm_trace.py:184] ai · 轮次=1 · 耗时 8421ms
    <content>
    （空）

    <reasoning_content>
    diff 里 Label 的索引看着不对，先把整个文件读出来确认边界。

    <tool_calls>
    file_read(path='format.go', start_line=1, end_line=30)

    <usage>
    in 1701 / out 92 / total 1793 / reasoning 41
    finish_reason tool_calls

标签可以扩展 —— 加字段就往 `fields()` 里多塞一项，不用改渲染。

## 只打「变化的部分」

不做「每轮重打全量上下文」：30 轮循环里那是 O(n²) 的日志量。
所以按**产生者**打，而不是按请求打：

| 消息 | 谁打 |
|---|---|
| system 提示词、模型配置 | `ReviewTrace.wrap_model_call`，首次调用时 |
| 新增的 user / ai 消息 | `ReviewTrace.wrap_model_call`，逐条比对已打过的下标 |
| 工具结果 | `ReviewTrace.wrap_tool_call`（顺带测工具执行耗时） |
| 中间件注入的 user（进度快照 / 提醒） | 也走 `wrap_model_call` —— 它们在下一次请求里才出现 |

## 为什么是类式中间件

`ReviewTrace` 持有跨轮次的状态（已打过的消息数、上一轮的输入 token）。
用 `@wrap_model_call` 装饰模块级函数的话，**所有 agent 共用一个实例**，
状态会串到下一组审查去。所以改成类，由 `build_review_agent` 每组造一个。
"""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# 字段为空时的占位。省略字段就看不出「确实没有」，统一打占位更清楚
EMPTY = "（空）"


def fields(*items: tuple[str, str]) -> str:
    """把若干「标签 + 内容」拼成正文块。

    字段为空也照样打标签 —— 统一比省略好读。

    Args:
        *items: ``(标签名, 内容)`` 序列

    Returns:
        形如 ``<content>\\nA\\n\\n<tool_calls>\\nC`` 的正文
    """
    return "\n\n".join(f"<{tag}>\n{content or EMPTY}" for tag, content in items)


def _format_tool_calls(message: AIMessage) -> str:
    """把一次响应里的工具调用渲染成 ``名字(参数)`` 逐行。

    用 Python 关键字风格而不是原始 JSON：模型发过来的 JSON 在日志里带一堆转义，
    没法读。`!r` 保证值仍可原样还原。

    Args:
        message: AI 消息

    Returns:
        每个调用一行；没有调用时返回空串
    """
    lines = []
    for call in message.tool_calls:
        args = ", ".join(f"{key}={value!r}" for key, value in call.get("args", {}).items())
        lines.append(f"{call.get('name', '?')}({args})")
    return "\n".join(lines)


def _format_usage(message: AIMessage, prev_input_tokens: int | None) -> str:
    """渲染 token 消耗。

    输入 token 带**环比增量** —— 历史膨胀是成本的大头，单看绝对值看不出来。

    Args:
        message: AI 消息
        prev_input_tokens: 上一轮的输入 token；首轮为 None

    Returns:
        两行文本：token 明细 + ``finish_reason``
    """
    usage = message.usage_metadata
    if not usage:
        return ""

    input_tokens = usage.get("input_tokens", 0)
    delta = f" ({input_tokens - prev_input_tokens:+d})" if prev_input_tokens is not None else ""
    parts = [
        f"in {input_tokens}{delta}",
        f"out {usage.get('output_tokens', 0)}",
        f"total {usage.get('total_tokens', 0)}",
    ]
    if cache_read := (usage.get("input_token_details") or {}).get("cache_read"):
        parts.append(f"cache_read {cache_read}")
    if reasoning := (usage.get("output_token_details") or {}).get("reasoning"):
        parts.append(f"reasoning {reasoning}")

    lines = [" / ".join(parts)]
    finish = message.response_metadata.get("finish_reason")
    if finish:
        lines.append(f"finish_reason {finish}")
    return "\n".join(lines)


def _ai_count(messages: Sequence[BaseMessage]) -> int:
    """数历史里有几条 AI 消息 —— 它就是已完成的轮次数。

    Args:
        messages: 消息历史

    Returns:
        AI 消息条数
    """
    return sum(1 for message in messages if isinstance(message, AIMessage))


class ReviewTrace(AgentMiddleware[Any, Any, Any]):
    """一次审查的轨迹中间件。

    状态是**按实例**的：`build_review_agent` 每组审查造一个，所以多个组之间
    不会串。
    """

    def __init__(self) -> None:
        """初始化。"""
        super().__init__()
        self._logged = 0
        # 首轮没有「上一轮」，环比增量从第二轮开始
        self._prev_input_tokens: int | None = None
        self._config_logged = False

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """打本轮新增的上下文，然后调用模型并打响应。

        Args:
            request: 模型调用请求
            handler: 真正的模型调用

        Returns:
            模型调用响应（原样透传）
        """
        self._log_request(request)

        started = time.monotonic()
        response = handler(request)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        round_no = _ai_count(request.messages) + 1
        for message in response.result:
            if isinstance(message, AIMessage):
                self._log_response(message, round_no=round_no, elapsed_ms=elapsed_ms)
                usage = message.usage_metadata
                # 记下来给下一轮算环比增量
                self._prev_input_tokens = usage.get("input_tokens") if usage else None
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """执行工具，并把结果与其耗时打出来。

        Args:
            request: 工具调用请求
            handler: 真正的工具调用

        Returns:
            工具的返回（原样透传）
        """
        started = time.monotonic()
        result = handler(request)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        for message in _tool_messages(result):
            logger.info(
                "tool · 轮次=%d · %s · 执行 %dms · 结果 %d 字符\n%s",
                _ai_count(request.state["messages"]),
                # name 是 ToolCall 的必填键，直接取下标
                request.tool_call["name"],
                elapsed_ms,
                len(str(message.content)),
                fields(("message", str(message.content))),
            )
        return result

    def _log_request(self, request: ModelRequest[Any]) -> None:
        """打本轮相对上次新增的上下文。

        **只打请求侧的消息**（user / 注入的提醒）。AI 响应由 `wrap_model_call`
        的响应路径打、工具结果由 `wrap_tool_call` 打（那里才有耗时）——
        不跳过的话它们会在下一次请求里被当成「新增消息」再打一遍。

        Args:
            request: 模型调用请求
        """
        if not self._config_logged:
            self._config_logged = True
            self._log_config(request)
            system_message = request.system_message
            if system_message is not None:
                logger.info("system\n%s", fields(("message", str(system_message.content))))

        for index, message in enumerate(request.messages[self._logged :], start=self._logged):
            if isinstance(message, (AIMessage, ToolMessage)):
                continue
            is_new = index == len(request.messages) - 1
            role = "user" if message.type == "human" else message.type
            # 轮次 = 这条消息属于第几次请求。数**它前面**的 AI 消息，
            # 不包含它自己 —— 否则首轮 user 会算成第 0 轮
            logger.info(
                "%s · 轮次=%d%s\n%s",
                role,
                _ai_count(request.messages[:index]) + 1,
                " · 本轮新增" if is_new else "",
                fields(("message", str(message.content))),
            )
        self._logged = len(request.messages)

    def _log_config(self, request: ModelRequest[Any]) -> None:
        """打一次模型配置。

        一次审查只打一次，成本极低，但排查「是不是档位开高了」时最先要看的就是它。

        Args:
            request: 模型调用请求
        """
        model = request.model
        logger.info(
            "模型配置\n%s",
            fields(
                (
                    "config",
                    "\n".join(
                        [
                            f"model={getattr(model, 'model_name', type(model).__name__)}"
                            f" · effort={getattr(model, 'reasoning_effort', '?')}"
                            f" · temperature={getattr(model, 'temperature', '?')}",
                            f"max_retries={getattr(model, 'max_retries', '?')}"
                            f" · timeout={getattr(model, 'request_timeout', '?')}",
                            f"tools({len(request.tools)})="
                            + ", ".join(str(getattr(tool, "name", tool)) for tool in request.tools),
                        ]
                    ),
                )
            ),
        )

    def _log_response(self, message: AIMessage, *, round_no: int, elapsed_ms: int) -> None:
        """打一次模型响应。

        Args:
            message: 模型返回的 AI 消息
            round_no: 轮次
            elapsed_ms: 本次调用耗时
        """
        reasoning = message.additional_kwargs.get("reasoning_content")
        logger.info(
            "ai · 轮次=%d · 耗时 %dms\n%s",
            round_no,
            elapsed_ms,
            fields(
                ("content", str(message.content)),
                ("reasoning_content", reasoning if isinstance(reasoning, str) else ""),
                ("tool_calls", _format_tool_calls(message)),
                ("usage", _format_usage(message, self._prev_input_tokens)),
            ),
        )


def _tool_messages(result: ToolMessage | Command[Any]) -> list[ToolMessage]:
    """从工具返回里取出要打的消息。

    写状态的工具返回 `Command`，ToolMessage 藏在 `update["messages"]` 里。

    Args:
        result: 工具的原始返回

    Returns:
        ToolMessage 列表；取不到时为空
    """
    if isinstance(result, ToolMessage):
        return [result]
    update = getattr(result, "update", None) or {}
    messages = update.get("messages") if isinstance(update, dict) else None
    return [item for item in messages or [] if isinstance(item, ToolMessage)]
