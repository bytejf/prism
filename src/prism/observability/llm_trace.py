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
| 新增的 user / ai 消息 | `ReviewTrace.wrap_model_call`，逐条比对**已打过的消息编号** |
| 工具结果 | `ReviewTrace.wrap_tool_call`（顺带测工具执行耗时） |
| 中间件注入的 user（进度快照 / 提醒） | 也走 `wrap_model_call` —— 它们在下一次请求里才出现 |

**为什么按消息编号去重，而不是按「已打到第几条」的下标**：压缩
（`core/agent/compression.py`）会把整段历史换掉，换完的消息数可能**恰好相等
甚至更多**，下标比对判不出变化 —— 压缩后的摘要与进度快照会被静默跳过
（实测踩到过）。按编号去重还顺带解决「压缩后要不要把 A 段重打一遍」：
它编号没变，自然不重打。

轮次号同理，自己数（`_round_no`），不数「历史里有几条 AI 消息」——
压缩会把它们删掉，轮次号会从 12 掉回 1。

## 为什么是类式中间件

`ReviewTrace` 持有跨轮次的状态（已打过的消息编号、上一轮的输入 token、轮次号）。
用 `@wrap_model_call` 装饰模块级函数的话，**所有 agent 共用一个实例**，
状态会串到下一组审查去。所以改成类，由 `build_review_agent` 每组造一个。
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
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


def _wire_definition(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """取工具**真正发给模型**的那份定义。

    不能拿 `tool_call_schema.model_json_schema()` 来对账 —— 那是 Pydantic 转成
    OpenAI 格式**之前**的形态。框架的出站转换还会做三件事：

    - 剥掉所有 `title`
    - 剥掉 `parameters` 顶层的 `description`
    - 把嵌套模型的 `$defs` + `$ref` **内联展开**

    实测 `report_hunks` 两份差 2000 vs 1279 字节，`end_done` 差 236 vs 36 ——
    拿转换前的视图核「模型看到了什么」，结论会是错的。

    Args:
        tool: 工具，或已经是字典形式的工具定义

    Returns:
        `{"type": "function", "function": {...}}` 形式的定义
    """
    if isinstance(tool, dict):
        return tool
    return convert_to_openai_tool(tool)


def _describe_tool(tool: BaseTool | dict[str, Any]) -> str:
    """渲染一个工具的完整定义。

    **参数打原始 JSON 而不是转成表格**：`report_hunks` 的参数是嵌套结构
    （`reports` → `items` → `comments` → `items`），拍平成「名字 / 类型 / 必填」
    会把层次丢掉 —— 而模型看到的是完整 schema，日志要跟着它走。

    Args:
        tool: 工具，或已经是字典形式的工具定义

    Returns:
        多行文本
    """
    function = _wire_definition(tool).get("function", {})
    lines = [
        f"name: {function.get('name') or EMPTY}",
        f"description: {function.get('description') or EMPTY}",
    ]
    if parameters := function.get("parameters"):
        lines += ["", "parameters:", json.dumps(parameters, ensure_ascii=False, indent=2)]
    return "\n".join(lines)


class ReviewTrace(AgentMiddleware[Any, Any, Any]):
    """一次审查的轨迹中间件。

    状态是**按实例**的：`build_review_agent` 每组审查造一个，所以多个组之间
    不会串。
    """

    def __init__(self) -> None:
        """初始化。"""
        super().__init__()
        # 打过的消息编号。**不能用「已打到第几条」的下标** —— 压缩会把整段历史
        # 换掉，换完的消息数可能**恰好相等甚至更多**，下标比对判不出变化，
        # 压缩后的摘要与进度快照会被静默跳过（实测踩到过）
        self._printed: set[str] = set()
        # 首轮没有「上一轮」，环比增量从第二轮开始
        self._prev_input_tokens: int | None = None
        self._config_logged = False
        # 自己数轮次，不用「历史里有几条 AI 消息」—— 压缩会把 AI 消息删掉，
        # 那样轮次会从 12 掉回 1，日志里的轮次号就没法对账了
        self._round_no = 0

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

        round_no = self._round_no + 1
        for message in response.result:
            if isinstance(message, AIMessage):
                self._log_response(message, round_no=round_no, elapsed_ms=elapsed_ms)
                self._round_no = round_no
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
                self._round_no,
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
            self._log_tools(request)
            system_message = request.system_message
            if system_message is not None:
                logger.info("system\n%s", fields(("message", str(system_message.content))))

        for index, message in enumerate(request.messages):
            if message.id is not None and message.id in self._printed:
                continue
            if message.id is not None:
                self._printed.add(message.id)
            if isinstance(message, (AIMessage, ToolMessage)):
                # 这两类由各自的路径打：响应在 wrap_model_call 的返回侧、
                # 工具结果在 wrap_tool_call 里（那里才有耗时）。标记成已打过，
                # 免得它们在下一次请求里被当成「新增消息」再打一遍
                continue
            role = "user" if message.type == "human" else message.type
            logger.info(
                "%s · 轮次=%d%s\n%s",
                role,
                # 轮次 = 这条消息属于第几次请求。它一定是在**上一次响应之后**才出现的，
                # 所以是本轮（`_round_no` 是已完成的轮数，加一即当前轮）。
                # 不能数「它前面有几条 AI 消息」—— 压缩会把那些 AI 消息删掉，轮次号会跳
                self._round_no + 1,
                " · 本轮新增" if index == len(request.messages) - 1 else "",
                fields(("message", str(message.content))),
            )

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
                            f"tools={len(request.tools)} 个（定义见下一条）",
                        ]
                    ),
                )
            ),
        )

    def _log_tools(self, request: ModelRequest[Any]) -> None:
        """打一次工具定义。

        工具和模型配置一样是**一次审查只打一次**的静态量 —— 但它决定了模型
        「能用什么、怎么用」，调工具相关的问题（模型不调、调错参数、描述写得
        有歧义）第一眼要看的就是它。

        Args:
            request: 模型调用请求
        """
        if not request.tools:
            return
        logger.info(
            "工具定义 · %d 个\n%s",
            len(request.tools),
            fields(
                *(
                    (f"tool_{index}", _describe_tool(tool))
                    for index, tool in enumerate(request.tools, start=1)
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
