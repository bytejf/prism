"""每轮模型调用的可观测性中间件。

`create_agent` 造出来的图是个黑盒：提示词写错了、历史膨胀了、缓存没命中，
从外面看不出来。这个中间件挂在模型调用外面，把**本轮**的请求与响应原样打出来。

打什么、为什么这么打：

- **system 段**：每轮长度都一样，全量重打是浪费，但有内容要校验，所以打全文
- **历史消息**：打「条数 + 每条 role/长度」，一眼看出历史有没有异常膨胀
- **最后一条消息全文**：它是本轮真正新增的那条（工具结果、进度快照、提醒）
- **响应**：正文、工具调用、思考正文、token（输入/输出/缓存/思考）

轮次编号由「历史里已有的 AIMessage 条数 + 1」推出来，不从 agent 状态里读 ——
这样本模块不需要 import 任何 prism 的东西，observability 不会反向依赖业务层。

日志走 stderr（`observability/logging.py` 统一配置），stdout 留给审查结果。
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)

# 单段正文的打印上限（字符）。prompt 校验需要看到内容，但整份 diff 可能上万字符，
# 不设上限会把日志冲垮。要完整内容就调到足够大，比如 PRISM_LOG_LEVEL=DEBUG 配合调大本值。
MAX_BODY_CHARS = 8000

# 消息摘要行里正文长度的列宽
_WIDTH = 7


def _clip(text: str) -> str:
    """截断过长正文，并标注原始长度。

    Args:
        text: 原始文本

    Returns:
        截断后的文本
    """
    if len(text) <= MAX_BODY_CHARS:
        return text
    return f"{text[:MAX_BODY_CHARS]}…（共 {len(text)} 字符，已截断）"


def _summarize(message: BaseMessage) -> str:
    """生成一条消息的摘要行。

    Args:
        message: 消息

    Returns:
        形如 ``ai       210  tool_calls=report_hunks`` 的一行
    """
    parts = [f"{message.type:<8}", f"{len(str(message.content)):>{_WIDTH}}"]
    if isinstance(message, AIMessage) and message.tool_calls:
        names = ",".join(str(call.get("name", "?")) for call in message.tool_calls)
        parts.append(f"tool_calls={names}")
    return "  ".join(parts)


def _describe_request(request: ModelRequest[Any]) -> str:
    """描述本轮发给模型的请求。

    Args:
        request: 模型调用请求

    Returns:
        多行文本
    """
    lines = [f"model   : {getattr(request.model, 'model_name', type(request.model).__name__)}"]

    system_message = request.system_message
    if system_message is not None:
        lines.append(f"system  : {len(str(system_message.content))} 字符")

    lines.append(f"messages: {len(request.messages)} 条")
    lines.extend(
        f"  [{index}] {_summarize(message)}" for index, message in enumerate(request.messages)
    )

    if request.messages:
        lines.append("--- 最后一条（本轮新增）---")
        lines.append(_clip(str(request.messages[-1].content)))

    if system_message is not None:
        lines.append("--- system ---")
        lines.append(_clip(str(system_message.content)))

    return "\n".join(lines)


def _format_usage(message: AIMessage) -> str:
    """格式化 token 消耗。

    Args:
        message: 模型返回的 AI 消息（只有它带 usage_metadata）

    Returns:
        形如 ``in=8421 out=312 total=8733 | cache_read=7936 | reasoning=180`` 的文本
    """
    usage = message.usage_metadata
    if not usage:
        return "（无）"

    parts = [
        f"in={usage.get('input_tokens', 0)}",
        f"out={usage.get('output_tokens', 0)}",
        f"total={usage.get('total_tokens', 0)}",
    ]
    if cache_read := (usage.get("input_token_details") or {}).get("cache_read"):
        parts.append(f"cache_read={cache_read}")
    if reasoning := (usage.get("output_token_details") or {}).get("reasoning"):
        parts.append(f"reasoning={reasoning}")
    return " | ".join(parts)


def _describe_response(response: ModelResponse[Any]) -> str:
    """描述模型本轮返回的内容。

    Args:
        response: 模型调用响应

    Returns:
        多行文本
    """
    if not response.result:
        return "（无消息）"

    message = response.result[-1]
    content = str(message.content)
    lines = [f"content : {_clip(content) if content else '（空）'}"]

    # usage / 工具调用 / 思考内容都只在 AI 消息上，非 AI 时只打正文
    if isinstance(message, AIMessage):
        if message.tool_calls:
            lines.append("tools   :")
            for call in message.tool_calls:
                args = json.dumps(call.get("args", {}), ensure_ascii=False)
                lines.append(f"  {call.get('name', '?')}({_clip(args)})")

        reasoning = message.additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            lines.append(f"思考    : {_clip(reasoning)}")

        lines.append(f"usage   : {_format_usage(message)}")

    return "\n".join(lines)


@wrap_model_call(name="LLMTrace")
def trace_model_call(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
) -> ModelResponse[Any]:
    """打印本轮模型调用的请求与响应。

    轮次 = 历史里已有的 AIMessage 条数 + 1（第 n 次调用前恰好有 n-1 条 AI 消息）。

    Args:
        request: 模型调用请求
        handler: 真正的模型调用

    Returns:
        模型调用响应（原样透传）
    """
    round_no = 1 + sum(1 for message in request.messages if isinstance(message, AIMessage))

    logger.info("──── LLM 请求 #%d ────\n%s", round_no, _describe_request(request))
    response = handler(request)
    logger.info("──── LLM 响应 #%d ────\n%s", round_no, _describe_response(response))

    return response
