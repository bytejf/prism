"""LLM 客户端：把配置变成 LangChain ChatModel。

用 **`ChatDeepSeek`** 而不是通用的 `ChatOpenAI`：思考模式下 API 返回的
`reasoning_content`（思考正文）只有 provider 专用实现才提取，
`ChatOpenAI` 明确不提取 —— 见 `langchain_openai/chat_models/base.py` 的类文档：
「`reasoning_content`, `reasoning_details`) are **not** extracted」。

重试与超时交给 LangChain / OpenAI SDK 的内置实现。

**这里不再包一层 invoke** —— 调用与工具绑定都交给框架
（`create_agent` 内部会 bind_tools）。
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from prism.models.llm import LLMConfig

logger = logging.getLogger(__name__)


class ReasoningEchoChatDeepSeek(ChatDeepSeek):
    """把上一轮的思考内容回传给 API。

    补的是 `ChatDeepSeek` 的一个不对称：**入站解析、出站丢弃**。
    它把响应里的 `reasoning_content` 存进 `AIMessage.additional_kwargs`，
    但出站序列化走的是 `langchain_openai` 的 `_convert_message_to_dict`，
    那个函数构造 assistant 消息时只放 content / role / tool_calls / name，
    不认 `reasoning_content` 这个键，于是原样丢掉。

    DeepSeek 文档要求「思考模式 + tools」时把上一轮的 `reasoning_content`
    回传，否则可能 HTTP 400。

    Notes:
        **这是防御性的，不是在修一个正在发生的 bug**。实测 `deepseek-flash`
        目前并不强制：把带 tool_calls 的 assistant 消息原样回传（reasoning_content
        已被丢弃）不会报错。换到强制的模型时本类才真正起作用。

        只补**最后一条** assistant 消息，不补全部历史：文档说的是「所有历史轮次
        都要回传」，但本项目的循环每轮都重发全量历史，全补会让 prompt 随轮次
        线性膨胀（思考块不小）。既然「一条都不补」实测也能跑通，就取最小改动。
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构造请求体，并补回最后一条 assistant 的思考内容。

        Args:
            input_: 本轮输入（消息列表或可转成消息的对象）
            stop: 停止词
            **kwargs: 其余请求参数

        Returns:
            请求体字典
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        sent = payload.get("messages")
        messages = self._convert_input(input_).to_messages()
        # 条数对不上说明转换不是一一对应（例如走了 responses API 分支），
        # 那就什么都不动 —— 宁可不注入，也不能错位注入到别人的消息上
        if not isinstance(sent, list) or len(sent) != len(messages):
            logger.debug(
                "消息条数不匹配（%s vs %s），跳过思考内容回传",
                len(sent) if isinstance(sent, list) else type(sent).__name__,
                len(messages),
            )
            return payload

        for message, raw in zip(reversed(messages), reversed(sent), strict=True):
            if not isinstance(message, AIMessage) or not isinstance(raw, dict):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                raw.setdefault("reasoning_content", reasoning)
            break  # 只看最后一条 assistant

        return payload


class LLMClient:
    """LLM 客户端封装。

    重试次数与超时来自 LLMConfig，由 SDK 内部实现。
    """

    def __init__(self, config: LLMConfig) -> None:
        """初始化 LLM 客户端。

        Args:
            config: LLM 配置

        Raises:
            ValueError: 未提供 api_key
        """
        if not config.api_key:
            msg = "LLMConfig.api_key 为空，请通过环境变量 DEEPSEEK_API_KEY 提供"
            raise ValueError(msg)

        self._config = config
        self._model = self._build_model(config)
        logger.debug(
            "初始化 LLM 客户端：model=%s base_url=%s reasoning_effort=%s "
            "max_retries=%d timeout=%ds",
            config.model,
            config.base_url,
            config.reasoning_effort,
            config.max_retries,
            config.timeout,
        )

    @staticmethod
    def _build_model(config: LLMConfig) -> BaseChatModel:
        """按配置构造 ChatDeepSeek。

        Args:
            config: LLM 配置

        Returns:
            LangChain ChatModel 实例
        """
        return ReasoningEchoChatDeepSeek(
            model=config.model,
            base_url=config.base_url,
            api_key=SecretStr(config.api_key),
            reasoning_effort=config.reasoning_effort,
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    @property
    def model(self) -> BaseChatModel:
        """构造好的 LangChain ChatModel。

        调用与工具绑定都交给框架（`create_agent` 内部会 bind_tools），
        这里不再自己包一层 invoke。

        Returns:
            配置好的 ChatModel 实例
        """
        return self._model
