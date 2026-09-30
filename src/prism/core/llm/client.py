"""LLM 客户端封装。

用 LangChain 的 ChatOpenAI 对接任意 OpenAI 兼容端点（V1 默认 DeepSeek），
通过 base_url 切换 provider，不耦合特定厂商。
重试与超时交给 LangChain / OpenAI SDK 的内置实现。
"""

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from prism.models.llm import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用失败。"""


class LLMClient:
    """LLM 客户端封装。

    重试次数与超时来自 LLMConfig，由 ChatOpenAI 内部实现。
    日志：调用开始 / 完成打 DEBUG，失败打 ERROR。
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
            "初始化 LLM 客户端：model=%s base_url=%s max_retries=%d timeout=%ds",
            config.model,
            config.base_url,
            config.max_retries,
            config.timeout,
        )

    @staticmethod
    def _build_model(config: LLMConfig) -> BaseChatModel:
        """按配置构造 ChatOpenAI。

        Args:
            config: LLM 配置

        Returns:
            LangChain ChatModel 实例
        """
        return ChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=SecretStr(config.api_key),
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: list[BaseTool] | None = None,
    ) -> AIMessage:
        """调用 LLM。

        Args:
            messages: 消息列表
            tools: 可用的工具列表（可选）

        Returns:
            LLM 响应（可能含 tool_calls）

        Raises:
            LLMError: 调用失败，或返回的不是 AIMessage
        """
        logger.debug("LLM 调用开始：%d 条消息，%d 个工具", len(messages), len(tools or []))

        runnable = self._model.bind_tools(tools) if tools else self._model
        try:
            response = runnable.invoke(messages)
        except Exception as exc:
            logger.error("LLM 调用失败：%s", exc)
            raise LLMError(f"LLM 调用失败：{exc}") from exc

        if not isinstance(response, AIMessage):
            msg = f"LLM 返回了非 AIMessage：{type(response).__name__}"
            logger.error(msg)
            raise LLMError(msg)

        logger.debug(
            "LLM 调用完成：内容 %d 字符，tool_calls %d 个，token 用量 %s",
            len(message_text(response)),
            len(response.tool_calls),
            response.usage_metadata,
        )
        return response


def message_text(message: BaseMessage) -> str:
    """把消息内容归一化成纯文本。

    LangChain v1 的 content 可能是字符串，也可能是内容块列表。

    Args:
        message: 任意消息

    Returns:
        纯文本内容；提取不到时返回空字符串
    """
    content = message.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
