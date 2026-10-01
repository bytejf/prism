"""LLM 客户端：把配置变成 LangChain ChatModel。

用 ChatOpenAI 对接任意 OpenAI 兼容端点（V1 默认 DeepSeek），
通过 base_url 切换 provider，不耦合特定厂商。
重试与超时交给 LangChain / OpenAI SDK 的内置实现。

**这里不再包一层 invoke** —— 调用与工具绑定都交给框架
（`create_agent` 内部会 bind_tools）。
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from prism.models.llm import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端封装。

    重试次数与超时来自 LLMConfig，由 ChatOpenAI 内部实现。
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

    @property
    def model(self) -> BaseChatModel:
        """构造好的 LangChain ChatModel。

        调用与工具绑定都交给框架（`create_agent` 内部会 bind_tools），
        这里不再自己包一层 invoke。

        Returns:
            ChatOpenAI 实例
        """
        return self._model
