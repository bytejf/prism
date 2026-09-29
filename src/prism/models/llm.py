"""LLM 相关的数据模型。"""

import os
from dataclasses import dataclass

# 默认对接 DeepSeek；换成任意 OpenAI 兼容端点只需改 base_url
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# API key 所在的环境变量名
API_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass
class LLMConfig:
    """LLM 配置。

    Attributes:
        model: 模型名，如 "deepseek-chat"
        base_url: API base URL
        api_key: API key（从环境变量读）
        temperature: 温度，默认 0
        max_retries: 重试次数，默认 3
        timeout: 超时秒数，默认 60
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量构造配置。

        model 与 base_url 用默认值，可在此之后改字段覆盖。

        Returns:
            使用默认 model / base_url 的配置

        Raises:
            ValueError: 未设置 DEEPSEEK_API_KEY
        """
        api_key = os.getenv(API_KEY_ENV, "").strip()
        if not api_key:
            raise ValueError(f"未设置环境变量 {API_KEY_ENV}")
        return cls(api_key=api_key)
