"""LLM 相关的数据模型。"""

import os
from dataclasses import dataclass

# 固定对接 DeepSeek（OpenAI 兼容端点）。V1 不做配置文件，
# 换 provider / 换模型直接改这两行。
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# API key 所在的环境变量名
API_KEY_ENV = "DEEPSEEK_API_KEY"

# 模型上下文窗口（token）。压缩的触发阈值由它算出来，所以**不硬编码在压缩逻辑里** ——
# 换模型、DeepSeek 升级窗口时只改这一处。实测当前模型是 1M，不是 128K。
DEFAULT_CONTEXT_WINDOW = 1_000_000

# 思考强度默认值。API 的默认档位是 high，显式降到 low：
# 审查任务里模型绝大多数轮次只是「读文件 → 报告结论」，不需要深推理，
# 而思考 token 计费且会让 prompt 一起变长（实测 thinking 开/关：38 vs 12 tokens）
DEFAULT_REASONING_EFFORT = "low"


@dataclass
class LLMConfig:
    """LLM 配置。

    Attributes:
        model: 模型名，如 "deepseek-flash"
        base_url: API base URL
        api_key: API key（从环境变量读）
        reasoning_effort: 思考强度。DeepSeek 的思考模式**默认开启且默认档位是
            high**（实测：不传任何参数也会有 reasoning_tokens），所以这里显式
            指定，避免默认值悄悄变化。可选 minimal / low / medium / high /
            xhigh / max / ultra（会映射到实际档位）
        temperature: 温度，默认 0。**思考模式下被 API 忽略**（文档明确不支持），
            保留字段只为非思考模型
        max_retries: 重试次数，默认 3
        timeout: 超时秒数，默认 60
        context_window: 上下文窗口（token），压缩据此判断何时触发
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 60
    context_window: int = DEFAULT_CONTEXT_WINDOW

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
