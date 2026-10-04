"""LLM 相关的数据模型。

**默认值不在这里**，在 `config/default.toml`；本文件只描述形状。
读取与填充见 `prism.config.settings`。
"""

from dataclasses import dataclass

# API key 所在的环境变量名。这不是「配置值」而是**约定的接口** ——
# 报错信息、文档、用户脚本都认这个名字，所以留在代码里，不进配置文件
API_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass
class LLMConfig:
    """LLM 配置。由 `prism.config.settings` 从 default.toml 填出来。

    Attributes:
        model: 模型名，如 "deepseek-flash"
        base_url: API base URL
        api_key: API key。**唯一从环境变量读的字段**，密钥不进配置文件
        reasoning_effort: 思考强度。DeepSeek 的思考模式**默认开启且默认档位是
            high**（实测：不传任何参数也会有 reasoning_tokens），所以显式指定，
            避免默认值悄悄变化。可选 minimal / low / medium / high / xhigh /
            max / ultra（会映射到实际档位）
        max_retries: 重试次数
        timeout: 超时秒数
        context_window: 上下文窗口（token），压缩据此判断何时触发
        temperature: 温度。**不是配置项** —— 思考模式下被 API 忽略
            （文档明确不支持），保留字段只为非思考模型
    """

    model: str
    base_url: str
    reasoning_effort: str
    max_retries: int
    timeout: int
    context_window: int
    api_key: str = ""
    temperature: float = 0.0
