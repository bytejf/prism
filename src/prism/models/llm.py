"""LLM 相关的数据模型。

**默认值不在这里**，在 `config/default.toml`；本文件只描述形状。
读取与填充见 `prism.config.settings`。
"""

from dataclasses import dataclass

# API key 所在的环境变量名。这不是「配置值」而是**约定的接口** ——
# 报错信息、文档、用户脚本都认这个名字，所以留在代码里，不进配置文件
API_KEY_ENV = "DEEPSEEK_API_KEY"

# `reasoning_effort` 的合法取值。同样是**接口契约**而非配置旋钮 ——
# 它由 API 定义，写进文件等于给人一个配错的机会，而这个错的后果是**静默**的：
# 实测往 API 传一个它不认的档位不报错，请求照发，只是安静地按默认档跑。
#
# 官方定义四个 —— `none` 关闭思考；`low` / `high` / `max` 开启思考并控制强度
# （不传时默认 `high`）。后三个是官方文档写明的**兼容别名**（`minimal`→`low`，
# `medium`/`xhigh`→`high`），一并收下：校验不该比 API 自己更窄。
REASONING_EFFORTS = ("none", "low", "high", "max", "minimal", "medium", "xhigh")


@dataclass
class LLMConfig:
    """LLM 配置。由 `prism.config.settings` 从 default.toml 填出来。

    Attributes:
        model: 模型名，如 "deepseek-flash"
        base_url: API base URL
        api_key: API key。**唯一从环境变量读的字段**，密钥不进配置文件
        reasoning_effort: 思考强度。DeepSeek 的思考模式**默认开启且默认档位是
            high**（实测：不传任何参数也会有 reasoning_tokens），所以显式指定，
            避免默认值悄悄变化。取值见 `REASONING_EFFORTS`；`none` 关闭思考，
            其余档位只控制强度、不开关
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
