"""上下文压缩的配置。

**默认值不在这里**，在 `config/default.toml`；本文件只描述形状与那点派生逻辑。
阈值是**固定的粗预留**，不做精细控制。设计文档 §4.2 要求累积实测数据后按分位数
校准（触发剩余取 P10），校准结果回填到 `default.toml`。
"""

from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """压缩的阈值与容错参数。由 `prism.config.settings` 从 default.toml 填出来。

    1M 窗口下 ``reserve_tokens = 20_000`` 只占 2%，压缩因此几乎不触发 ——
    **这是有意的**：V1 的压缩是保险机制，不是常规操作。

    Attributes:
        context_window: 模型窗口（token）。来自 `[llm].context_window`，
            不在 `[compression]` 里重复写
        reserve_tokens: 固定预留。触发判定与压缩请求的预算共用它，含义见
            `usable_tokens`。它是**估算出来的粗值**，里面已经含了压缩提示词
            （system + user）的开销和摘要输出 —— 摘要长度是唯一算不准的量，
            靠这个余量兜住
        max_compressions: 单组最多压缩几次（§3.3）。第一次是常规操作，
            第二次是「历史反复膨胀」的兜底，第三次的信息损失不可接受
        max_consecutive_failures: 连续失败多少次熔断（§2.11）。熔断后本组不再
            尝试压缩，直接走截断
    """

    context_window: int
    reserve_tokens: int
    max_compressions: int
    max_consecutive_failures: int

    @property
    def usable_tokens(self) -> int:
        """扣掉固定预留后「可用」的 token 数 —— 两处共用同一个数。

        - **触发**：已用达到它就压缩。等价于「剩余 < 预留」，但计数只有一个口径
        - **压缩请求的硬约束**：压缩 sys + 压缩 user + 历史 ≤ 它

        两处是同一个数，因为说的是同一件事：压缩请求装的就是触发那一刻的上下文，
        而清理只会让它更小。
        """
        return self.context_window - self.reserve_tokens
