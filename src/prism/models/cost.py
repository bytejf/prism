"""成本相关的纯数据模型。

只描述数据结构，不含行为（换算与判定在 `core/agent/budget.py`）。
"""

from dataclasses import dataclass

# 价格的单位是「元 / 百万 token」。换 provider 或调价时改配置，不改代码 ——
# 价格是最典型的旋钮：它一定会变，而且配错了只是预算松紧不对，看得见。
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class CostConfig:
    """每百万 token 的价格（元）。

    三档价差很大，这正是「轮数」当不了成本代理的原因：缓存命中的输入比未命中
    便宜一个数量级，输出又比未命中贵几倍。

    Attributes:
        cache_hit: 输入里命中前缀缓存的部分
        cache_miss: 输入里未命中的部分
        output: 输出（含思考 token —— API 把它算在输出里）
    """

    cache_hit: float
    cache_miss: float
    output: float


@dataclass(frozen=True)
class UsageTotals:
    """累计用量（token）。

    Attributes:
        calls: 计入的模型调用次数
        cache_hit: 命中缓存的输入 token
        cache_miss: 未命中的输入 token
        output: 输出 token（含思考）
        missing: **拿不到用量**的调用次数。非零意味着成本只能算个下界，
            必须让它可见 —— 否则「成本闸」会因为一份缺失的用量而静默失效
    """

    calls: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    output: int = 0
    missing: int = 0

    @property
    def input_tokens(self) -> int:
        """输入总 token（命中 + 未命中）。"""
        return self.cache_hit + self.cache_miss
