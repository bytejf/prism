"""审查预算：把 token 用量折算成钱，并按预算判定该不该停。

为什么约束落在「钱 + 时间」而不是轮数：轮数是**代理指标**，同一轮的成本能差几十倍
（实测：一轮 `out 23`（调 end_done），另一轮 `out 1555`（报 7 条评论 + 366 个思考
token））。实测一次完整审查的成本构成是 **输出 54% / 未命中输入 35% / 命中输入
11%** —— 三者单价不同，轮数一个都数不出来。轮数只保留「不依赖任何数据」的那点
价值（防图失控），见 `agent.py` 的 `recursion_limit`。

## 两处记账方式不一样，各有原因

- **审查循环**：按轮**增量累加**进状态。不能扫最终消息 —— 压缩会把历史消息从列表里
  删掉，账会漏
- **反思**：直接扫终态消息。它不压缩，一张图最多 3 次调用，扫一遍最省事

两处产出的都是 `UsageTotals`，按同一张价格表折算，全局预算把两边加在一起。
"""

import logging
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from prism.models.cost import PER_MILLION, CostConfig, UsageTotals

logger = logging.getLogger(__name__)

# 停下来的原因码（进 `ReviewState.stop_reason`，也是给人看的）：钱 / 时间 / 工具失败
STOP_COST = "cost"
STOP_TIME = "time"
STOP_TOOL_FAILURES = "tool_failures"


def usage_of(messages: Sequence[BaseMessage]) -> UsageTotals:
    """扫一遍消息，累加所有 AI 消息的用量。

    只适合**没有压缩**的场景（反思）。审查循环用 `add_usage` 增量累加 ——
    压缩会把消息删掉，扫出来的账是漏的。

    Args:
        messages: 消息列表

    Returns:
        累计用量
    """
    total = UsageTotals()
    for message in messages:
        if isinstance(message, AIMessage):
            total = add_usage(total, usage_of_message(message))
    return total


def usage_of_message(message: AIMessage) -> UsageTotals:
    """取一条 AI 消息的用量。

    `usage_metadata` 缺失时返回 `missing=1` 而不是零 —— 「拿不到用量」与「用量是零」
    必须能分开，否则成本闸会静默失效。

    Args:
        message: AI 消息

    Returns:
        这一条消息的用量
    """
    usage = message.usage_metadata
    if not usage:
        return UsageTotals(missing=1)
    details = usage.get("input_token_details") or {}
    cache_hit = int(details.get("cache_read") or 0)
    input_tokens = int(usage.get("input_tokens") or 0)
    # 框架给的 input_tokens 是**总数**（含缓存命中），命中部分是它的子集
    cache_miss = max(input_tokens - cache_hit, 0)
    return UsageTotals(
        calls=1,
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        output=int(usage.get("output_tokens") or 0),
    )


def add_usage(base: UsageTotals, delta: UsageTotals) -> UsageTotals:
    """累加两份用量。

    Args:
        base: 已有用量
        delta: 新增用量

    Returns:
        新的用量对象
    """
    return UsageTotals(
        calls=base.calls + delta.calls,
        cache_hit=base.cache_hit + delta.cache_hit,
        cache_miss=base.cache_miss + delta.cache_miss,
        output=base.output + delta.output,
        missing=base.missing + delta.missing,
    )


def cost_yuan(usage: UsageTotals, prices: CostConfig) -> float:
    """把用量折算成钱（元）。

    Args:
        usage: 累计用量
        prices: 每百万 token 的价格

    Returns:
        金额（元）
    """
    total = (
        usage.cache_hit * prices.cache_hit
        + usage.cache_miss * prices.cache_miss
        + usage.output * prices.output
    )
    return total / PER_MILLION


def exceeded(
    usage: UsageTotals,
    elapsed_seconds: float,
    *,
    prices: CostConfig,
    max_cost: float,
    max_seconds: float,
) -> str | None:
    """判定是否超预算。

    **钱和时间都要判**：钱会因用量缺失而失效（那时只有时间闸能兜），时间则永远在涨，
    两个判据谁先到算谁。价格表由调用方传进来 —— 这个函数是纯的，测试不用碰配置。

    Args:
        usage: 累计用量
        elapsed_seconds: 已用时间（秒）
        prices: 价格表
        max_cost: 花费上限（元）。<= 0 表示不限制
        max_seconds: 时长上限（秒）。<= 0 表示不限制

    Returns:
        `STOP_COST` / `STOP_TIME`；都没超时返回 None
    """
    if max_cost > 0 and cost_yuan(usage, prices) >= max_cost:
        return STOP_COST
    if max_seconds > 0 and elapsed_seconds >= max_seconds:
        return STOP_TIME
    return None


def consumed_ratio(
    usage: UsageTotals,
    elapsed_seconds: float,
    *,
    prices: CostConfig,
    max_cost: float,
    max_seconds: float,
) -> float:
    """已用掉单组预算的比例（钱与时间取**较大**的那个）。

    收尾闸要问的不是「超没超」，是「还剩多少」—— 剩下的得够交卷。
    与 `exceeded` 同一套口径，只是一个答「越界没有」、一个答「走了多远」。

    `<= 0` 的那一路表示不限制，**不参与取大**（否则恒为 0，闸永远不响）。

    Args:
        usage: 累计用量
        elapsed_seconds: 已用时间（秒）
        prices: 价格表
        max_cost: 花费上限（元）。<= 0 表示不限制
        max_seconds: 时长上限（秒）。<= 0 表示不限制

    Returns:
        0.0~1.0+ 的比例；两路都不限制时返回 0.0
    """
    ratios: list[float] = []
    if max_cost > 0:
        ratios.append(cost_yuan(usage, prices) / max_cost)
    if max_seconds > 0:
        ratios.append(elapsed_seconds / max_seconds)
    return max(ratios) if ratios else 0.0


def describe(usage: UsageTotals, prices: CostConfig) -> str:
    """把用量渲染成一句人看的说明（进日志与终端输出）。

    Args:
        usage: 累计用量
        prices: 价格表

    Returns:
        形如 ``12 次调用 · 输入 146663（命中 118016）· 输出 14360 · 约 0.35 元``
    """
    parts = [
        f"{usage.calls} 次调用",
        f"输入 {usage.input_tokens}（命中 {usage.cache_hit}）",
        f"输出 {usage.output}",
        f"约 {cost_yuan(usage, prices):.3f} 元",
    ]
    if usage.missing:
        parts.append(f"⚠️ {usage.missing} 次调用拿不到用量")
    return " · ".join(parts)
