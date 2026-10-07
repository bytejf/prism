"""预算与成本折算的单元测试。

分成两层：**纯函数**（折算、累加、判定 —— 不碰配置也不碰 LLM）与**接线**
（状态里怎么记、闸门怎么生效，后者在 `test_main_loop.py` 与 `test_cli.py` 里）。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from prism.core.agent.budget import (
    STOP_COST,
    STOP_TIME,
    add_usage,
    cost_yuan,
    describe,
    exceeded,
    usage_of,
    usage_of_message,
)
from prism.models.cost import CostConfig, UsageTotals

PRICES = CostConfig(cache_hit=0.5, cache_miss=4.0, output=12.0)


def message(
    *, input_tokens: int = 1000, cache_read: int = 800, output: int = 100, usage: bool = True
) -> AIMessage:
    """造一条带用量的 AI 消息。

    Args:
        input_tokens: 输入总 token（含缓存命中）
        cache_read: 其中的缓存命中的部分
        output: 输出 token
        usage: 是否带用量元数据

    Returns:
        AIMessage
    """
    if not usage:
        return AIMessage(content="", usage_metadata=None)
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output,
            "total_tokens": input_tokens + output,
            "input_token_details": {"cache_read": cache_read},
        },
    )


def test_cache_hit_is_a_subset_of_input() -> None:
    """命中是输入总数的**子集**，未命中要减出来。

    框架给的 `input_tokens` 是总数（含命中）。直接拿它当「未命中」算，会把最贵的
    那一档价格乘上一个被高估的量 —— 实测成本里的 35% 是这一档。
    """
    usage = usage_of_message(message(input_tokens=1000, cache_read=800))

    assert usage.cache_hit == 800
    assert usage.cache_miss == 200, "1000 - 800"
    assert usage.input_tokens == 1000


def test_missing_usage_is_not_zero() -> None:
    """拿不到用量记 `missing`，不是记零 —— 两者必须能分开。"""
    usage = usage_of_message(message(usage=False))

    assert usage.calls == 0
    assert usage.missing == 1


def test_no_cache_details_means_all_missed() -> None:
    """没有缓存明细时全部算未命中（贵的那一档）—— 宁可高估，不要低估。"""
    raw = AIMessage(
        content="",
        usage_metadata={"input_tokens": 500, "output_tokens": 20, "total_tokens": 520},
    )
    usage = usage_of_message(raw)

    assert usage.cache_miss == 500
    assert usage.cache_hit == 0


def test_usage_of_scans_only_ai_messages() -> None:
    """扫消息只认 AI 消息 —— 人类消息与工具结果没有用量。"""
    total = usage_of([HumanMessage(content="x"), message(output=7), message(output=3)])

    assert total.calls == 2
    assert total.output == 10


def test_cost_uses_three_rates() -> None:
    """三档单价各乘各的。

    用一组手算得出来的数字钉住公式：200×4 + 800×0.5 + 100×12 = 2400（微元）
    """
    usage = UsageTotals(calls=1, cache_hit=800, cache_miss=200, output=100)

    assert cost_yuan(usage, PRICES) == pytest.approx(0.0024)


def test_cost_is_dominated_by_output_not_by_token_count() -> None:
    """输出的单价最高，**省 token ≠ 省钱** —— 这条是「轮数当不了成本代理」的算术版。

    同样 1000 个 token：输出 1000 个要 0.012 元，命中缓存的输入只要 0.0005 元 —— 差 24 倍。
    """
    output_heavy = cost_yuan(UsageTotals(output=1000), PRICES)
    cache_heavy = cost_yuan(UsageTotals(cache_hit=1000), PRICES)

    assert output_heavy / cache_heavy == pytest.approx(24.0)


def test_add_usage_sums_every_field() -> None:
    """累加要带上 missing —— 漏了它，「拿不到用量」会在汇总时消失。"""
    total = add_usage(
        UsageTotals(calls=1, cache_hit=10, cache_miss=20, output=30, missing=0),
        UsageTotals(calls=2, cache_hit=1, cache_miss=2, output=3, missing=1),
    )

    assert total == UsageTotals(calls=3, cache_hit=11, cache_miss=22, output=33, missing=1)


def test_exceeded_returns_none_when_within_budget() -> None:
    """没超就是没超（闸门不能常驻）。"""
    assert exceeded(UsageTotals(), 0.0, prices=PRICES, max_cost=2.0, max_seconds=300) is None


def test_exceeded_on_cost() -> None:
    """花费到点即停。"""
    usage = UsageTotals(output=200_000)  # 200k 输出 ≈ 2.4 元

    assert exceeded(usage, 1.0, prices=PRICES, max_cost=2.0, max_seconds=300) == STOP_COST


def test_exceeded_on_time() -> None:
    """时间到点即停 —— 用量拿不到时它是唯一还在涨的量。"""
    assert exceeded(UsageTotals(), 301.0, prices=PRICES, max_cost=2.0, max_seconds=300) == STOP_TIME


def test_non_positive_limit_means_unlimited() -> None:
    """上限填 0 或负数表示「这一档不限制」，不是「立刻停」。

    这条要写死：配错了（比如填 0）时，最坏的结果是少一道闸，而不是每条 review
    一开局就停。
    """
    usage = UsageTotals(output=10_000_000)

    assert exceeded(usage, 99999.0, prices=PRICES, max_cost=0.0, max_seconds=0) is None


def test_describe_mentions_missing_usage() -> None:
    """说明里要带上「拿不到用量」的提示 —— 花的钱只是个下界。"""
    text = describe(UsageTotals(calls=1, missing=2), PRICES)

    assert "拿不到用量" in text
    assert "0.000 元" in text
