"""反思循环的状态与静态上下文。

与 `core/agent/state.py` 同构：走 `create_agent(state_schema=...)`，**会变的东西**
（已受理的删除、漏报、受理次数）在状态里，**一次调用内不变的东西**（候选编号、
hunk 表）在 `ReflectionContext` 里，两者不要混。

## 为什么用 dict 装删除表，而不是 list

`comment_id` 天然是键：重复上报同一条就是覆盖，不需要额外的去重逻辑；模型一轮
里并行发多个工具调用时（审查那边实测 21 轮里有 8 轮），各写各的键也不会互相盖。
这和 `hunk_status` 用 dict + reducer 是同一个理由。
"""

import operator
from dataclasses import dataclass
from typing import Annotated

from langchain.agents.middleware import AgentState

from prism.models.reflection import PotentialMiss


@dataclass(frozen=True)
class Removal:
    """一条被去掉的评论。

    留一份完整的记录（编号、路径、依据、指认）—— 删评论是本流程里唯一有破坏性的
    动作，它必须可审计。

    Attributes:
        comment_id: 本次候选清单里的编号
        path: 该评论的文件路径
        ground: 删除依据（A / B）
        reasoning: 模型指认的那一处 diff
    """

    comment_id: str
    path: str
    ground: str
    reasoning: str


def merge_removals(left: dict[str, Removal], right: dict[str, Removal]) -> dict[str, Removal]:
    """删除表的 reducer —— 按编号合并。

    并行工具调用写回来的两份**增量**按 key 合并，各改各的互不覆盖。

    Args:
        left: 已有表
        right: 本次写入的增量

    Returns:
        合并后的表
    """
    return {**left, **right}


class ReflectionState(AgentState):
    """反思循环的状态。

    Attributes:
        removals: 已受理的删除（评论编号 → 记录）。**这是权威结论**
        misses: 已受理的疑似漏报
        reported: 成功受理的次数。它同时也是**退出依据** —— 大于 0 就说明模型
            已经交出了完整结论，不必再调一次模型去听它说「好的，完成」
    """

    removals: Annotated[dict[str, Removal], merge_removals]
    misses: Annotated[list[PotentialMiss], operator.add]
    reported: Annotated[int, operator.add]


@dataclass(frozen=True)
class ReflectionContext:
    """反思的静态上下文（一次调用内不变）。

    Attributes:
        comment_paths: 候选评论的编号 → 文件路径。**校验与补路径都靠它** ——
            编号对不对、路径是什么，都由这张表说了算，不让模型自己编
        hunk_paths: hunk 编号 → 文件路径。漏报信号的 path 从这里查
    """

    comment_paths: dict[str, str]
    hunk_paths: dict[str, str]
