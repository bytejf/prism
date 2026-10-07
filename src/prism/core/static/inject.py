"""静态发现的注入：按「整条请求」的 token 预算切分。

设计文档 18 号 §2.4：单组提示词的预算是

    提示词模板 + hunk 内容 + 静态发现 ≤ max_prompt_tokens

超了就**从静态发现那头砍**（按优先级降序丢），diff 一个字都不砍 —— 静态发现是
线索，待审代码是主体，装不下时先丢线索。同一条思路：不做固定条数上限，因为
「20 条」这种数字与它占多少 token 没有关系。

估算用**字符数 ÷ 2.5**（设计 §2.4 的取舍）：零依赖，误差 ±20% 由预留余量吸收。
要精确到 DeepSeek 的分词器就得引入 tiktoken，而它算的也不是同一个分词器。

预算测得的是**真实渲染出来的文本**（增量渲染后量长度），不是另一套「每条多少
token」的估算模型 —— 那种两处会漂。

**一处已知的偏差**：量基准时 diff 块上还没挂 `static_hint` 属性（那是注入之后
才渲染出来的），少算了每个 hunk 十几个字符。方向是保守的（真实请求比估算的略大），
量级可忽略。
"""

import logging
from collections.abc import Sequence

from prism.config.settings import settings
from prism.core.guidance import templates as tpl
from prism.core.guidance.builder import build_diff_block, build_requirements, build_static_block
from prism.models.grouping import Group
from prism.models.static import StaticFinding

logger = logging.getLogger(__name__)

# 字符数 → token 的除数。中文约 1.5 字/token、英文约 4 字符/token，取中间值
CHARS_PER_TOKEN = 2.5


def estimate_tokens(text: str) -> int:
    """粗估一段文本的 token 数（字符数 ÷ 2.5）。

    Args:
        text: 任意文本

    Returns:
        token 估算值（向上取整，空串为 0）
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def base_prompt_tokens(group: Group) -> int:
    """算「不含静态发现」的那部分提示词的 token 数。

    system 段也算进去 —— 它每一轮都要重新发一次，预算管的是整条请求。

    Args:
        group: 审查单元

    Returns:
        估算 token 数
    """
    return (
        estimate_tokens(tpl.SYSTEM)
        + estimate_tokens(build_requirements())
        + estimate_tokens(build_diff_block(group))
    )


def select_within_budget(group: Group, findings: Sequence[StaticFinding]) -> list[StaticFinding]:
    """挑出本组要注入的静态发现。

    入参已按优先级降序，所以「装不下就停」等价于「先丢最不要紧的」。累积到某条
    装不下就停，不再往后试 —— 后面的是更低优先级，只会更装不下。

    Args:
        group: 审查单元
        findings: 本组文件上的静态发现，按优先级降序

    Returns:
        注入的发现列表（可能为空）
    """
    if not findings:
        return []

    budget = settings.static.max_prompt_tokens
    available = budget - base_prompt_tokens(group)
    if available <= 0:
        logger.warning(
            "静态发现全部丢弃：本组提示词已用 %d token，超过预算 %d（%s）",
            budget - available,
            budget,
            group.group_id,
        )
        return []

    kept: list[StaticFinding] = []
    for item in findings:
        trial = [*kept, item]
        if estimate_tokens(build_static_block(trial)) > available:
            break
        kept = trial

    if len(kept) < len(findings):
        logger.warning(
            "%s 的静态发现按 token 预算截断：%d → %d 条（预算 %d，已用 %d）",
            group.group_id,
            len(findings),
            len(kept),
            budget,
            budget - available,
        )
    else:
        logger.debug(
            "%s 注入静态发现 %d 条（预算 %d，提示词已用 %d）",
            group.group_id,
            len(kept),
            budget,
            budget - available,
        )
    return kept
