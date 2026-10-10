"""规则加载工具：把 L1 索引点到名的规则正文取回来。

设计见 `acr-min/08-规则注入/26-规则库组织与渐进式加载设计.md` §6.1。

**唯一一个不吃 `runtime` 的审查工具**：规则库是**随包发布**的静态内容，
与被审查仓库、与当前这一组都无关 —— 没有要从 `ReviewContext` / `ReviewState`
里取的东西。所以它是纯函数，也正因此可以单独调用（用例里直接调，不用搭图）。
"""

import logging

from langchain_core.tools import tool

from prism.core.rules.library import default_library

logger = logging.getLogger(__name__)

# 单次最多加载几条（设计 §6.1）。超了**整次拒绝**并说明原因，不悄悄截断：
# 截断会让模型以为自己看到了全部，而它恰恰是因为「命中太多」才需要被拦一下
MAX_RULE_IDS = 5


@tool(
    parse_docstring=True,
    description=(
        "加载审查规则的正文（正例、反例、边界情况）。\n"
        "「代码审查规则索引」只给「代码特征 → 规则代号」，代码里看到某个特征时用它把"
        "对应规则读进来，再按规则判断。一次最多 5 条；超过说明这一处涉及的主题太杂，"
        "先挑最对得上的那几条。"
    ),
)
def load_rules(rule_ids: list[str]) -> str:
    """加载规则正文。

    Args:
        rule_ids: 规则代号列表，取自「代码审查规则索引」，如 ["G-01", "G-03"]

    Returns:
        每条规则的正文；被完全覆盖的公共规则返回「改用哪一条」，未知代号单独列出来
    """
    if len(rule_ids) > MAX_RULE_IDS:
        logger.warning(
            "load_rules 一次要了 %d 条（上限 %d），整次拒绝", len(rule_ids), MAX_RULE_IDS
        )
        return (
            f"错误：一次最多加载 {MAX_RULE_IDS} 条规则，这次要了 {len(rule_ids)} 条。"
            f"先挑与当前代码最对得上的那几条；确实需要更多，就分几次调用。"
        )

    library = default_library()
    logger.info("load_rules：%s", ", ".join(rule_ids) or "（空）")
    if not rule_ids:
        return "错误：没有给规则代号。请从「代码审查规则索引」里选，如 [\"G-01\"]。"
    return library.render(list(rule_ids))
