"""静态指导组装。

把审查单元（Group）渲染成一段静态指导文本，放在 user 消息的开头（diff 之前）。
V1 不注入调用图、静态分析发现与规则；这四段全部是静态内容 + 本组 hunk 清单。

**文本本身整块写在 `templates.py` 里**，这里只负责填变量 —— 这样一眼能看出
LLM 收到的整体结构，而不是 `"\\n".join((...))` 一行一行拼。
"""

import logging
from itertools import groupby

from prism.core.ast.mapper import format_location, format_symbol
from prism.core.guidance import templates as tpl
from prism.models.grouping import Group

logger = logging.getLogger(__name__)


def build_guidance(group: Group) -> str:
    """组装静态指导文本。

    Args:
        group: 审查单元

    Returns:
        静态指导文本，包含审查范围、审查要求、报告节奏、结束审查
    """
    text = "\n\n".join(
        (
            _render_scope(group),
            _render_dimensions(),
            _render_pace(),
            _render_finish(),
        )
    )
    logger.debug(
        "组装静态指导：%s（%d 个 hunk，%d 字符）", group.group_id, len(group.hunks), len(text)
    )
    return text


def _render_scope(group: Group) -> str:
    """渲染审查范围：hunk 编号区间 + 逐个 hunk 的位置与归属。

    Args:
        group: 审查单元

    Returns:
        审查范围段落
    """
    if not group.hunks:
        return tpl.SCOPE_EMPTY

    listing = "\n\n".join(
        f"{format_location(item)}\n  归属：{format_symbol(item.symbol)}" for item in group.hunks
    )
    first, last = group.hunks[0].hunk_id, group.hunks[-1].hunk_id
    return tpl.SCOPE.format(
        span=first if first == last else f"{first} ~ {last}",
        count=len(group.hunks),
        listing=listing,
    )


def _render_dimensions() -> str:
    """渲染强制审查维度段落。

    Returns:
        审查要求段落
    """
    items = "\n".join(
        f"{index}. {text}" for index, text in enumerate(tpl.REVIEW_DIMENSIONS, start=1)
    )
    return tpl.DIMENSIONS.format(items=items)


def _render_pace() -> str:
    """渲染报告节奏段落。

    Returns:
        报告节奏段落
    """
    return tpl.PACE.format(items="\n".join(tpl.REPORT_PACE))


def _render_finish() -> str:
    """渲染结束审查段落。

    Returns:
        结束审查段落
    """
    return tpl.FINISH


def build_diff_text(group: Group) -> str:
    """把本组 hunk 渲染成 diff 文本。

    hunk 正文只有内容行（见 Hunk.content），这里为每个 hunk 补两样东西：

    - 文件分隔行 `--- path ---`，按文件分组
    - hunk 标记 `@@ H-01 path:6 @@`，既做边界分隔，又把 hunk 编号和位置直接
      摆在正文旁边，模型调 report_hunks 时不用回头翻指导文本

    标记内容全部来自我们自己的字段（改动行算出来的），不用 git 那段
    「最近的前一个声明」—— 实测它 18~35% 的 hunk 会指错（见 parser._make_hunk）。

    Args:
        group: 审查单元

    Returns:
        diff 文本，按文件分组
    """
    # group.hunks 已按 (文件路径, 行号) 排序，同一文件的 hunk 是连续的，
    # 一次 groupby 就够 —— 不必按文件反复过滤整个列表
    files = [
        tpl.DIFF_FILE.format(
            path=path,
            hunks="\n".join(
                f"@@ {format_location(item)} @@\n{item.part.content}" for item in hunks
            ),
        )
        for path, hunks in groupby(group.hunks, key=lambda item: item.file_path)
    ]
    return "\n".join(files)
