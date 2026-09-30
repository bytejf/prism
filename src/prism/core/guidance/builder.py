"""静态指导组装。

把审查单元（Group）渲染成一段静态指导文本，放在 user 消息的开头（diff 之前）。
V1 不注入调用图、静态分析发现与规则；这四段全部是静态内容 + 本组 hunk 清单。
"""

import logging

from prism.core.ast.mapper import format_location, format_symbol
from prism.models.grouping import Group

logger = logging.getLogger(__name__)

# 强制审查维度（V1 固定，不做用户配置）
REVIEW_DIMENSIONS: tuple[str, ...] = (
    "**业务逻辑正确性** — 改动是否符合业务语义？是否存在逻辑错误？",
    "**跨文件一致性** — 调用方 / 实现方是否同步适配？",
    "**意图与实现一致性** — 改动是否实现了 commit message 声称的意图？",
    "**边界条件** — 空值、并发、错误路径、极端输入是否处理？",
    "**副作用** — 改动是否影响其他功能的隐式契约？",
)

# 报告节奏引导
REPORT_PACE: tuple[str, ...] = (
    "- 每完成 1~2 个 hunk 的审查，立即调用 report_hunks 报告结论",
    "- 不要等到所有 hunk 审完后一次性报告",
    "- 如果某个 hunk 需要深入探索，可以在探索完成后报告该 hunk",
    "- 如果发现多个问题属于同一 hunk，可以合并到一次 report_hunks 中",
)


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
    lines = ["### 审查范围"]
    if not group.hunks:
        lines.append("本组没有 hunk。")
        return "\n".join(lines)

    first = group.hunks[0].hunk_id
    last = group.hunks[-1].hunk_id
    span = first if first == last else f"{first} ~ {last}"
    lines.append(f"本组 hunk：{span}（共 {len(group.hunks)} 个）")

    for item in group.hunks:
        lines.append("")
        lines.append(format_location(item))
        lines.append(f"  归属：{format_symbol(item.symbol)}")
    return "\n".join(lines)


def _render_dimensions() -> str:
    """渲染强制审查维度段落。

    Returns:
        审查要求段落
    """
    lines = [
        "### 审查要求（对所有 hunk 强制适用）",
        "",
        "无论初始注意力等级如何，每个 hunk 都必须完整审查以下维度：",
        "",
    ]
    lines.extend(f"{index}. {text}" for index, text in enumerate(REVIEW_DIMENSIONS, start=1))
    lines.extend(
        (
            "",
            "⚠️ 注意：所有 hunk 都可能存在严重缺陷。",
            "静态分析无法发现业务语义级的 Bug，你的语义审查是发现这类问题的唯一防线。",
        )
    )
    return "\n".join(lines)


def _render_pace() -> str:
    """渲染报告节奏段落。

    Returns:
        报告节奏段落
    """
    lines = ["### 报告节奏", "", "随着审查进度逐步报告，不要攒批。", ""]
    lines.extend(REPORT_PACE)
    return "\n".join(lines)


def _render_finish() -> str:
    """渲染结束审查段落。

    Returns:
        结束审查段落
    """
    return "\n".join(
        (
            "### 结束审查",
            "",
            "所有 hunk 审查完毕后，调用 end_done 结束审查。",
            "如果有 hunk 未标记 complete，end_done 会被拒绝，并提示剩余 hunk。",
        )
    )


def build_diff_text(group: Group) -> str:
    """把本组 hunk 渲染成 diff 文本。

    hunk.content 只有正文（见 Hunk.content），这里为每个 hunk 补两样东西：

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
    parts: list[str] = []
    for path in group.files:
        hunks = [item for item in group.hunks if item.file_path == path]
        if not hunks:
            continue
        parts.append(f"--- {path} ---")
        for item in hunks:
            parts.append(f"@@ {format_location(item)} @@")
            parts.append(item.part.content)
    return "\n".join(parts)


def build_system_prompt() -> str:
    """组装主任务 system prompt。

    只放角色定义与硬约束；审查维度、报告节奏、结束方式写在
    build_guidance() 的输出里，避免同一条要求在两处重复。
    这里只做一处指路，不再抄一遍具体条目。

    Returns:
        system prompt 文本
    """
    return "\n".join(
        (
            "你是一名资深 Go 代码审查员。你正在审查一个变更集（PR）中的若干 hunk。",
            "",
            "## 工作方式",
            "",
            "- 先用 file_read 读取 hunk 周边上下文，必要时用 code_search 查调用方与定义",
            "- 审查完成后用 report_hunks 报告该 hunk 的结论（有问题就给出评论，没问题也要 "
            "complete=true）",
            "- 所有 hunk 报告完毕后调用 end_done 结束",
            "",
            "## 硬约束",
            "",
            "- 只审查给定的 hunk，不要审查未列出的代码",
            "- 每条评论必须给出 severity、category、content、existing_code",
            "- 报告节奏与审查维度的具体要求见随后的 user 消息",
        )
    )
