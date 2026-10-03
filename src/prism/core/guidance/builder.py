"""静态指导组装。

把审查单元（Group）渲染成 user 消息的两块：**diff 块**（审查范围 + `<diff>`）
与**要求块**（审查维度 + 工作方式 + 结束审查）。V1 不注入调用图、静态分析发现
与规则。

两块的分工按「数据 / 流程」来分：待审代码是有字段的数据，用 XML；要求是流程
描述，用 markdown。**数据在前、要求在后** —— 模型决策前最后读到的是一段短要求，
而不是还要翻过一整段 diff 才知道要审什么。

**文本本身整块写在 `templates.py` 里**，这里只负责填变量 —— 这样一眼能看出
LLM 收到的整体结构，而不是 `"\\n".join((...))` 一行一行拼。
"""

import logging
from itertools import groupby

from prism.core.guidance import templates as tpl
from prism.models.ast import HunkWithSymbol
from prism.models.grouping import Group

logger = logging.getLogger(__name__)


def build_diff_block(group: Group) -> str:
    """组装 diff 块：hunk 清单摘要 + 本组待审代码。

    Args:
        group: 审查单元

    Returns:
        diff 块文本；本组没有 hunk 时只给一句说明
    """
    if not group.hunks:
        return tpl.SCOPE_EMPTY

    first, last = group.hunks[0].hunk_id, group.hunks[-1].hunk_id
    text = tpl.SCOPE.format(
        span=first if first == last else f"{first} ~ {last}",
        count=len(group.hunks),
        files=len(group.files),
        diff=_render_diff(group),
    )
    logger.debug(
        "组装 diff 块：%s（%d 个 hunk / %d 个文件，%d 字符）",
        group.group_id,
        len(group.hunks),
        len(group.files),
        len(text),
    )
    return text


def build_requirements() -> str:
    """组装要求块：审查维度 + 工作方式 + 结束审查。

    三小段都是静态的，不随审查单元变化 —— 随组变化的东西全在 diff 块里。

    Returns:
        要求块文本
    """
    return "\n\n".join((_render_dimensions(), _render_workflow(), tpl.FINISH))


def _render_dimensions() -> str:
    """渲染强制审查维度段落。

    Returns:
        审查要求段落
    """
    items = "\n".join(
        f"{index}. {text}" for index, text in enumerate(tpl.REVIEW_DIMENSIONS, start=1)
    )
    return tpl.DIMENSIONS.format(items=items)


def _render_workflow() -> str:
    """渲染工作方式段落（含报告节奏）。

    Returns:
        工作方式段落
    """
    return tpl.WORKFLOW_SECTION.format(items="\n".join(f"- {text}" for text in tpl.WORKFLOW))


def _render_diff(group: Group) -> str:
    """把本组 hunk 渲染成 `<diff>` 块。

    一个文件一个 `<file>`，文件内的 hunk 各自一个 `<hunk>`。**hunk 的位置、归属、
    可读范围全部挂在标签属性上，正文里不再重复标一遍** —— 旧版在「审查范围」里
    列一遍清单、又在 diff 正文里标一遍编号，模型得在两处之间来回对编号。

    `group.hunks` 已按 (文件路径, 行号) 排序，同一文件的 hunk 是连续的，
    一次 groupby 就够 —— 不必按文件反复过滤整个列表。

    Args:
        group: 审查单元

    Returns:
        `<diff>` 块文本
    """
    blocks: list[str] = ["<diff>"]
    for path, grouped in groupby(group.hunks, key=lambda item: item.file_path):
        hunks = list(grouped)
        blocks.append(
            tpl.DIFF_FILE.format(
                path=path,
                extra=_file_extra(hunks[0]),
                hunks="\n".join(_render_hunk(item) for item in hunks),
            )
        )
    blocks.append("</diff>")
    return "\n".join(blocks)


def _file_extra(first: HunkWithSymbol) -> str:
    """渲染 `<file>` 上的可选属性。

    两个都只在特定形态下出现，所以做成可选属性而不是空值属性：出现即意味着
    「这个文件改名了 / 被删了」，模型不用去比对属性值。

    Args:
        first: 该文件的第一个 hunk（文件级信息在同一个文件的每个 hunk 上都一样）

    Returns:
        ` deleted="true"` / ` renamed-from="..."` / 空串
    """
    if first.deleted:
        return ' deleted="true"'
    if first.renamed_from is not None:
        return f' renamed-from="{first.renamed_from}"'
    return ""


def _render_hunk(item: HunkWithSymbol) -> str:
    """渲染单个 `<hunk>`。

    位置用 git 的 `起始行,行数` 写法而不是「起始行-结束行」：纯删除在新文件侧
    是**空区间**（`6,0`），区间写法表达不了空区间，文件头纯删除更会退化成
    「第 0 行」这种废坐标。`0` 在这个写法里天然就是「该侧没有内容」。

    Args:
        item: 带符号的 hunk

    Returns:
        `<hunk>` 标签文本
    """
    part = item.part
    extra = ""
    if item.head_lines is not None:
        start, end = item.head_lines
        extra = f' lines="{start}-{end}"'
    return tpl.HUNK.format(
        hunk_id=item.hunk_id,
        old=f"{part.old_start},{part.old_lines}",
        new=f"{part.new_start},{part.new_lines}",
        symbol=item.symbol.name,
        kind=item.symbol.kind,
        extra=extra,
        content=part.content,
    )
