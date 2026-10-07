"""静态指导组装。

把审查单元（Group）渲染成 user 消息的三块：**diff 块**（审查范围 + `<diff>`）、
**静态发现块**（静态分析提示，可能为空）、**要求块**（主任务 + 工作方式 +
结束审查）。V1 不注入调用图与规则。

三块的分工按「数据 / 流程」来分：待审代码是有字段的数据，用 XML；要求是流程
描述，用 markdown。**数据在前、要求在后** —— 模型决策前最后读到的是一段短要求，
而不是还要翻过一整段 diff 才知道要审什么。

**文本本身整块写在 `templates.py` 里**，这里只负责填变量 —— 这样一眼能看出
LLM 收到的整体结构，而不是 `"\\n".join((...))` 一行一行拼。

静态发现的两处呈现（设计文档 18 号 §2.6）：**独立成段**列出（保留跨 hunk 的
全局视角」，另外在每个 hunk 上标 `static_hint="S-01"` 作为与 hunk 的桥。
"""

import logging
from collections.abc import Sequence

from prism.core.guidance import templates as tpl
from prism.models.ast import HunkWithSymbol
from prism.models.grouping import Group
from prism.models.static import NO_HUNK_MARK, PRIORITY_RANK, StaticFinding

logger = logging.getLogger(__name__)

# 没有静态发现的 hunk 的排序档位，排在 low 之后（设计 §2.8 的第 3 条）
NO_HINT_RANK = len(PRIORITY_RANK)


def build_diff_block(group: Group, findings: Sequence[StaticFinding] = ()) -> str:
    """组装 diff 块：hunk 清单摘要 + 本组待审代码。

    Args:
        group: 审查单元
        findings: 注入本组的静态发现（为空时行为与没有静态分析时完全一致）

    Returns:
        diff 块文本；本组没有 hunk 时只给一句说明
    """
    if not group.hunks:
        return tpl.SCOPE_EMPTY

    hints = _hints_by_hunk(findings)
    ids = sorted(item.hunk_id for item in group.hunks)
    span = ids[0] if len(ids) == 1 else f"{ids[0]} ~ {ids[-1]}"
    # **这里不打日志**：分组装箱的试算（`grouping.group_tokens`）会拿一个临时组
    # 反复调本函数量成本，O(k²) 次。日志挂在真调用点（`initial_state` / `reflect`），
    # 那里才有真的组名与「这是审查还是反思」的语境
    return tpl.SCOPE.format(
        span=span,
        count=len(group.hunks),
        files=len(group.files),
        order_note=tpl.ORDER_NOTE if hints else "",
        diff=_render_diff(group, hints),
    )


def build_static_block(findings: Sequence[StaticFinding]) -> str:
    """组装静态发现块（设计 §2.9 第一层）。

    **带前导空行**：它是可选段，`USER` 模板里跟在 diff 块后面，靠这个空行决定
    有没有段间隔。没有发现时返回空串 —— 不写「本次没有静态发现」这类废话。

    Args:
        findings: 注入本组的静态发现

    Returns:
        带前导空行的段落文本；没有发现时返回空串
    """
    if not findings:
        return ""
    rows = "\n".join(_render_static_row(item) for item in findings)
    return "\n\n" + tpl.STATIC_SECTION.format(rows=rows)


def build_requirements() -> str:
    """组装要求块：主任务 + 工作方式 + 结束审查。

    三小段都是静态的，不随审查单元变化 —— 随组变化的东西全在 diff 块里。

    Returns:
        要求块文本
    """
    return "\n\n".join((_render_main_task(), _render_workflow(), tpl.FINISH))


def _render_main_task() -> str:
    """渲染主任务段（6 条维度）。

    Returns:
        主任务段落
    """
    items = "\n".join(
        f"{index}. {text}" for index, text in enumerate(tpl.MAIN_TASK_DIMENSIONS, start=1)
    )
    return tpl.MAIN_TASK.format(items=items)


def _render_workflow() -> str:
    """渲染工作方式段落（含报告节奏）。

    Returns:
        工作方式段落
    """
    return tpl.WORKFLOW_SECTION.format(items="\n".join(f"- {text}" for text in tpl.WORKFLOW))


def _render_static_row(item: StaticFinding) -> str:
    """渲染静态发现表格的一行。

    描述里的 `|` 必须转义：表格单元格里一个裸竖线会把这一行切断，后面几列全错位。
    截断与压行在分析阶段就做完了（`analyzer._flatten`）。

    Args:
        item: 静态发现

    Returns:
        markdown 表格行
    """
    return tpl.STATIC_ROW.format(
        finding_id=item.finding_id,
        path=item.path,
        line=item.line,
        mark=tpl.PRIORITY_MARKS.get(item.priority, item.priority),
        hunk_id=item.hunk_id if item.hunk_id is not None else NO_HUNK_MARK,
        text=item.text.replace("|", "\\|"),
    )


def _hints_by_hunk(findings: Sequence[StaticFinding]) -> dict[str, list[StaticFinding]]:
    """按 hunk 编号归拢静态发现（关联不上 hunk 的丢弃）。

    Args:
        findings: 注入本组的静态发现

    Returns:
        hunk 编号 → 发现列表；没有任何关联时返回空字典
    """
    result: dict[str, list[StaticFinding]] = {}
    for item in findings:
        if item.hunk_id is None:
            continue
        result.setdefault(item.hunk_id, []).append(item)
    return result


def _display_order(
    hunks: Sequence[HunkWithSymbol], hints: dict[str, list[StaticFinding]]
) -> list[HunkWithSymbol]:
    """按静态发现优先级给 hunk 重排展示顺序（设计 §2.8）。

    三条规则：关联发现的最高优先级降序 → hunk 编号升序 → 没有关联的排最后。
    **只改展示顺序，不改 hunk 编号** —— 编号是报告评论与查日志的坐标，重排会让
    「H-05 在 H-01 前面」变成一种可解释的跳跃，而编号一动就全乱。

    Args:
        hunks: 本组 hunk（已按路径 + 行号排序）
        hints: hunk 编号 → 发现列表

    Returns:
        重排后的新列表；没有任何关联时返回原序列的副本
    """
    if not hints:
        return list(hunks)

    def key(item: HunkWithSymbol) -> tuple[int, str]:
        related = hints.get(item.hunk_id)
        if not related:
            return (NO_HINT_RANK, item.hunk_id)
        return (min(PRIORITY_RANK.get(f.priority, NO_HINT_RANK) for f in related), item.hunk_id)

    return sorted(hunks, key=key)


def _render_diff(group: Group, hints: dict[str, list[StaticFinding]]) -> str:
    """把本组 hunk 渲染成 `<diff>` 块。

    一个文件一个 `<file>`，文件内的 hunk 各自一个 `<hunk>`。**hunk 的位置、归属、
    可读范围全部挂在标签属性上，正文里不再重复标一遍** —— 旧版在「审查范围」里
    列一遍清单、又在 diff 正文里标一遍编号，模型得在两处之间来回对编号。

    按文件分组用 `groupby` 要求同文件的 hunk 连续 —— 展示顺序重排之后这条不再
    天然成立（重排是按优先级），所以这里先按文件路径收进有序字典，再逐块渲染。

    Args:
        group: 审查单元
        hints: hunk 编号 → 发现列表

    Returns:
        `<diff>` 块文本
    """
    by_file: dict[str, list[HunkWithSymbol]] = {}
    for item in _display_order(group.hunks, hints):
        by_file.setdefault(item.file_path, []).append(item)

    blocks: list[str] = ["<diff>"]
    for path, hunks in by_file.items():
        blocks.append(
            tpl.DIFF_FILE.format(
                path=path,
                extra=_file_extra(hunks[0]),
                hunks="\n".join(_render_hunk(item, hints.get(item.hunk_id)) for item in hunks),
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


def _render_hunk(item: HunkWithSymbol, hints: Sequence[StaticFinding] | None) -> str:
    """渲染单个 `<hunk>`。

    位置用 git 的 `起始行,行数` 写法而不是「起始行-结束行」：纯删除在新文件侧
    是**空区间**（`6,0`），区间写法表达不了空区间，文件头纯删除更会退化成
    「第 0 行」这种废坐标。`0` 在这个写法里天然就是「该侧没有内容」。

    `static_hint` **只放编号，不放严重性**（设计 §2.7 层 1）：hunk 属性描述的是
    代码位置，严重性是静态工具的推测，重复一遍就会与静态发现列表里的口径打架。

    Args:
        item: 带符号的 hunk
        hints: 关联到这个 hunk 的静态发现；没有时为 None

    Returns:
        `<hunk>` 标签文本
    """
    part = item.part
    extra = ""
    if item.head_lines is not None:
        start, end = item.head_lines
        extra = f' lines="{start}-{end}"'
    if hints:
        extra += f' static_hint="{" ".join(f.finding_id for f in hints)}"'
    return tpl.HUNK.format(
        hunk_id=item.hunk_id,
        old=f"{part.old_start},{part.old_lines}",
        new=f"{part.new_start},{part.new_lines}",
        symbol=item.symbol.name,
        kind=item.symbol.kind,
        extra=extra,
        content=part.content,
    )
