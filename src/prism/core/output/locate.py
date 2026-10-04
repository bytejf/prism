"""评论定位：把 LLM 报的评论挂到具体行号上。

**零 LLM** 的确定性步骤 —— 同一输入同一输出。三层回退：

| 层 | 搜索范围 | locate_level | 多命中时 |
|---|---|---|---|
| 1 | 评论自己那个 hunk | ``hunk`` | **取第一个** |
| 2 | 本组其它 hunk | ``group`` | 放弃 → ``failed`` |
| 3 | 都不中 | ``failed`` | 收拢到未定位聚合 |

## 为什么匹配面是「新增行」，不是「整个文件」

diff 由 ``-U0`` 生成（`loader.CONTEXT_LINES`），hunk 正文里只有改动行。
定位只在 ``+`` 行上做，两条理由：

1. **只有 ``+`` 行有被审查版本的坐标**。纯删除的 hunk ``new_lines == 0``，
   整段在新版本里没有位置，命中它也给不出行号；拿旧版行号回填会让同一个
   字段里混进两套坐标 —— 那正是本项目最怕的错法。
2. **行内评论只能挂在改动行上**（GitHub / GitLab 的 PR 注释就是这个约束）。
   挂在「文件里有、但不在 diff 里」的行上，消费方根本挂不出去。所以
   搜索面取「本组的改动行」不只是省事，它是唯一有用的那个面。

## 为什么组内命中要改写 hunk_id

第二层可能在**另一个文件**里命中，而 ``path`` 是从 ``hunk_id`` 查出来的。
若保留原编号，评论的 ``path`` 与 ``start_line`` 会指向不同文件 —— 两个字段
各说各话，还都不报错。所以命中哪个 hunk，``hunk_id`` 就改成哪个（``path``
随之走），并记一条 INFO。
"""

import logging
from collections import Counter
from typing import NamedTuple

from prism.core.diff.parser import UNKNOWN_PATH
from prism.models.ast import HunkWithSymbol
from prism.models.comment import (
    LOCATE_FAILED,
    LOCATE_GROUP,
    LOCATE_HUNK,
    REASON_EMPTY_EXISTING_CODE,
    REASON_MULTI_MATCH,
    REASON_NO_MATCH,
    Comment,
)
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)

# diff 正文的行首记号：`+` 新增、`-` 删除
_PREFIXES = ("+", "-")


class Location(NamedTuple):
    """一条评论的定位结果。

    Attributes:
        hunk_id: 命中所在 hunk；未定位时仍是评论原报的那个
        start_line: 命中范围的起始行（被审查版本的坐标）；未定位为 0
        end_line: 命中范围的结束行（同上）；未定位为 0
        level: 定位层级（hunk / group / failed）
        reason: 未定位原因，仅 level 为 failed 时非空
        candidates: 决定这次定位的候选位置数 —— 0 表示没找到，1 表示唯一命中，
            >1 表示在**同一层**找到多处（hunk 层取首个、组内层放弃）
    """

    hunk_id: str
    start_line: int
    end_line: int
    level: str
    reason: str | None
    candidates: int = 0


def normalize(line: str) -> str:
    """归一化一行：去掉 diff 行首的 ``+`` / ``-`` 记号与首尾空白。

    **大小写不动** —— Go 是大小写敏感的语言，``Err`` 与 ``err`` 是两个东西。
    缩进不影响：Go 里缩进不承载语义，而且模型抄写时重新缩进过，也不该因此
    定位失败。前缀去不去都行，模型从 diff 抄会带上、从源码抄不会带。

    Args:
        line: diff 正文的一行（可能带前缀），或模型给的原文

    Returns:
        归一化后的文本
    """
    body = line[1:] if line[:1] in _PREFIXES else line
    return body.strip()


def _added_lines(part: Hunk) -> list[tuple[int, str]]:
    """hunk 片里带行号的新增行。

    ``+`` 行按顺序占用被审查版本的行号，``-`` 行不占 —— 它属于旧版侧。
    所以计数器只在遇到 ``+`` 时前进，这条是行号算对的关键：把 ``-`` 也算进去，
    改一行删一行的那种 hunk 会整体偏移。

    Args:
        part: hunk 片

    Returns:
        (行号, 归一化后的正文) 列表
    """
    lines: list[tuple[int, str]] = []
    number = part.new_start
    for line in part.content.splitlines():
        if line.startswith("+"):
            lines.append((number, normalize(line)))
            number += 1
    return lines


def _matches(needle: list[str], added: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """在新增行里找 needle，返回所有命中的 (起始行, 结束行)。

    两边都先去掉空行再比：diff 里新增一个空行会渲染成孤零零一个 ``+``，
    归一化后是空串。空行不承载定位信息，让它参与比对只会把连续匹配打断。

    Args:
        needle: 归一化并去掉空行后的 existing_code（**非空**，调用方保证）
        added: 带行号的新增行

    Returns:
        命中的行号区间列表，按位置升序
    """
    surface = [(number, text) for number, text in added if text]
    width = len(needle)
    return [
        (surface[index][0], surface[index + width - 1][0])
        for index in range(len(surface) - width + 1)
        if [text for _, text in surface[index : index + width]] == needle
    ]


def locate(hunk_id: str, existing_code: str, hunks: list[HunkWithSymbol]) -> Location:
    """把一条评论挂到行号上。

    Args:
        hunk_id: 评论报告的 hunk 编号
        existing_code: 评论给的代码片段
        hunks: **本组**的全部 hunk（第一、二层都在它里面找）

    Returns:
        定位结果
    """
    needle = [text for text in (normalize(line) for line in existing_code.splitlines()) if text]
    # schema 只保证「字段存在」，保证不了「内容有意义」：空串、纯空白都可能在
    # 这里出现。它没有可定位的信息，但不丢评论 —— 后端收拢到聚合
    if not needle:
        return Location(hunk_id, 0, 0, LOCATE_FAILED, REASON_EMPTY_EXISTING_CODE)

    own = next((item for item in hunks if item.hunk_id == hunk_id), None)
    if own is not None:
        hits = _matches(needle, _added_lines(own.part))
        if hits:
            # 同一 hunk 内多命中**取第一个**，不放弃：两个候选都在同一段改动里，
            # 取首个比把评论丢进聚合更划算（代价只是行号可能偏几行）。
            # 决策的代价要靠数字盯住，所以 candidates 带出去给 locate_group 计数
            return Location(hunk_id, hits[0][0], hits[0][1], LOCATE_HUNK, None, len(hits))

    # 第一层没中才走第二层 —— 第一层中了就不必再找（它已在最精确的那层）
    group_hits = [
        (item.hunk_id, start, end)
        for item in hunks
        for start, end in _matches(needle, _added_lines(item.part))
    ]
    if len(group_hits) == 1:
        hit_id, start, end = group_hits[0]
        return Location(hit_id, start, end, LOCATE_GROUP, None, 1)
    if len(group_hits) > 1:
        return Location(hunk_id, 0, 0, LOCATE_FAILED, REASON_MULTI_MATCH, len(group_hits))
    return Location(hunk_id, 0, 0, LOCATE_FAILED, REASON_NO_MATCH)


def locate_group(group: Group, hunk_comments: dict[str, list[CommentInput]]) -> list[Comment]:
    """定位一个组产出的全部评论，转成可输出的 Comment。

    评论**一条都不丢**：定位失败的那些照样产出 Comment，只是带上
    ``locate_level = failed`` 与原因，由渲染层单独列出。

    Args:
        group: 审查单元，第二层的搜索范围就是它
        hunk_comments: hunk 编号 → 该 hunk 收集到的评论

    Returns:
        Comment 列表，顺序与 hunk_comments 一致（排序交给 sort_comments）
    """
    index = {item.hunk_id: item for item in group.hunks}
    levels: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    ambiguous = 0
    comments: list[Comment] = []

    for hunk_id, items in hunk_comments.items():
        for raw in items:
            found = locate(hunk_id, raw.existing_code, group.hunks)
            levels[found.level] += 1
            if found.level == LOCATE_HUNK and found.candidates > 1:
                ambiguous += 1
            if found.reason is not None:
                reasons[found.reason] += 1

            # report_hunks 已经挡掉未知编号，走到这里说明编号与本组对不上
            item = index.get(found.hunk_id)
            if item is None:
                logger.warning("评论的 hunk_id 在本组查不到：%s", found.hunk_id)
                path = UNKNOWN_PATH
            else:
                path = item.file_path
                if found.hunk_id != hunk_id:
                    logger.info(
                        "评论原报 %s，实际定位在 %s（%s:%d）",
                        hunk_id,
                        found.hunk_id,
                        path,
                        found.start_line,
                    )

            comments.append(
                Comment(
                    path=path,
                    severity=raw.severity,
                    category=raw.category,
                    content=raw.content,
                    existing_code=raw.existing_code,
                    hunk_id=found.hunk_id,
                    locate_level=found.level,
                    locate_reason=found.reason,
                    suggestion_code=raw.suggestion_code,
                    start_line=found.start_line,
                    end_line=found.end_line,
                )
            )

    total = len(comments)
    if total:
        # §4.1 要求的「定位开始」那条不单独打：它是瞬时的，评论数在同一条里
        # 就够，两条日志之间没有可观测的间隔
        logger.info(
            "定位完成 %s：%d 条，hunk 内 %d / 组内 %d / 未定位 %d，成功率 %.0f%%"
            "（其中同 hunk 多命中取首个 %d 条）",
            group.group_id,
            total,
            levels[LOCATE_HUNK],
            levels[LOCATE_GROUP],
            levels[LOCATE_FAILED],
            (total - levels[LOCATE_FAILED]) / total * 100,
            ambiguous,
        )
    if levels[LOCATE_FAILED]:
        logger.warning(
            "未定位 %d 条（%s），已在输出里单列、未丢弃",
            levels[LOCATE_FAILED],
            "、".join(f"{name} {count}" for name, count in sorted(reasons.items())),
        )
    return comments
