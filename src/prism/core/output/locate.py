"""评论定位：把 LLM 报的评论挂到具体行号上。

**零 LLM** 的确定性步骤 —— 同一输入同一输出。

## 两个维度：层 × 侧

| 维度 | 取值 | 含义 |
|---|---|---|
| **层**（`locate_level`） | hunk / group / failed | 在哪个范围里找到的 |
| **侧**（`side`） | RIGHT / LEFT | 命中的是**新版还是旧版**的行 |

层和侧**正交**，各有各的精度含义：层是「离这条评论自己那个 hunk 有多远」，
侧是「行号该去哪一版里读」。

## 为什么删除行也要能挂

`side` 照抄 GitHub 的 PR 行内评论模型（`side: LEFT | RIGHT`）。它的限制
从来不是「只能评论新行」，而是**「行必须在 diff 里」** —— 删除行是一等公民，
GitHub 对 hunk 之外的行才返回 422。

所以「评论讲的是被删掉的代码」这类不该丢进聚合：它的结论其实落在新版上
（守卫被删掉之后，那个位置上没有守卫了），但它引用的**证据**是被删的那几行，
把它们挂回**旧版行号**才是精确的答案。

实测依据：`prism-min-test` 上按「只搜新增行」跑，7 条评论有 2 条落聚合，
**全部**是删除类（`13-评论定位实现反馈.md` §7.3）。

## 一个片段可能对应两套坐标 —— 按行首记号拆

`existing_code` 是模型从 diff 或 `file_read` 抄来的，行首的 `+` / `-` 记号
**带不带都行**，而这个记号是关于「哪一侧」的唯一提示。于是先按记号拆成
几个 needle，再按固定优先级依次搜：

| 优先级 | needle | 搜哪侧 |
|---|---|---|
| 1 | `+` 行 | RIGHT |
| 2 | 无记号的 | RIGHT |
| 3 | `-` 行 | LEFT |
| 4 | 无记号的 | LEFT |

**层优先于侧**：本 hunk 内的 LEFT 命中比别组的 RIGHT 命中精确得多，所以把
一层里的四个 needle 都比完，再进下一层。

规则是「**第一个有命中的就定案**」：混着抄（`-旧` + `+新`）时落在 `+` 那侧
—— 评论讲的是替换出来的新代码。同一侧多命中仍按层分：hunk 层取首个
（两个候选都在同一段改动里，取首个比丢进聚合划算）、组内层放弃。

因为「无记号」的 needle 在两侧都试，无记号的删除行引用（模型可能顺手把
`-` 抹掉）也能落到 LEFT 上。

## 匹配面只有 diff 里的改动行

diff 由 `-U0` 生成（`loader.CONTEXT_LINES`），hunk 正文里只有改动行，能搜的
只有它们。这不是将就：**行内评论只能挂在 diff 里的行上**，「文件里有、但这次
没改」的行挂不出去。

## 为什么组内命中要改写 hunk_id

第二层可能在**另一个文件**里命中，而 `path` 是从 `hunk_id` 查出来的。
若保留原编号，评论的 `path` 与 `start_line` 会指向不同文件 —— 两个字段
各说各话，还都不报错。所以命中哪个 hunk，`hunk_id` 就改成哪个（`path` 随之走），
并记一条 INFO。
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
    SIDE_LEFT,
    SIDE_RIGHT,
    Comment,
)
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)

# diff 正文的行首记号 → 该行属于哪一侧。空串表示模型抄来时把记号抹掉了，
# 那就要两侧都试（见模块 docstring 的优先级表）
MARKER_SIDE: dict[str, str] = {"+": SIDE_RIGHT, "-": SIDE_LEFT}


class Location(NamedTuple):
    """一条评论的定位结果。

    Attributes:
        hunk_id: 命中所在 hunk；未定位时仍是评论原报的那个
        start_line: 命中范围的起始行，按 ``side`` 解释；未定位为 0
        end_line: 命中范围的结束行，按 ``side`` 解释；未定位为 0
        level: 搜索层级（hunk / group / failed）
        side: 行号属于哪一侧（LEFT / RIGHT）；未定位为 None
        reason: 未定位原因，仅 level 为 failed 时非空
        candidates: 决定这次定位的候选位置数 —— 0 没找到、1 唯一、>1 表示在
            同一层同一侧找到多处（hunk 层取首个、组内层放弃）
    """

    hunk_id: str
    start_line: int
    end_line: int
    level: str
    side: str | None
    reason: str | None
    candidates: int = 0


def _marker(line: str) -> str:
    """取一行的 diff 记号。

    Args:
        line: 可能带 `+` / `-` 前缀的一行

    Returns:
        `+` / `-`；没有记号时为空串
    """
    return line[:1] if line[:1] in MARKER_SIDE else ""


def normalize(line: str) -> str:
    """归一化一行：去掉 diff 行首的 `+` / `-` 记号与首尾空白。

    **大小写不动** —— Go 是大小写敏感的语言，`Err` 与 `err` 是两个东西。
    缩进不影响：Go 里缩进不承载语义，而且模型抄写时重新缩进过，也不该因此
    定位失败。

    Args:
        line: diff 正文的一行（可能带前缀），或模型给的原文

    Returns:
        归一化后的文本
    """
    body = line[1:] if _marker(line) else line
    return body.strip()


def _needles(existing_code: str) -> dict[str, list[str]]:
    """按行首记号把片段拆成三组归一化后的行。

    空行一律丢掉：diff 里新增/删除一个空行会渲染成孤零零一个 `+` / `-`，
    归一化后是空串；让它参与比对只会把连续匹配打断。

    Args:
        existing_code: 评论给的代码片段

    Returns:
        `{"+": [...], "-": [...], "": [...]}`，空串那组是模型抹掉了记号的
    """
    groups: dict[str, list[str]] = {marker: [] for marker in (*MARKER_SIDE, "")}
    for line in existing_code.splitlines():
        text = normalize(line)
        if text:
            groups[_marker(line)].append(text)
    return groups


def _search_plan(groups: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """按优先级列出 (侧, needle)。

    Args:
        groups: `_needles` 拆出来的三组

    Returns:
        搜索计划；片段不可用时为空列表
    """
    plus, minus, plain = groups["+"], groups["-"], groups[""]
    plan: list[tuple[str, list[str]]] = []
    if plus:
        plan.append((SIDE_RIGHT, plus))
    if plain:
        plan.append((SIDE_RIGHT, plain))
    if minus:
        plan.append((SIDE_LEFT, minus))
    if plain:
        plan.append((SIDE_LEFT, plain))
    return plan


def _side_lines(part: Hunk, side: str) -> list[tuple[int, str]]:
    """hunk 片里**某一侧**的改动行，带该侧的行号。

    RIGHT 取 `+` 行、行号从 ``new_start`` 起；LEFT 取 `-` 行、从 ``old_start``
    起。**两个计数器各走各的**：`+` 行只推进新版计数、`-` 行只推进旧版计数，
    而上下文行（空格开头）**两侧都推进**。三处漏一处都不行 —— 拿对面那侧的行
    推进本侧，改一行删一行的 hunk 会整体偏移；上下文行不推进，则从第一个上下文
    行起一直偏。两种偏移的结果都**不报错、不越界，读起来完全正常**，只是指向
    隔壁几行。

    上下文那一支是**防御性**的：我们的 diff 由 `loader.CONTEXT_LINES = 0` 生成
    （等价 `git diff -U0`），hunk 正文里没有上下文行，所以它目前走不到。留着是
    因为「本函数只对 `-U0` 成立」是个**隐式前提** —— 哪天 diff 来源变了，行号会
    从第一行起全错且没有任何信号（本项目最不能忍的那类）。对 `-U0` 两种写法等价，
    `tests/unit/test_locate.py` 分别钉住两条路径。

    Args:
        part: hunk 片
        side: LEFT / RIGHT

    Returns:
        (该侧行号, 归一化后的正文) 列表
    """
    right = side == SIDE_RIGHT
    old_no, new_no = part.old_start, part.new_start
    lines: list[tuple[int, str]] = []
    for line in part.content.splitlines():
        head = line[:1]
        if head == "+":
            if right:
                lines.append((new_no, normalize(line)))
            new_no += 1
        elif head == "-":
            if not right:
                lines.append((old_no, normalize(line)))
            old_no += 1
        elif head == " " or line == "":
            old_no += 1
            new_no += 1
        # `\ No newline at end of file` 不占号
    return lines


def _matches(needle: list[str], lines: list[tuple[int, str]]) -> list[tuple[int, int]]:
    """在一侧的行里找 needle，返回所有命中的 (起始行, 结束行)。

    两边都先去掉空行再比（理由同 `_needles`）。

    Args:
        needle: 归一化并去掉空行后的片段（**非空**，调用方保证）
        lines: 带行号的一侧内容

    Returns:
        命中的行号区间列表，按位置升序
    """
    surface = [(number, text) for number, text in lines if text]
    width = len(needle)
    return [
        (surface[index][0], surface[index + width - 1][0])
        for index in range(len(surface) - width + 1)
        if [text for _, text in surface[index : index + width]] == needle
    ]


def locate(hunk_id: str, existing_code: str, hunks: list[HunkWithSymbol]) -> Location:
    """把一条评论挂到行号上。

    先试评论自己那个 hunk（层 1），再试本组其它 hunk（层 2）；**每层里都按
    `_search_plan` 的顺序试完两侧**才算这层没命中。

    Args:
        hunk_id: 评论报告的 hunk 编号
        existing_code: 评论给的代码片段
        hunks: **本组**的全部 hunk

    Returns:
        定位结果
    """
    plan = _search_plan(_needles(existing_code))
    # schema 只保证「字段存在」，保证不了「内容有意义」：空串、纯空白、纯空行
    # 都可能在这里出现。它没有可定位的信息，但不丢评论 —— 后端收拢到聚合
    if not plan:
        return Location(hunk_id, 0, 0, LOCATE_FAILED, None, REASON_EMPTY_EXISTING_CODE)

    own = next((item for item in hunks if item.hunk_id == hunk_id), None)
    if own is not None:
        for side, needle in plan:
            hits = _matches(needle, _side_lines(own.part, side))
            if hits:
                # 同一 hunk 内多命中**取第一个**，不放弃：两个候选都在同一段
                # 改动里，取首个比把评论丢进聚合更划算（代价只是行号可能偏
                # 几行）。代价要靠数字盯住，所以 candidates 带出去计数
                return Location(hunk_id, hits[0][0], hits[0][1], LOCATE_HUNK, side, None, len(hits))

    # 层 1 没中才走层 2 —— 它已在最精确的那层找过两侧了
    for side, needle in plan:
        group_hits = [
            (item.hunk_id, start, end)
            for item in hunks
            for start, end in _matches(needle, _side_lines(item.part, side))
        ]
        if len(group_hits) == 1:
            hit_id, start, end = group_hits[0]
            return Location(hit_id, start, end, LOCATE_GROUP, side, None, 1)
        if len(group_hits) > 1:
            return Location(hunk_id, 0, 0, LOCATE_FAILED, None, REASON_MULTI_MATCH, len(group_hits))
    return Location(hunk_id, 0, 0, LOCATE_FAILED, None, REASON_NO_MATCH)


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
                    side=found.side,
                    locate_reason=found.reason,
                    suggestion_code=raw.suggestion_code,
                    rule_ref=raw.rule_ref,
                    finding_ref=raw.finding_ref,
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
            "（删除侧 %d 条，同 hunk 多命中取首个 %d 条）",
            group.group_id,
            total,
            levels[LOCATE_HUNK],
            levels[LOCATE_GROUP],
            levels[LOCATE_FAILED],
            (total - levels[LOCATE_FAILED]) / total * 100,
            sum(1 for item in comments if item.side == SIDE_LEFT),
            ambiguous,
        )
    if levels[LOCATE_FAILED]:
        logger.warning(
            "未定位 %d 条（%s），已在输出里单列、未丢弃",
            levels[LOCATE_FAILED],
            "、".join(f"{name} {count}" for name, count in sorted(reasons.items())),
        )
    return comments
