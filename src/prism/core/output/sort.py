"""评论排序。

审查后流程的最后一站：把定位好的评论按 severity + 文件路径 + 行号排序，
产出可复现的顺序。补路径与定位在 `locate.py`，这里只管顺序。

排序必须可复现：同一输入两次运行输出顺序一致。
"""

import logging

from prism.models.comment import SIDE_LEFT, SIDE_RIGHT, Comment

logger = logging.getLogger(__name__)

# severity 到排序权重的映射，权重越小越靠前
SEVERITY_RANK: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

# 未知 severity 的权重：比所有已知值都大，排在最后
UNKNOWN_SEVERITY_RANK = len(SEVERITY_RANK)

# 侧在排序里的先后：新增侧在前。**必须进排序键** —— 左右两套行号是两个
# 坐标系，直接比大小没有意义（LEFT 7 与 RIGHT 10 比出来的顺序纯属巧合）。
# 分组展示反而清楚：先看「新版上的问题」，再看「删掉的那些」。
SIDE_RANK: dict[str, int] = {SIDE_RIGHT: 0, SIDE_LEFT: 1}

# 未定位（side 为 None）的权重：比两侧都大。它会进「未定位」段单独展示，
# 这里只需要一个确定的顺序
UNKNOWN_SIDE_RANK = len(SIDE_RANK)


def sort_comments(comments: list[Comment]) -> list[Comment]:
    """按 severity + 文件路径 + 侧排序评论。

    排序规则：
    1. severity 降序：high → medium → low，未知 severity 排最后
    2. 同 severity 按文件路径升序
    3. 同文件按侧分组：新增侧（被审查版本的坐标）在前，删除侧（基准版本的
       坐标）在后
    4. 同侧按 start_line 升序

    第 3 条是为了**不让两套坐标交错** —— 见 `SIDE_RANK` 的说明。

    排序稳定：键相同的评论保持原始收集顺序。未定位的评论（``side`` 为 None）
    会排在最后，渲染层会把它们单独挑出来放进末尾的聚合段。

    Args:
        comments: 原始评论列表

    Returns:
        排序后的评论列表（新列表，不修改原列表）
    """
    ordered = sorted(comments, key=_sort_key)
    logger.info("排序完成：%d 条评论", len(ordered))
    for severity in sorted(SEVERITY_RANK, key=lambda item: SEVERITY_RANK[item]):
        count = sum(1 for comment in ordered if comment.severity == severity)
        logger.debug("severity=%s：%d 条", severity, count)
    return ordered


def _sort_key(comment: Comment) -> tuple[int, str, int, int]:
    """排序键：(severity 权重, 文件路径, 侧权重, 起始行号)。

    Args:
        comment: 单条评论

    Returns:
        排序键元组
    """
    return (
        SEVERITY_RANK.get(comment.severity, UNKNOWN_SEVERITY_RANK),
        comment.path,
        SIDE_RANK.get(comment.side or "", UNKNOWN_SIDE_RANK),
        comment.start_line,
    )
