"""评论排序。

审查后流程的最后一站：把定位好的评论按 severity + 文件路径 + 行号排序，
产出可复现的顺序。补路径与定位在 `locate.py`，这里只管顺序。

排序必须可复现：同一输入两次运行输出顺序一致。
"""

import logging

from prism.models.comment import Comment

logger = logging.getLogger(__name__)

# severity 到排序权重的映射，权重越小越靠前
SEVERITY_RANK: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

# 未知 severity 的权重：比所有已知值都大，排在最后
UNKNOWN_SEVERITY_RANK = len(SEVERITY_RANK)


def sort_comments(comments: list[Comment]) -> list[Comment]:
    """按 severity + 文件路径排序评论。

    排序规则：
    1. severity 降序：high → medium → low，未知 severity 排最后
    2. 同 severity 按文件路径升序
    3. 同文件按 start_line 升序

    排序稳定：键相同的评论保持原始收集顺序。

    未定位的评论（``start_line == 0``）会排在同文件最前 —— 这不影响输出，
    渲染层会把它们单独挑出来放到末尾的聚合段里。

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


def _sort_key(comment: Comment) -> tuple[int, str, int]:
    """排序键：(severity 权重, 文件路径, 起始行号)。

    Args:
        comment: 单条评论

    Returns:
        排序键元组
    """
    rank = SEVERITY_RANK.get(comment.severity, UNKNOWN_SEVERITY_RANK)
    return rank, comment.path, comment.start_line
