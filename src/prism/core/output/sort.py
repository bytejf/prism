"""评论整理与排序。

审查后流程的第一步：把各组的原始评论（CommentInput）补齐文件路径，
再按 severity + 文件路径排序，产出可复现的评论列表。

排序必须可复现：同一输入两次运行输出顺序一致。
"""

import logging

from prism.core.agent.state import MainLoopResult
from prism.core.diff.parser import UNKNOWN_PATH
from prism.models.comment import Comment
from prism.models.grouping import Group

logger = logging.getLogger(__name__)

# severity 到排序权重的映射，权重越小越靠前
SEVERITY_RANK: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

# 未知 severity 的权重：比所有已知值都大，排在最后
UNKNOWN_SEVERITY_RANK = len(SEVERITY_RANK)


def build_hunk_index(groups: list[Group]) -> dict[str, str]:
    """构建 hunk_id → 文件路径 的索引。

    CommentInput 只有 hunk_id 没有路径，输出前需要这张表补齐。

    Args:
        groups: 已分组的审查单元

    Returns:
        hunk 编号到文件路径的映射
    """
    return {item.hunk_id: item.file_path for group in groups for item in group.hunks}


def to_comments(results: list[MainLoopResult], groups: list[Group]) -> list[Comment]:
    """把各组原始评论转换成带文件路径的 Comment。

    保持各组、各评论的原始收集顺序；排序由 sort_comments 负责。
    hunk_id 查不到路径时（理论上不该发生）用占位符并告警。

    Args:
        results: 各组的循环产出
        groups: 已分组的审查单元

    Returns:
        Comment 列表，顺序为收集顺序
    """
    index = build_hunk_index(groups)
    comments: list[Comment] = []
    for result in results:
        for hunk_id, items in result.hunk_comments.items():
            path = index.get(hunk_id)
            if path is None:
                logger.warning("评论的 hunk_id 找不到文件路径：%s", hunk_id)
                path = UNKNOWN_PATH
            comments.extend(
                Comment(
                    path=path,
                    severity=raw.severity,
                    category=raw.category,
                    content=raw.content,
                    existing_code=raw.existing_code,
                    hunk_id=hunk_id,
                    suggestion_code=raw.suggestion_code,
                )
                for raw in items
            )

    logger.debug("整理评论：%d 条", len(comments))
    return comments


def sort_comments(comments: list[Comment]) -> list[Comment]:
    """按 severity + 文件路径排序评论。

    排序规则：
    1. severity 降序：high → medium → low，未知 severity 排最后
    2. 同 severity 按文件路径升序
    3. 同文件按 start_line 升序

    排序稳定：键相同的评论保持原始收集顺序。

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
    return (rank, comment.path, comment.start_line)
