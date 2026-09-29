"""评论相关的数据模型。"""

from dataclasses import dataclass


@dataclass
class Comment:
    """单条评论（最终输出形态）。

    与 CommentInput 的区别：Comment 是补齐了文件路径、准备排序输出的形态，
    CommentInput 是 LLM 报告时的原始形态（不含路径，只带 hunk_id）。

    Attributes:
        path: 文件路径
        severity: 严重程度（high / medium / low）
        category: 问题类别
        content: 评论正文
        existing_code: 用于定位的代码片段
        hunk_id: 所属 hunk 编号
        suggestion_code: 建议的修改（可选）
        start_line: 起始行号（V1 未定位，默认 0）
        end_line: 结束行号（V1 未定位，默认 0）
    """

    path: str
    severity: str
    category: str
    content: str
    existing_code: str
    hunk_id: str
    suggestion_code: str | None = None
    start_line: int = 0
    end_line: int = 0


@dataclass
class ReviewStats:
    """审查统计。

    Attributes:
        base_branch: base 分支名
        head_sha: head SHA
        files_count: 文件数
        hunks_count: hunk 数
        comments_count: 评论数
        duration_seconds: 耗时
    """

    base_branch: str
    head_sha: str
    files_count: int
    hunks_count: int
    comments_count: int
    duration_seconds: float
