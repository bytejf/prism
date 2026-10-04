"""评论相关的数据模型。"""

from dataclasses import dataclass

# 定位层级。值本身是**契约不是旋钮**（消费方按它分支），所以写死在代码里，
# 不进 default.toml —— 见 CLAUDE.md「什么进文件、什么不进」。
LOCATE_HUNK = "hunk"  # 在评论自己那个 hunk 内命中
LOCATE_GROUP = "group"  # 在本组其它 hunk 内命中（评论报的编号是错的）
LOCATE_FAILED = "failed"  # 未定位，收拢到聚合

# 未定位原因，仅 LOCATE_FAILED 时非空
REASON_NO_MATCH = "no_match"  # existing_code 在本组改动行里一次都没出现
REASON_MULTI_MATCH = "multi_match"  # 出现多处，放弃
REASON_EMPTY_EXISTING_CODE = "empty_existing_code"  # 模型没给可用的片段


@dataclass
class Comment:
    """单条评论（最终输出形态）。

    与 CommentInput 的区别：Comment 是补齐了文件路径与行号、准备排序输出的
    形态，CommentInput 是 LLM 报告时的原始形态（不含路径，只带 hunk_id）。

    Attributes:
        path: 文件路径
        severity: 严重程度（high / medium / low）
        category: 问题类别
        content: 评论正文
        existing_code: 用于定位的代码片段
        hunk_id: 所属 hunk 编号。**组内命中时是命中的那个**，不是模型报的那个
            —— path 由它查出，两者必须同源（见 core/output/locate.py）
        locate_level: 定位层级（hunk / group / failed）
        locate_reason: 未定位原因，仅 locate_level 为 failed 时非空
        suggestion_code: 建议的修改（可选）
        start_line: 起始行号（**被审查版本的坐标**）；未定位为 0
        end_line: 结束行号（同上）；未定位为 0
    """

    path: str
    severity: str
    category: str
    content: str
    existing_code: str
    hunk_id: str
    locate_level: str
    locate_reason: str | None = None
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
        comments_count: 评论数
        duration_seconds: 耗时（秒）
    """

    base_branch: str
    head_sha: str
    files_count: int
    comments_count: int
    duration_seconds: float
