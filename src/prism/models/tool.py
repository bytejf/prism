"""工具相关的纯数据模型。

只描述数据结构，不含行为、不含可变状态。
运行时对象（HunkStatus / ToolContext）放在 core/tools/context.py。
"""

from dataclasses import dataclass, field


@dataclass
class CommentInput:
    """LLM 报告的评论输入（未定位）。

    Attributes:
        severity: 严重程度（high / medium / low）
        category: 问题类别
        content: 评论正文
        existing_code: 用于定位的代码片段
        suggestion_code: 建议的修改（可选）
        rule_ref: 引用的规则编号（可选，V1 不用）
    """

    severity: str
    category: str
    content: str
    existing_code: str
    suggestion_code: str | None = None
    rule_ref: str | None = None


@dataclass
class HunkReport:
    """单个 hunk 的报告。

    Attributes:
        hunk_id: hunk 编号（如 "H-01"）
        complete: 是否审查完毕
        comments: 评论列表（可为空）
    """

    hunk_id: str
    complete: bool
    comments: list[CommentInput] = field(default_factory=list)
