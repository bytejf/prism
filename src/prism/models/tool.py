"""工具相关的纯数据模型。

只描述数据结构，不含行为、不含可变状态。
运行时对象（HunkStatus / ReviewState）放在 core/agent/state.py。
"""

from dataclasses import dataclass


@dataclass
class CommentInput:
    """LLM 报告的评论输入（未定位）。

    Attributes:
        severity: 严重程度（high / medium / low）
        category: 问题类别
        content: 评论正文
        existing_code: 用于定位的代码片段
        suggestion_code: 建议的修改（可选）
    """

    severity: str
    category: str
    content: str
    existing_code: str
    suggestion_code: str | None = None
