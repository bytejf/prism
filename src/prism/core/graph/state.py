"""Main 循环的状态定义。"""

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from prism.core.tools.context import ToolContext
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# 最大轮次：max(MIN_ROUNDS, hunk 数 × ROUNDS_PER_HUNK)
MIN_ROUNDS = 30
ROUNDS_PER_HUNK = 5

# 连续无产出到这些轮次时注入提醒（文档要求提醒两次后退出）
NO_PROGRESS_REMIND_ROUNDS: tuple[int, ...] = (5, 7)

# 连续无产出达到该轮次则强制退出
NO_PROGRESS_EXIT_ROUNDS = 8

# 退出原因取值
EXIT_END_DONE = "end_done"
EXIT_END_DONE_WITH_GAPS = "end_done_with_gaps"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_NO_PROGRESS = "no_progress"


class MainState(TypedDict):
    """Main 循环的状态。

    Attributes:
        messages: 消息历史（LangGraph 自动追加）
        tool_context: 工具执行上下文（含 hunk 状态、评论）
        round_count: 当前轮次
        max_rounds: 最大轮次
        exit_reason: 退出原因（end_done / end_done_with_gaps / max_rounds / no_progress）
        group: 当前审查单元
        no_report_rounds: 连续无产出的轮次数
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_context: ToolContext
    round_count: int
    max_rounds: int
    exit_reason: str | None
    group: Group
    no_report_rounds: int


@dataclass
class MainLoopResult:
    """Main 循环的产出。

    与 MainState 的区别：这是给调用方（分组调度、输出渲染）用的最终结果，
    不含消息历史等过程数据。

    Attributes:
        group_id: 审查单元编号
        exit_reason: 退出原因
        round_count: 实际跑了多少轮
        comments: 收集到的原始评论
        pending_hunk_ids: 退出时仍未审查完毕的 hunk 编号
        round_trace: 逐轮摘要，用于打印运行日志
    """

    group_id: str
    exit_reason: str
    round_count: int
    comments: list[CommentInput] = field(default_factory=list)
    pending_hunk_ids: list[str] = field(default_factory=list)
    round_trace: list[str] = field(default_factory=list)


def max_rounds_for(hunk_count: int) -> int:
    """按 hunk 数计算最大轮次。

    Args:
        hunk_count: 本组 hunk 数

    Returns:
        max(MIN_ROUNDS, hunk 数 × ROUNDS_PER_HUNK)
    """
    return max(MIN_ROUNDS, hunk_count * ROUNDS_PER_HUNK)
