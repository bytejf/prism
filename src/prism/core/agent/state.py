"""审查 Agent 的可变状态。

走 `create_agent(state_schema=...)`：hunk 审查进度、评论、各类计数都在这里，
它们在循环里不断变化，要跟着 checkpoint 走。

静态量在 `context.ReviewContext` 里，两者不要混。
"""

from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentState

from prism.models.grouping import Group
from prism.models.tool import CommentInput

# hunk 审查状态的取值
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

# 最大轮次：max(MIN_ROUNDS, hunk 数 × ROUNDS_PER_HUNK)
MIN_ROUNDS = 30
ROUNDS_PER_HUNK = 5

# 连续无产出到这些轮次时注入提醒（文档要求提醒两次后退出）
NO_PROGRESS_REMIND_ROUNDS: tuple[int, ...] = (5, 7)

# 连续无产出达到该轮次则强制退出
NO_PROGRESS_EXIT_ROUNDS = 8

# 允许强制接受 end_done 前最多拒绝几次
MAX_REJECTIONS = 1

# 退出原因取值
EXIT_END_DONE = "end_done"
EXIT_END_DONE_WITH_GAPS = "end_done_with_gaps"
EXIT_MAX_ROUNDS = "max_rounds"
EXIT_NO_PROGRESS = "no_progress"


@dataclass
class HunkStatus:
    """hunk 的审查状态。

    Attributes:
        hunk_id: hunk 编号
        status: 状态（pending / in_progress / done）
        comments: 已收集的评论
    """

    hunk_id: str
    status: str
    comments: list[CommentInput] = field(default_factory=list)


class ReviewState(AgentState[Any]):
    """审查循环的状态。

    继承 LangChain 的 `AgentState`（它已提供
    `messages: Annotated[list[AnyMessage], add_messages]` 与 jump_to 等）。

    写成 `AgentState[Any]` 而不是裸 `AgentState`：`AgentState` 是
    `TypedDict, Generic[ResponseT]`，留空会让类型参数变成未知，PyCharm 据此
    判定它接不上 `create_agent(state_schema=type[AgentState[ResponseT]])`。
    本状态不用 structured_response，`Any` 就是它的实际类型。

    Attributes:
        hunk_status: hunk 状态表（hunk_id → HunkStatus）
        all_comments: 所有已收集的评论
        round_count: 已完成的模型调用轮次
        max_rounds: 最大轮次
        no_report_rounds: 连续无产出的轮次数
        end_done_count: end_done 被调用的次数（用于拒绝机制）
        group: 当前审查单元
    """

    hunk_status: dict[str, HunkStatus]
    all_comments: list[CommentInput]
    round_count: int
    max_rounds: int
    no_report_rounds: int
    end_done_count: int
    group: Group


def pending_ids(hunk_status: dict[str, HunkStatus]) -> list[str]:
    """未审查完毕的 hunk 编号。

    Args:
        hunk_status: hunk 状态表

    Returns:
        编号列表，按字典序排序
    """
    return sorted(hunk_id for hunk_id, item in hunk_status.items() if item.status != STATUS_DONE)


def done_count(hunk_status: dict[str, HunkStatus]) -> int:
    """已审查完毕的 hunk 数。

    Args:
        hunk_status: hunk 状态表

    Returns:
        已标记 done 的 hunk 数量
    """
    return sum(1 for item in hunk_status.values() if item.status == STATUS_DONE)


def end_done_verdict(end_done_count: int, pending: list[str]) -> tuple[bool, list[str]]:
    """判定 end_done 是否接受结束。

    工具与退出判定共用本函数，避免规则在两处漂移。
    调用前 end_done 已经累加过 ``end_done_count``。

    Args:
        end_done_count: end_done 已被调用的次数
        pending: 未审的 hunk 编号列表

    Returns:
        (是否接受结束, 未审的 hunk 编号列表)
    """
    if not pending:
        return True, []
    return end_done_count > MAX_REJECTIONS, pending


def max_rounds_for(hunk_count: int) -> int:
    """按 hunk 数计算最大轮次。

    Args:
        hunk_count: 本组 hunk 数

    Returns:
        max(MIN_ROUNDS, hunk 数 × ROUNDS_PER_HUNK)
    """
    return max(MIN_ROUNDS, hunk_count * ROUNDS_PER_HUNK)


def exit_reason_of(state: ReviewState) -> str | None:
    """从**终态**读出退出原因，而不是一路传递。

    这样 `exit_reason` 不必进 state —— 它是终态的一个纯函数，好测也好读。
    优先级与旧的 check_exit 一致：end_done 裁决 → 轮次耗尽 → 连续无产出。

    Args:
        state: 循环结束时的状态

    Returns:
        退出原因；三个条件都不满足时返回 None（正常不该发生，由调用方报错）
    """
    pending = pending_ids(state["hunk_status"])
    if state["end_done_count"] > 0 and end_done_verdict(state["end_done_count"], pending)[0]:
        return EXIT_END_DONE_WITH_GAPS if pending else EXIT_END_DONE

    if state["round_count"] >= state["max_rounds"]:
        return EXIT_MAX_ROUNDS

    if state["no_report_rounds"] >= NO_PROGRESS_EXIT_ROUNDS:
        return EXIT_NO_PROGRESS

    return None


@dataclass
class MainLoopResult:
    """Main 循环的产出。

    与 ReviewState 的区别：这是给调用方（分组调度、输出渲染）用的最终结果，
    不含消息历史等过程数据。

    Attributes:
        group_id: 审查单元编号
        exit_reason: 退出原因
        round_count: 实际跑了多少轮
        hunk_comments: hunk 编号 → 该 hunk 收集到的评论（按编号排序）
        pending_hunk_ids: 退出时仍未审查完毕的 hunk 编号
    """

    group_id: str
    exit_reason: str
    round_count: int
    hunk_comments: dict[str, list[CommentInput]] = field(default_factory=dict)
    pending_hunk_ids: list[str] = field(default_factory=list)

    @property
    def comments(self) -> list[CommentInput]:
        """所有评论，按 hunk 编号顺序展开。

        按 hunk 编号键控而不是拍平成一个列表，是因为 CommentInput
        本身不带 hunk_id —— 只有携带它的容器才知道评论属于哪个 hunk
        （输出阶段需要靠它补文件路径）。

        Returns:
            评论列表
        """
        return [
            comment
            for hunk_id in sorted(self.hunk_comments)
            for comment in self.hunk_comments[hunk_id]
        ]
