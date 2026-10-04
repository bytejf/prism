"""审查 Agent 的可变状态。

走 `create_agent(state_schema=...)`：hunk 审查进度、评论、各类计数都在这里，
它们在循环里不断变化，要跟着 checkpoint 走。

静态量在 `context.ReviewContext` 里，两者不要混。
"""

from dataclasses import dataclass, field
from typing import Annotated

from langchain.agents.middleware import AgentState

from prism.config.settings import settings
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# hunk 审查状态的取值
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

# 轮次公式、无产出提醒/退出阈值、end_done 拒绝上限**都是配置项**，
# 见 default.toml 的 [review]。这里不写第二份默认值。

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


def merge_hunk_status(
    left: dict[str, HunkStatus], right: dict[str, HunkStatus]
) -> dict[str, HunkStatus]:
    """合并 hunk 状态表 —— 并行工具调用的写入 reducer。

    必须在 `ReviewState` **之前**定义：reducer 是在类体求值时就被引用的，
    放到后面会 NameError。

    前提是工具写回的是**增量**（只含本次改动的 hunk，见 `report_hunks`）：
    并行调用各改各的 hunk，按 key 合并即可。写全量表不行 —— 两份全量快照里
    都记着对方那个 hunk 的旧值，后一份会把前一份的改动整个盖掉。

    Args:
        left: 已有状态表
        right: 本次写入的增量

    Returns:
        合并后的状态表
    """
    return {**left, **right}


class ReviewState(AgentState):
    """审查循环的状态。

    继承 LangChain 的 `AgentState`（它已提供
    `messages: Annotated[list[AnyMessage], add_messages]` 与 jump_to 等）。

    这里**故意写裸的 `AgentState`**，不写 `AgentState[Any]`：
    后者虽然是框架自己的写法（见 `_DefaultAgentState`），但 PyCharm 会报
    「TypedDict cannot inherit from a non-TypedDict base class」——
    它认不出「泛型 TypedDict 再被下标」这层。mypy strict 与 pyright 对两种写法
    都无异议，取不打扰 IDE 的那个。

    带 `Annotated` 的字段有 reducer —— 模型一轮里可以并行发多个工具调用，
    这些调用在**同一个 super-step** 里跑，返回的 `Command.update` 会被同时写入。
    LangGraph 对没有 reducer 的键只允许一步一写，两个 `report_hunks` 撞在一起
    就会抛 `InvalidUpdateError: At key 'hunk_status': Can receive only one value
    per step`。实测 21 轮里有 8 轮是并行调用，其中 1 轮正是这种情况。

    Attributes:
        hunk_status: hunk 状态表（hunk_id → HunkStatus）。并行写按 key 合并
        round_count: 已完成的模型调用轮次
        max_rounds: 最大轮次
        no_report_rounds: 连续无产出的轮次数
        end_done_count: 上次有进展以来 end_done 被调用的次数（拒绝机制用）
        end_done_mark: 上次重置时已完成的 hunk 数，用来判断「有没有新进展」
        group: 当前审查单元
    """

    hunk_status: Annotated[dict[str, HunkStatus], merge_hunk_status]
    round_count: int
    max_rounds: int
    no_report_rounds: int
    end_done_count: int
    end_done_mark: int
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
    return end_done_count > settings.review.max_rejections, pending


def max_rounds_for(hunk_count: int) -> int:
    """按 hunk 数计算最大轮次。

    两个数都来自配置的 `[review]`。

    Args:
        hunk_count: 本组 hunk 数

    Returns:
        max(min_rounds, hunk 数 × rounds_per_hunk)
    """
    return max(
        settings.review.min_rounds,
        hunk_count * settings.review.rounds_per_hunk,
    )


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

    if state["no_report_rounds"] >= settings.review.no_progress_exit_rounds:
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
