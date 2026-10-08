"""审查 Agent 的可变状态。

走 `create_agent(state_schema=...)`：hunk 审查进度、评论、各类计数都在这里，
它们在循环里不断变化，要跟着 checkpoint 走。

静态量在 `context.ReviewContext` 里，两者不要混。
"""

from dataclasses import dataclass, field
from typing import Annotated

from langchain.agents.middleware import AgentState

from prism.config.settings import settings
from prism.core.agent.budget import STOP_COST, STOP_TIME, STOP_TOOL_FAILURES
from prism.models.cost import UsageTotals
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# hunk 审查状态的取值。
#
# **只有两态**：报过 = done。曾经还有 `in_progress`（模型用 `complete=false` 表达
# 「报了部分结论、还没审完」）—— 那个字段后来整个删掉了：报过就没有终局语义，
# 同一个 hunk 随时可以再报、评论会累加，所以「审到一半」这个状态没有对应物
STATUS_PENDING = "pending"
STATUS_DONE = "done"

# 提醒阈值、收尾闸比例、end_done 拒绝上限**都是配置项**，见 default.toml 的
# [review]。这里不写第二份默认值。

# 退出原因取值
EXIT_END_DONE = "end_done"
EXIT_END_DONE_WITH_GAPS = "end_done_with_gaps"
# 撞上 recursion_limit：**不该发生**（说明前面几道闸一个都没生效），带缺口退出
EXIT_RECURSION = "recursion"
# 图抛了别的异常：同样带缺口退出，不把整条 review 带走
EXIT_ERROR = "error"
# 超预算：钱 / 时间。**这是主约束**（轮数只是兜底），两个原因分开记，
# 因为处置不同 —— 花钱超了要调价格表或提示词，时间超了要看单次延迟
EXIT_BUDGET_COST = "budget_cost"
EXIT_BUDGET_TIME = "budget_time"
# 工具连续/累计失败太多：**这是「不要把预算浪费在无效调用上」那道闸**，
# 与预算分开，因为处置不同 —— 它说明有东西在系统性失败，该去看日志而不是调预算
EXIT_TOOL_FAILURES = "tool_failures"


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
        round_count: 已完成的模型调用轮次（**只是观测数据**，不再参与退出判定）
        no_report_rounds: 连续无产出（没有 report_hunks / end_done）的轮次数。
            **只驱动软提醒**，不再参与退出判定 —— 硬的那道是收尾闸（预算口径）
        end_done_count: 上次有进展以来 end_done 被调用的次数（拒绝机制用）
        end_done_mark: 上次重置时已完成的 hunk 数，用来判断「有没有新进展」
        group: 当前审查单元
        usage: 累计用量（**按轮增量累加**，不能最后扫消息：压缩会把历史删掉）
        explore_closed: 探索工具是否已被停用（单组钱或时间用掉
            `settings.review.explore_close_ratio` 时为 True，**单调不回退**）。
            停用只留 `report_hunks` / `end_done` —— **不丢结论**：与其把已经花掉的
            预算连同查到的结论一起丢掉，不如逼它把已有结论报出来
        stop_reason: 停下来的原因（``cost`` / ``time`` / ``tool_failures``），没停时为 ""。
            **单键、先写者胜**：同一轮里有两个中间件都决定停时，谁先写谁的原因留下 ——
            顺序由 `build_review_agent` 的中间件列表定（预算在两个停手里排最前）
    """

    hunk_status: Annotated[dict[str, HunkStatus], merge_hunk_status]
    round_count: int
    no_report_rounds: int
    end_done_count: int
    end_done_mark: int
    group: Group
    usage: UsageTotals
    stop_reason: str
    explore_closed: bool


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


def exit_reason_of(state: ReviewState) -> str | None:
    """从**终态**读出退出原因，而不是一路传递。

    这样 `exit_reason` 不必进 state —— 它是终态的一个纯函数，好测也好读。
    顺序：**end_done 裁决 → 预算 → 工具失败**。

    顺序是有讲究的：审完了就是审完了（不该报成「因预算停止」而给用户一个假缺口）。

    Args:
        state: 循环结束时的状态

    Returns:
        退出原因；三个条件都不满足时返回 None（正常不该发生，由调用方报错）
    """
    pending = pending_ids(state["hunk_status"])
    # 顺序：**审完了就是审完了** —— 让预算把一个已完成的组标成「因预算停止」，
    # 是给用户一个假的缺口
    if state["end_done_count"] > 0 and end_done_verdict(state["end_done_count"], pending)[0]:
        return EXIT_END_DONE_WITH_GAPS if pending else EXIT_END_DONE

    if state["stop_reason"] == STOP_COST:
        return EXIT_BUDGET_COST
    if state["stop_reason"] == STOP_TIME:
        return EXIT_BUDGET_TIME
    if state["stop_reason"] == STOP_TOOL_FAILURES:
        return EXIT_TOOL_FAILURES

    return None


@dataclass
class MainLoopResult:
    """Main 循环的产出。

    与 ReviewState 的区别：这是给调用方（分组调度、输出渲染）用的最终结果，
    不含消息历史等过程数据。

    Attributes:
        group_id: 审查单元编号
        usage: 本组用量（含审查循环；反思的用量另加在调用方）
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
    usage: UsageTotals = field(default_factory=UsageTotals)

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
