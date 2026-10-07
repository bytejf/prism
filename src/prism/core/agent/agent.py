"""审查 Agent 的组装与运行。

整个 Main 循环交给 LangChain v1 的 `create_agent`：

- **模型节点 / 工具节点 / 循环边** —— 框架内置，不再手写 StateGraph
- **退出与止损** —— 语义退出（审完 / 无产出）、钱与时间预算、工具失败预算，
  全在 `middleware.py` 里；`recursion_limit` 是最后一道（防图失控，正常碰不到）
- **进度快照、纯文本提醒** —— 同样在这两个钩子里
- **静态上下文** —— `context_schema`，按次传入
- **审查状态** —— `state_schema`，跟着 checkpoint 走

对照旧实现：5 个手写节点 + 3 个路由函数，全部消失。
"""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from opentelemetry import trace

from prism.config.settings import settings
from prism.core.agent.budget import describe
from prism.core.agent.compression import Compressor
from prism.core.agent.context import ReviewContext
from prism.core.agent.middleware import (
    ReviewProgress,
    ToolFailureBudget,
    review_pacing,
    tool_call_dedup,
)
from prism.core.agent.state import (
    EXIT_ERROR,
    EXIT_RECURSION,
    STATUS_PENDING,
    HunkStatus,
    MainLoopResult,
    ReviewState,
    exit_reason_of,
    pending_ids,
)
from prism.core.grouping.packages import package_label
from prism.core.guidance import templates as tpl
from prism.core.guidance.builder import build_diff_block, build_requirements, build_static_block
from prism.core.guidance.tokens import estimate_tokens
from prism.core.tools.code_search import code_search
from prism.core.tools.end_done import end_done
from prism.core.tools.file_read import file_read
from prism.core.tools.file_read_diff import file_read_diff
from prism.core.tools.report_hunks import report_hunks
from prism.models.compression import CompressionConfig
from prism.models.cost import UsageTotals
from prism.models.diff import Diff
from prism.models.grouping import Group
from prism.models.static import StaticFinding
from prism.observability import semconv as sc
from prism.observability.llm_trace import ReviewTrace
from prism.observability.tracing import set_attributes

logger = logging.getLogger(__name__)

# 每模块自己的 tracer：没配 tracing 时是 no-op，埋点处不需要分支判断
tracer = trace.get_tracer(__name__)


def build_tools() -> list[BaseTool]:
    """构造 5 个审查工具。

    工具通过 `ToolRuntime` 取静态上下文与状态（框架注入，模型看不到该参数），
    因此不需要按运行时构造 —— 图构建时定一次即可。

    Returns:
        工具列表
    """
    return [file_read, file_read_diff, code_search, report_hunks, end_done]


# 交给 `create_agent` 的那个图和它的实际类型不一样，这里手写一份更精确的。
#
# `create_agent` 的四个 overload 都不把 `state_schema` / `context_schema`
# 串进返回类型：传 `ReviewState` + `ReviewContext`，mypy 推出来仍然是
# `CompiledStateGraph[AgentState[Any], Any, InputAgentState, OutputAgentState[Any]]`。
# 照抄那个签名，`invoke` 的 `context=` 就会退化成 `Any`，传错也不报错。
#
# 四个类型参数是 `CompiledStateGraph[StateT, ContextT, InputT, OutputT]`，
# 按我们的用法填：状态进、状态出，上下文是 ReviewContext。
Agent = CompiledStateGraph[ReviewState, ReviewContext, ReviewState, ReviewState]


def build_review_agent(model: BaseChatModel, compression: CompressionConfig | None = None) -> Agent:
    """组装审查 Agent。

    Args:
        model: LangChain ChatModel
        compression: 压缩配置；传 None 表示本组不压缩（测试里常用）

    Returns:
        编译好的 agent 图
    """
    # 两处 cast，各是一处框架的签名缺口：
    # 1. middleware：AgentMiddleware 声明了三个类型参数，而 create_agent 的
    #    middleware 参数只填了两个（漏了 ResponseT），框架自带的
    #    ModelCallLimitMiddleware 同样过不了这个检查。
    # 2. 整体：框架推出来的返回类型不含上面 Agent 别名里的精确参数。
    return cast(
        Agent,
        create_agent(
            model=model,
            tools=build_tools(),
            system_prompt=tpl.SYSTEM,
            middleware=cast(
                list[AgentMiddleware[Any, Any, Any]],
                [
                    # 可观测性放最外层：它包住的是「真正发给模型的请求」，
                    # 排在语义中间件后面才能看到它们改完之后的最终形态。
                    # 每组审查新建一个实例 —— 它持有跨轮次的状态（已打过的
                    # 消息数、上一轮输入 token），共用实例会串到下一组去
                    ReviewTrace(),
                    # 压缩器同理：压缩次数、连续失败数、熔断标记都是每组的，
                    # 共用一个实例会把上一组的熔断带到下一组
                    ReviewProgress(
                        Compressor(model, compression) if compression is not None else None
                    ),
                    review_pacing,
                    # 兜住工具异常 + 给「失败」本身设预算。
                    # 放最内侧（倒数第二个）：ReviewTrace 在它外面，于是失败的那次
                    # 调用照样有一条带耗时的 tool 日志，日志形态不因失败而变
                    ToolFailureBudget(),
                    # 参数完全相同的只读调用不重复执行，挡在真正的执行之前。
                    # 放在兜底**里面**：这一步自己的 bug 也会被兜成一条工具结果，
                    # 而不是把整组带走
                    tool_call_dedup,
                ],
            ),
            state_schema=ReviewState,
            context_schema=ReviewContext,
        ),
    )


def initial_state(group: Group, static_findings: Sequence[StaticFinding] = ()) -> ReviewState:
    """构造审查循环的初始状态。

    提示词文本在 `guidance/templates.py` 里整块书写：
    system 段交给 `create_agent(system_prompt=...)`，
    user 段（diff 块 + 静态发现块 + 要求块）作为首条用户消息 ——
    **数据在前、线索居中、要求在后**。

    Args:
        group: 审查单元
        static_findings: 注入本组的静态发现（已按 token 预算裁过），
            由 `static.analyzer.select_for_group` 按文件挑出（额度在分组那一步已经算过）

    Returns:
        初始 ReviewState
    """
    diff_block = build_diff_block(group, static_findings)
    static_block = build_static_block(static_findings)
    requirements = build_requirements()
    user_text = tpl.USER.format(
        diff_block=diff_block,
        static_block=static_block,
        requirements=requirements,
    )
    logger.debug(
        "审查提示词：%s（%d 个 hunk / %d 个文件，diff %d 字符 + 静态发现 %d 条，共 %d 字符）",
        group.group_id,
        len(group.hunks),
        len(group.files),
        len(diff_block),
        len(static_findings),
        len(user_text),
    )
    return ReviewState(
        messages=[HumanMessage(content=user_text)],
        hunk_status={
            item.hunk_id: HunkStatus(hunk_id=item.hunk_id, status=STATUS_PENDING)
            for item in group.hunks
        },
        round_count=0,
        no_report_rounds=0,
        end_done_count=0,
        end_done_mark=0,
        group=group,
        usage=UsageTotals(),
        stop_reason="",
    )


def run_review_agent(
    group: Group,
    model: BaseChatModel,
    project_root: str,
    head_rev: str = "",
    compression: CompressionConfig | None = None,
    diff_index: Mapping[str, Diff] | None = None,
    static_findings: Sequence[StaticFinding] = (),
) -> MainLoopResult:
    """跑一个审查单元的循环。

    Args:
        group: 审查单元
        model: LangChain ChatModel
        project_root: 项目根目录
        head_rev: 被审查的版本，工具据此读取源码而不是读工作区
        compression: 压缩配置。**默认开启**，用配置文件的 [compression]；
            1M 窗口下几乎不会触发，它是保险不是常规操作。传
            ``dataclasses.replace(settings.compression, context_window=...)``
            收小窗口就能在测试里逼出压缩
        diff_index: `路径 → Diff` 索引，供 `file_read_diff` 查跨文件改动。
            不传时该工具只能回「不可用」—— 只有测试会这样传
        static_findings: 注入本组的静态发现。**由调用方先算好**（挑文件 + 按预算
            截断），这里只管渲染 —— 后置校验要用同一批编号，各算一次就会漂

    Returns:
        循环产出（退出原因、评论、未审 hunk 等）

    Raises:
        ValueError: 图结束时读不出退出原因
    """
    logger.info(
        "开始审查 %s（%s，%d 个文件，%d 个 hunk）",
        group.group_id,
        package_label(group),
        len(group.files),
        len(group.hunks),
    )

    started = time.monotonic()
    agent = build_review_agent(model, compression or settings.compression)
    initial = initial_state(group, static_findings)
    with tracer.start_as_current_span("group_review") as span:
        set_attributes(
            span,
            {
                sc.PRISM_GROUP_ID: group.group_id,
                sc.PRISM_PACKAGE_LABEL: package_label(group),
                sc.PRISM_HUNK_COUNT: len(group.hunks),
                # 起手提示词的实际大小（不是估算模型：量的是真渲染出来的那份文本）
                sc.PRISM_TOKEN_ESTIMATE: estimate_tokens(str(initial["messages"][0].content)),
            },
        )
        final, crashed = _run_graph(agent, initial, group, project_root, head_rev, diff_index or {})
        set_attributes(
            span,
            {
                sc.PRISM_EXIT_REASON: crashed or exit_reason_of(final),
                sc.PRISM_ROUND_COUNT: final["round_count"],
                sc.PRISM_PENDING_HUNKS: len(pending_ids(final["hunk_status"])),
            },
        )

    if crashed is None:
        exit_reason = exit_reason_of(final)
        if exit_reason is None:
            msg = "审查循环结束时读不出 exit_reason，退出判定有误"
            raise ValueError(msg)
    else:
        exit_reason = crashed

    hunk_status = final["hunk_status"]
    # 按 hunk 编号键控返回：CommentInput 不带 hunk_id，
    # 输出阶段要靠这个键补文件路径
    hunk_comments = {
        hunk_id: list(status.comments) for hunk_id, status in sorted(hunk_status.items())
    }
    pending = pending_ids(hunk_status)
    total_comments = sum(len(items) for items in hunk_comments.values())
    # 耗时与用量都要打：预算闸是按这两个量判的，排查「这组为什么停」时
    # 第一个要看的就是它们（只看轮次看不出花了多快）
    logger.info(
        "结束审查 %s：exit_reason=%s，轮次 %d，评论 %d 条，未审 %d 个，耗时 %.1f 秒，用量 %s",
        group.group_id,
        exit_reason,
        final["round_count"],
        total_comments,
        len(pending),
        time.monotonic() - started,
        describe(final["usage"], settings.cost),
    )
    return MainLoopResult(
        group_id=group.group_id,
        exit_reason=exit_reason,
        round_count=final["round_count"],
        hunk_comments=hunk_comments,
        pending_hunk_ids=pending,
        usage=final["usage"],
    )


def _run_graph(
    agent: Agent,
    initial: ReviewState,
    group: Group,
    project_root: str,
    head_rev: str,
    diff_index: Mapping[str, Diff],
) -> tuple[ReviewState, str | None]:
    """跑一遍图，返回**最后看到的状态**与「异常退出原因」（正常结束时为 None）。

    用 `stream(stream_mode="values")` 而不是 `invoke`：每一步都拿到一份完整状态，
    于是**图崩了也能退回最后一个好状态**。这不是锦上添花 —— 渲染在整条 review 的
    最后一步，一旦异常穿出去，已经审出来的评论会跟它一起没（`ToolFailureBudget`
    修的是工具那一层，这是它外面那一层）。

    两类异常都在这里收口：

    - `GraphRecursionError`：`recursion_limit` 到了。**这是「不该发生」**：它响说明
      前面几道闸（语义退出、预算、模型调用上限）一个都没生效，套的是我们自己的
      bug，所以记 ERROR 带堆栈，而不是当普通退出
    - 其余异常：同样收口成「带缺口退出」。**不吞** —— ERROR + 完整堆栈照记，
      只是不让它把整条 review 带走

    Args:
        agent: 编译好的审查图
        initial: 初始状态
        group: 审查单元（只用于日志）
        project_root: 项目根目录
        head_rev: 被审查的版本
        diff_index: `路径 → Diff` 索引

    Returns:
        (最后一份状态, 退出原因码)；正常结束时第二个元素为 None
    """
    context = ReviewContext(project_root=project_root, head_rev=head_rev, diffs=diff_index)
    # LangGraph 的递归上限按 super-step 算（实测 create_agent 每轮约 6.1 步）。
    # **这是最后一道，正常永远碰不到** —— 终止保证来自语义退出与预算。取一个宽松的
    # 绝对数而不是按轮数乘：乘出来的余量只有 15%（6.1 vs 7），框架或我们多加一个
    # 钩子就会让它反过来先响，而它响是异常路径。
    #
    # 标注成 RunnableConfig 不是装饰：裸 dict 字面量会被推成 `dict[str, int]`，
    # `stream` 的四个 overload 会全都不匹配（`invoke` 那边碰巧不挑）。这也是框架
    # 的签名缺口，记在 17 号台账里
    config: RunnableConfig = {"recursion_limit": settings.review.max_super_steps}
    final: ReviewState = initial
    try:
        for chunk in agent.stream(initial, context=context, config=config, stream_mode="values"):
            final = cast(ReviewState, chunk)
    except GraphRecursionError as exc:
        logger.error(
            "审查 %s 撞上 recursion_limit（这是不该发生的一类，说明前几道闸没生效）",
            group.group_id,
            exc_info=exc,
        )
        return final, EXIT_RECURSION
    except Exception as exc:
        logger.error("审查 %s 抛异常，按带缺口退出处理", group.group_id, exc_info=exc)
        return final, EXIT_ERROR
    return final, None
