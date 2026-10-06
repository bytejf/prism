"""审查 Agent 的组装与运行。

整个 Main 循环交给 LangChain v1 的 `create_agent`：

- **模型节点 / 工具节点 / 循环边** —— 框架内置，不再手写 StateGraph
- **轮次上限** —— 框架的 `ModelCallLimitMiddleware`
- **语义退出、进度快照、纯文本提醒** —— `middleware.py` 里的两个钩子
- **静态上下文** —— `context_schema`，按次传入
- **审查状态** —— `state_schema`，跟着 checkpoint 走

对照旧实现：5 个手写节点 + 3 个路由函数，全部消失。
"""

import logging
from collections.abc import Mapping
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from prism.config.settings import settings
from prism.core.agent.compression import Compressor
from prism.core.agent.context import ReviewContext
from prism.core.agent.middleware import ReviewProgress, review_pacing, tool_error_guard
from prism.core.agent.state import (
    STATUS_PENDING,
    HunkStatus,
    MainLoopResult,
    ReviewState,
    exit_reason_of,
    max_rounds_for,
    pending_ids,
)
from prism.core.guidance import templates as tpl
from prism.core.guidance.builder import build_diff_block, build_requirements
from prism.core.tools.code_search import code_search
from prism.core.tools.end_done import end_done
from prism.core.tools.file_read import file_read
from prism.core.tools.file_read_diff import file_read_diff
from prism.core.tools.report_hunks import report_hunks
from prism.models.compression import CompressionConfig
from prism.models.diff import Diff
from prism.models.grouping import Group
from prism.observability.llm_trace import ReviewTrace

logger = logging.getLogger(__name__)


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


def build_review_agent(
    model: BaseChatModel, max_rounds: int, compression: CompressionConfig | None = None
) -> Agent:
    """组装审查 Agent。

    Args:
        model: LangChain ChatModel
        max_rounds: 单组最大模型调用次数（框架的 ModelCallLimitMiddleware 执行）
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
                    # 轮次上限：框架自带，不用自己数
                    ModelCallLimitMiddleware(run_limit=max_rounds, exit_behavior="end"),
                    # 兜住工具抛出的异常，别让它炸掉整组审查。
                    # 放最内侧（最后一个）：ReviewTrace 在它外面，于是失败的那次
                    # 调用照样有一条带耗时的 tool 日志，日志形态不因失败而变
                    tool_error_guard,
                ],
            ),
            state_schema=ReviewState,
            context_schema=ReviewContext,
        ),
    )


def initial_state(group: Group, max_rounds: int) -> ReviewState:
    """构造审查循环的初始状态。

    提示词文本在 `guidance/templates.py` 里整块书写：
    system 段交给 `create_agent(system_prompt=...)`，
    user 段（diff 块 + 要求块）作为首条用户消息 —— **diff 在前、要求在后**。

    Args:
        group: 审查单元
        max_rounds: 单组最大模型调用次数

    Returns:
        初始 ReviewState
    """
    user_text = tpl.USER.format(
        diff_block=build_diff_block(group),
        requirements=build_requirements(),
    )
    return ReviewState(
        messages=[HumanMessage(content=user_text)],
        hunk_status={
            item.hunk_id: HunkStatus(hunk_id=item.hunk_id, status=STATUS_PENDING)
            for item in group.hunks
        },
        round_count=0,
        max_rounds=max_rounds,
        no_report_rounds=0,
        end_done_count=0,
        end_done_mark=0,
        group=group,
    )


def run_review_agent(
    group: Group,
    model: BaseChatModel,
    project_root: str,
    head_rev: str = "",
    compression: CompressionConfig | None = None,
    diff_index: Mapping[str, Diff] | None = None,
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

    Returns:
        循环产出（退出原因、评论、未审 hunk 等）

    Raises:
        ValueError: 图结束时读不出退出原因
    """
    max_rounds = max_rounds_for(len(group.hunks))
    logger.info(
        "开始审查 %s（%s，%d 个文件，%d 个 hunk，上限 %d 轮）",
        group.group_id,
        group.package_path,
        len(group.files),
        len(group.hunks),
        max_rounds,
    )

    agent = build_review_agent(model, max_rounds, compression or settings.compression)
    # 出参 cast：不传 version 时框架走 `version="v1"` 那个 overload，
    # 返回类型是 `dict[str, Any] | Any`，不认识 state_schema 里的自定义键。
    # 不 cast，下面 `final["hunk_status"]` 就退化成 Any，这一层的检查等于没加。
    final = cast(
        ReviewState,
        agent.invoke(
            initial_state(group, max_rounds),
            context=ReviewContext(
                project_root=project_root,
                head_rev=head_rev,
                diffs=diff_index or {},
            ),
            # LangGraph 的递归上限按 super-step 算。实测 create_agent 每轮约
            # 6.1 步（30 轮 = 182 步），旧手写图的 ×5 不够，取 ×7 留余量。
            # 真正的终止保证是 ModelCallLimitMiddleware，这里只是防图跑飞。
            config={"recursion_limit": max_rounds * 7 + 20},
        ),
    )

    exit_reason = exit_reason_of(final)
    if exit_reason is None:
        msg = "审查循环结束时读不出 exit_reason，退出判定有误"
        raise ValueError(msg)

    hunk_status = final["hunk_status"]
    # 按 hunk 编号键控返回：CommentInput 不带 hunk_id，
    # 输出阶段要靠这个键补文件路径
    hunk_comments = {
        hunk_id: list(status.comments) for hunk_id, status in sorted(hunk_status.items())
    }
    pending = pending_ids(hunk_status)
    total_comments = sum(len(items) for items in hunk_comments.values())
    logger.info(
        "结束审查 %s：exit_reason=%s，轮次 %d，评论 %d 条，未审 %d 个",
        group.group_id,
        exit_reason,
        final["round_count"],
        total_comments,
        len(pending),
    )
    return MainLoopResult(
        group_id=group.group_id,
        exit_reason=exit_reason,
        round_count=final["round_count"],
        hunk_comments=hunk_comments,
        pending_hunk_ids=pending,
    )
