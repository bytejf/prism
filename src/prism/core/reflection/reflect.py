"""评论反思：一组一次 LLM 调用做证伪，**只删不增**。

一组审查跑完、定位做完之后交给复核者。它只做一件事：找出那些被 diff
**直接推翻**的评论，去掉。不新增评论、不改写评论 —— 职责单一，所以它既不会
把召回率带高，也不会把精确率带偏。

## 为什么是「证伪」而不是「验证」

证伪只要找到一条反例，验证要对正确性下肯定判断。「我看到这行与你说法相反」
是可复核的；「我确认你说得对」很难达成，而且错了没人知道 —— 成本与可信度都
站证伪这一边。

## 为什么一组一次，而不是每个文件一次

一组一次调用：成本低（N 个文件 1 次调用），而且能看到跨文件的模式。代价是
**模式匹配偏置** —— 一次看到十几条评论时容易去找「模式」，所以提示词里明确
要求逐条独立判断。

## 信息边界：只看 diff + 评论 + 定位元数据

**不给**审查者的工具调用结果与推理过程。给了它会顺着前面那套逻辑「同意」，
复核就退化成自我背书。定位元数据不破坏这条边界 —— 它是代码算出来的事实，
不是上一位审查者的判断。

## 实现形态：走 `create_agent` 图（第二轮改的）

第一版是裸 `bind_tools(...).invoke(...)` + 自己解析校验，**不建图**。改成图是
为了两件事：

1. **模型要有纠错机会** —— 校验搬进工具（`tools.py`），问题作为工具结果回给它，
   它能在同一轮对话里改一版再报。裸调用的版本里，模型报了个不存在的编号，
   我们只能事后 fail-open，而它毫不知情
2. **日志与审查流程一致** —— 挂上 `ReviewTrace`，于是「工具定义（含完整参数
   schema）/ 每次工具调用的参数与耗时」是同一套输出。裸调用那条路径上
   **一个字都不打** —— `bind_tools` 只是把 schema 发给 API，本地不留痕

代价是要有自己的状态与中间件（`state.py` + 下面的 `ReflectionCompletion`），
但与 `core/agent/` 是同构的：读过审查循环的人能直接读这里。

## 为什么受理完就退出，不多调一次模型

`create_agent` 的默认退出条件是「模型不再调用工具」。工具一旦受理，如果不管它，
图会再调一次模型去听它说「好的，完成」—— 那是一次纯浪费的调用（结论已经在手
里了）。所以 `ReflectionCompletion` 在 `before_model` 里看到 `reported > 0` 就
`jump_to="end"`。审查那边用同一个手法为 `end_done` 省掉最后一次调用。

## 失败一律 fail-open，且**不重试**

出错、或模型始终交不出可受理的结论，就原样保留全部评论并打 WARNING。
不额外重试的理由：压缩失败会降级上下文质量且不为人知，值得原样再要一次；
复核失败是纯 no-op（评论一条不少）。**纠错靠的是工具把问题回给模型**，
那是同一轮对话内的事，不是我们事后再发一次请求。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from prism.core.agent.budget import usage_of
from prism.core.guidance import templates as tpl
from prism.core.guidance.builder import build_diff_block
from prism.core.reflection.state import (
    ReflectionContext,
    ReflectionState,
    Removal,
)
from prism.core.reflection.tools import report_incorrect_comments
from prism.models.comment import Comment
from prism.models.cost import UsageTotals
from prism.models.grouping import Group
from prism.models.reflection import PotentialMiss
from prism.observability.llm_trace import ReviewTrace, fields

logger = logging.getLogger(__name__)

# 评论在反思提示词里的编号：与 `H-01` 一套写法，只在这一次调用内有意义
COMMENT_ID_PREFIX = "C-"
COMMENT_ID_WIDTH = 2

# 模型调用上限：首次提交 + 一次纠错 + 一次余量。
#
# **不做成配置项**：它是「重试几次」的安全阀，不是可调行为；调小了会让纠错
# 半途而废且不报错（静默），调大只是多花钱。真需要时再进 default.toml。
REFLECTION_MAX_CALLS = 3

# 某个字段没有内容时在提示词里的占位
EMPTY_TEXT = "（无）"


@dataclass
class ReflectionOutcome:
    """一次反思的结果。

    Attributes:
        kept: 保留的评论。**失败时是原样的全部评论**（fail-open）
        removed: 被去掉的评论及其依据（审计用）
        misses: 疑似漏报信号。独立字段，不进 `kept`
        failure: 失败原因；没失败为 None
        usage: 本次反思的用量。**全局预算要把它算进去** —— 组级闸只看得到审查
            循环（它在反思之前就结束了），只算循环会漏掉每组一次的这笔钱
    """

    kept: list[Comment]
    removed: list[Removal] = field(default_factory=list)
    misses: list[PotentialMiss] = field(default_factory=list)
    failure: str | None = None
    usage: UsageTotals = field(default_factory=UsageTotals)


class ReflectionCompletion(AgentMiddleware[ReflectionState, ReflectionContext, Any]):
    """结论受理完就退出，不再多调一次模型。

    见模块 docstring 「为什么受理完就退出」。判断依据是状态里的 `reported`
    （受理次数），而不是「模型有没有再调工具」—— 后者要等它自己停下来，
    那正是要多花的那一次调用。
    """

    def before_model(
        self, state: ReflectionState, runtime: Runtime[ReflectionContext]
    ) -> dict[str, Any] | None:
        """受理过就跳去结束。

        Args:
            state: 当前状态
            runtime: 运行时（本中间件只读状态）

        Returns:
            需要退出时 ``{"jump_to": "end"}``；否则返回 None 不干预
        """
        if state.get("reported"):
            return {"jump_to": "end"}
        return None


# `can_jump_to` 的读取路径是 `getattr(middleware.__class__, "before_model").__can_jump_to__`
# （见 `langchain/agents/factory.py` 的 `_get_can_jump_to`）。装饰器写法把这个属性
# 挂在包装函数上，类式中间件取不到 —— 所以手动挂。与 `core/agent/middleware.py`
# 的 `ReviewProgress` 是同一处坑。
ReflectionCompletion.before_model.__can_jump_to__ = ["end"]  # type: ignore[attr-defined]

# 交给 `create_agent` 的图与实际类型不一样，这里手写一份更精确的 —— 理由同
# `core/agent/agent.py` 的 `Agent` 别名：框架的 overload 不把 state/context 串进返回类型。
ReflectionAgent = CompiledStateGraph[
    ReflectionState, ReflectionContext, ReflectionState, ReflectionState
]


def build_reflection_agent(model: BaseChatModel) -> ReflectionAgent:
    """组装反思 Agent。

    Args:
        model: 主模型。复核是**换一个视角读同一批材料**，不是换一个模型

    Returns:
        编译好的 agent 图
    """
    # 两处 cast，各是一处框架的签名缺口（与 core/agent/agent.py 同源）：
    # middleware 的类型参数只填了两个、返回类型不含上面那个精确别名。
    return cast(
        ReflectionAgent,
        create_agent(
            model=model,
            tools=[report_incorrect_comments],
            system_prompt=tpl.REFLECTION_SYSTEM,
            middleware=cast(
                list[AgentMiddleware[Any, Any, Any]],
                [
                    # 可观测性放最外层：它包住的是「真正发给模型的请求」。
                    # 每次反思新建一个实例 —— 它持有跨轮次的状态（已打过的消息
                    # 编号、轮次号），共用会串到下一组
                    ReviewTrace(),
                    ReflectionCompletion(),
                    ModelCallLimitMiddleware(run_limit=REFLECTION_MAX_CALLS, exit_behavior="end"),
                ],
            ),
            state_schema=ReflectionState,
            context_schema=ReflectionContext,
        ),
    )


def _locate_attr(comment: Comment) -> str:
    """渲染评论的定位状态。

    Args:
        comment: 未定位时带原因，形如 ``failed:no_match``

    Returns:
        `locate` 属性的值
    """
    if comment.locate_reason:
        return f"{comment.locate_level}:{comment.locate_reason}"
    return comment.locate_level


def _position_attr(comment: Comment) -> str:
    """渲染评论的位置属性（侧 + 行号）。

    未定位的评论没有位置，返回空串 —— 宁可少给一个属性，也不给一个
    「行号 0」这种要模型去猜的假坐标。

    Args:
        comment: 评论

    Returns:
        形如 `` side="LEFT" lines="7-9"``；未定位时为空串
    """
    if comment.side is None:
        return ""
    span = (
        str(comment.start_line)
        if comment.start_line == comment.end_line
        else f"{comment.start_line}-{comment.end_line}"
    )
    return f' side="{comment.side}" lines="{span}"'


def _render_comment(comment_id: str, comment: Comment) -> str:
    """渲染单条待复核的评论。

    Args:
        comment_id: 本次候选清单里的编号
        comment: 评论

    Returns:
        `<comment>` 标签文本
    """
    body = f"断言：\n{comment.content}\n\n引用代码：\n{comment.existing_code or EMPTY_TEXT}"
    return tpl.REFLECTION_COMMENT.format(
        comment_id=comment_id,
        path=comment.path,
        severity=comment.severity,
        category=comment.category,
        locate=_locate_attr(comment),
        position=_position_attr(comment),
        body=body,
    )


def reflect(model: BaseChatModel, group: Group, comments: list[Comment]) -> ReflectionOutcome:
    """复核一组评论，去掉被 diff 推翻的那些。

    Args:
        model: 主模型
        group: 审查单元，复核只看它的 diff
        comments: 已定位的评论

    Returns:
        反思结果；失败时 `failure` 非空、`kept` 是原样的全部评论
    """
    if not comments:
        return ReflectionOutcome(kept=[])

    numbered = [
        (f"{COMMENT_ID_PREFIX}{index:0{COMMENT_ID_WIDTH}d}", item)
        for index, item in enumerate(comments, start=1)
    ]
    logger.info("反思 · 开始 · %s · 候选 %d 条", group.group_id, len(comments))

    context = ReflectionContext(
        # 校验编号、补路径都靠这张表 —— 不让模型自己编路径
        comment_paths={item_id: item.path for item_id, item in numbered},
        hunk_paths={item.hunk_id: item.file_path for item in group.hunks},
    )
    diff_block = build_diff_block(group)
    comments_block = "\n".join(_render_comment(item_id, item) for item_id, item in numbered)
    prompt = tpl.REFLECTION_USER.format(diff=diff_block, comments=comments_block)
    logger.debug(
        "反思提示词：%s（diff %d 字符 + 评论 %d 条 %d 字符，共 %d 字符）",
        group.group_id,
        len(diff_block),
        len(numbered),
        len(comments_block),
        len(prompt),
    )
    initial = ReflectionState(
        messages=[HumanMessage(content=prompt)], removals={}, misses=[], reported=0
    )

    started = time.monotonic()
    try:
        final = cast(
            ReflectionState,
            build_reflection_agent(model).invoke(
                initial,
                context=context,
                # 与审查同一套余量：LangGraph 按 super-step 算，实测每轮约 6.1 步
                config={"recursion_limit": REFLECTION_MAX_CALLS * 7 + 20},
            ),
        )
    # 宽捕是**故意的**：复核失败必须 fail-open，不能让整组审查崩掉
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("反思 · 失败 · %s，已保留全部 %d 条", reason, len(comments))
        return ReflectionOutcome(kept=list(comments), failure=reason)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    # 反思不压缩，所以直接扫终态消息最省事（审查循环那边不能这么干 —— 压缩会把
    # 历史删掉，账会漏，见 core/agent/budget.py 的说明）
    usage = usage_of(final["messages"])

    # 受理过才算成功。**只看这个**，不看「模型有没有调过工具」：调过但每次都被
    # 退回的，与一次都没调的，对我们来说是同一件事 —— 手上没有可信结论
    if not final["reported"]:
        reason = "模型没交出可受理的结论（未调用工具，或反复被退回）"
        logger.warning("反思 · 失败 · %s，已保留全部 %d 条", reason, len(comments))
        return ReflectionOutcome(kept=list(comments), failure=reason, usage=usage)

    removals = sorted(final["removals"].values(), key=lambda item: item.comment_id)
    misses = list(final["misses"])
    kept = [item for item_id, item in numbered if item_id not in final["removals"]]
    logger.info(
        "反思 · 完成 · %s · 删除 %d / 保留 %d（删除率 %.0f%%）· 耗时 %dms\n%s",
        group.group_id,
        len(removals),
        len(kept),
        len(removals) / len(comments) * 100,
        elapsed_ms,
        fields(
            # 删评论是唯一有破坏性的动作，依据与指认全部留档
            (
                "removed",
                "\n".join(
                    f"{item.comment_id} {item.path} Ground {item.ground}：{item.reasoning}"
                    for item in removals
                ),
            ),
            (
                "misses",
                "\n".join(f"{item.hunk_id} {item.path}：{item.description}" for item in misses),
            ),
        ),
    )
    return ReflectionOutcome(kept=kept, removed=removals, misses=misses, usage=usage)
