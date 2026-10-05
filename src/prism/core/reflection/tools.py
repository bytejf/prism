"""反思工具：把复核结论**作为一次工具调用**交回来。

## 为什么是真工具，而不是「只当输出格式的 schema」

第一版是 `convert_to_openai_tool(Verdict)` + 裸 `invoke`：只借工具的**形状**
当输出格式，没有执行体。那个做法有个硬伤 —— **模型没有任何纠错机会**：它报的
`C-09` 不存在、或者说不出理由，我们只能在事后 fail-open（一条都不删），而它
毫不知情。做成实体工具之后，校验结果可以作为**工具结果**回给它，它能在同一轮
对话里改一版再报。

顺带还解决了第二件事：走 `ReviewTrace` 中间件，于是「工具定义（含完整参数
schema）/ 每次工具调用的参数与耗时」这套日志与审查流程**完全一致**
（见 `observability/llm_trace.py`）—— 之前那条路径上一个字都不打。

## 校验的两档处理，是有意区分的

- **`removals` 有一处不合法 → 整份退回**：不写任何状态、把问题回给模型重报。
  为什么不做「接受好的、退回坏的」：半份结论进了状态之后，「失败即全部保留」
  这条语义就说不清了（已经删了一部分，还算失败吗）
- **`misses` 有一处不合法 → 丢掉那一条，其余照收**：漏报是**提示不是评论**，
  不能因为一条 hunk 编号写错，把整份删除结论也按住。丢掉的那条会写进工具结果

## 形态约定

与审查的四个工具一致：`@tool(parse_docstring=True, description="…")`，
**不给 `args_schema`** —— 给了它 `runtime: ToolRuntime[...]` 的注入就失效
（见 CLAUDE.md 的「框架缺口」表）。嵌套结构靠 Pydantic 模型的类型注解表达。
"""

import logging
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from prism.core.reflection.state import ReflectionContext, ReflectionState, Removal
from prism.models.reflection import PotentialMiss

logger = logging.getLogger(__name__)

TOOL_NAME = "report_incorrect_comments"


class RemovalArg(BaseModel):
    """报告一条被 diff 推翻的评论。"""

    comment_id: str = Field(description="评论编号，必须来自待复核清单，如 C-01")
    ground: Literal["A", "B"] = Field(
        description="A=评论描述的代码不在 diff 里；B=diff 里某一行字面地推翻了它的核心论断"
    )
    reasoning: str = Field(description="指认**具体哪一处** diff 推翻了它；指不出来就不要报这一条")


class MissArg(BaseModel):
    """报告一处疑似漏报。"""

    hunk_id: str = Field(description="相关 hunk 的编号，如 H-03")
    description: str = Field(description="diff 里哪处改动疑似没被任何评论覆盖")


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py / CLAUDE.md。
#
# 描述里明说「结束时必须调用一次」：模型不调工具 = 图按默认条件退出 = 我们读不到
# 结论，只能按失败处理。空 removals 是**合法**提交，等于「全部保留」。
@tool(
    parse_docstring=True,
    description=(
        "提交复核结论：报告被 diff 推翻、要去掉的评论。\n"
        "**复核结束时必须调用一次**（哪怕一条都不删，也要调用并传空的 removals，"
        "那表示全部保留）。\n"
        "如果本次调用被拒绝，工具会把问题告诉你 —— 改正后重新调用一次完整结论。"
    ),
)
def report_incorrect_comments(
    removals: list[RemovalArg],
    misses: list[MissArg],
    runtime: ToolRuntime[ReflectionContext, ReflectionState],
) -> Command[Any]:
    """提交复核结论。

    `removals` / `misses` 的说明走下面的 `Args:`，**嵌套字段的说明走
    `RemovalArg` / `MissArg` 的 `Field`** —— 嵌套结构只能由 Pydantic 模型表达。
    `runtime` 由框架注入、模型看不到，会被自动排除。

    Args:
        removals: 判定要去掉的评论；一条都没有就传空列表（等于全部保留）
        misses: 疑似漏报；没有就留空
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        状态更新（受理的删除 / 漏报 + 受理次数 + 给模型的回执）
    """
    context = runtime.context
    accepted: dict[str, Removal] = {}
    problems: list[str] = []

    for item in removals:
        comment_id = item.comment_id.strip()
        path = context.comment_paths.get(comment_id)
        if path is None:
            problems.append(
                f"{item.comment_id!r} 不在本次候选清单里"
                f"（清单：{', '.join(sorted(context.comment_paths))}）"
            )
            continue
        if not item.reasoning.strip():
            problems.append(
                f"{comment_id} 没给出理由 —— 指不出具体哪一处 diff 就不能删它，"
                "要么补上指认，要么别报这一条"
            )
            continue
        # 同一编号报两次就是覆盖：同一个 key，不存在歧义
        accepted[comment_id] = Removal(
            comment_id=comment_id,
            path=path,
            ground=item.ground,
            reasoning=item.reasoning.strip(),
        )

    if problems:
        # 整份退回：不写状态，让模型拿一个**完整**的结论重报
        logger.warning("反思 · 工具退回 · %s", "；".join(problems))
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "本次提交**未被受理**，问题如下：\n"
                            + "\n".join(f"- {text}" for text in problems)
                            + "\n\n请改正后重新调用一次，报出**完整的**结论。"
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    resolved, dropped = _resolve_misses(context, misses)
    logger.info(
        "反思 · 工具受理 · 删除 %d 条，漏报 %d 条（丢弃 %d 条）",
        len(accepted),
        len(resolved),
        len(dropped),
    )
    kept_count = len(context.comment_paths) - len(accepted)
    parts = [f"复核结论已受理：要去掉 {len(accepted)} 条，保留 {kept_count} 条"]
    if accepted:
        parts.append(
            "去掉的："
            + "、".join(f"{cid}（Ground {item.ground}）" for cid, item in accepted.items())
        )
    if dropped:
        parts.append("⚠️ 以下漏报的 hunk 编号不存在，已丢弃：" + "；".join(dropped))
    return Command(
        update={
            "removals": accepted,
            "misses": resolved,
            "reported": 1,
            "messages": [ToolMessage(content="\n".join(parts), tool_call_id=runtime.tool_call_id)],
        }
    )


def _resolve_misses(
    context: ReflectionContext, misses: list[MissArg]
) -> tuple[list[PotentialMiss], list[str]]:
    """整理漏报信号：补路径、丢掉不可用的。

    路径由 `hunk_id` 查出，**不让模型自己写** —— 同一个坐标只留一个来源。

    Args:
        context: 静态上下文（提供 hunk 表）
        misses: 模型报的漏报信号

    Returns:
        (可用的漏报, 被丢弃的原因描述)
    """
    resolved: list[PotentialMiss] = []
    dropped: list[str] = []
    for item in misses:
        hunk_id = item.hunk_id.strip()
        path = context.hunk_paths.get(hunk_id)
        if path is None:
            dropped.append(f"{item.hunk_id!r}")
            continue
        if not item.description.strip():
            # 没有描述等于没有信息，连丢弃原因都不值得记
            continue
        resolved.append(
            PotentialMiss(hunk_id=hunk_id, path=path, description=item.description.strip())
        )
    return resolved, dropped
