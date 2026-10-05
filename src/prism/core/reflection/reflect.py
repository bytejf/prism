"""评论反思：一次 LLM 调用完成证伪，**只删不增**。

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

## 为什么不用 `with_structured_output`

**实测**：思考模式下用不了。

    with_structured_output(Verdict)
        → 400 Thinking mode does not support this tool_choice
    with_structured_output(..., method="json_schema")
        → 同上
    with_structured_output(..., method="json_mode")
        → 400（提示词里带 "json" 能修，但它只保证合法 JSON、不保证 schema）
    bind_tools(tool_choice="auto")
        → ✓ 正常，args 已被 SDK 解析成 dict

`with_structured_output` 的 function_calling 路线靠**强制** `tool_choice`，而
DeepSeek 思考模式不接受强制 —— 审查循环能用工具，是因为它从不强制（`auto`）。
所以这里也走 `bind_tools`，回参自己用 Pydantic 校验：拿到的东西一样是结构化
的，只是少了框架那层壳。

## 为什么只绑一个工具

OCR 绑两个（`report_incorrect_comments` + `approve_all_comments`）。我们绑一个，
`removals` 传空列表就等于「全部保留」。

**这个取舍实测过**（同一个「全部保留」的用例，两种形态各跑 3 次）：

    两个工具：approve_all_comments ×3
    单工具：  report_incorrect_comments {removals: []} ×3

担心的「只给一个『报告要删的』工具，模型会为了有东西可报而硬找一条」**没有
出现**，两种形态都稳定给出「全保留」。既然行为一致，取代码少的那条：两个
schema 变一个、解析分支减半，漏报字段也不必在两处各写一遍。（n=3 不是强结论，
只够支撑「先按简单的来」。）

## 信息边界：只看 diff + 评论 + 定位元数据

**不给**审查者的工具调用结果与推理过程。给了它会顺着前面那套逻辑「同意」，
复核就退化成自我背书。定位元数据不破坏这条边界 —— 它是代码算出来的事实，
不是上一位审查者的判断。

## 失败一律 fail-open，且**不重试**

出错就原样保留全部评论并打 WARNING。不重试的理由：压缩失败会降级上下文质量
且不为人知，值得原样再要一次；复核失败是纯 no-op（评论一条不少），而重试会让
每次格式错误的代价翻倍。§6.2 的失败率目标（≤5%）先测出来再说 —— 不够再加。

## 绝不因为模型的输出而删错评论

三道闸门，都在代码里而不是只在提示词里：

1. **编号必须在本次候选清单里** —— 对不上就忽略。猜错一次就是删掉一条无关评论
2. **理由不能为空** —— 提示词要求「指不出具体哪一处 diff 就保留」，空理由是
   这条规则在 schema 层面唯一可判定的形态
3. **漏报的 hunk 编号必须存在** —— 锚不到 diff 上的漏报提示没有可用之处，
   而它出现本身就说明这次输出不可靠
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from prism.core.guidance import templates as tpl
from prism.core.guidance.builder import build_diff_block
from prism.models.comment import Comment
from prism.models.grouping import Group
from prism.models.reflection import PotentialMiss
from prism.observability.llm_trace import fields

logger = logging.getLogger(__name__)

# 评论在反思提示词里的编号：与 `H-01` 一套写法，只在这一次调用内有意义
COMMENT_ID_PREFIX = "C-"
COMMENT_ID_WIDTH = 2

# 工具名。`convert_to_openai_tool` 取的是 Pydantic 类名（CamelCase），
# 这里改回下划线风格：与 OCR 的 `report_incorrect_comments` 一致，日志也好读
TOOL_NAME = "report_incorrect_comments"

# 某个字段没有内容时在提示词里的占位
EMPTY_TEXT = "（无）"


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


class Verdict(BaseModel):
    """提交复核结论。

    **没有评论被 diff 推翻时 removals 传空列表** —— 那等于全部保留。
    """

    removals: list[RemovalArg] = Field(
        default_factory=list, description="被 diff 推翻、要去掉的评论；一条都没有就传空列表"
    )
    misses: list[MissArg] = Field(default_factory=list, description="疑似漏报；没有就留空")


@dataclass(frozen=True)
class Removal:
    """一条被去掉的评论。

    留一份完整的记录（编号、路径、依据、指认）—— 删评论是本流程里唯一有破坏性的
    动作，它必须可审计。

    Attributes:
        comment_id: 本次候选清单里的编号
        path: 该评论的文件路径
        ground: 删除依据（A / B）
        reasoning: 模型指认的那一处 diff
    """

    comment_id: str
    path: str
    ground: str
    reasoning: str


@dataclass
class ReflectionOutcome:
    """一次反思的结果。

    Attributes:
        kept: 保留的评论。**失败时是原样的全部评论**（fail-open）
        removed: 被去掉的评论及其依据（审计用）
        misses: 疑似漏报信号。独立字段，不进 `kept`
        failure: 失败原因；没失败为 None
    """

    kept: list[Comment]
    removed: list[Removal] = field(default_factory=list)
    misses: list[PotentialMiss] = field(default_factory=list)
    failure: str | None = None


def _verdict_tool() -> dict[str, Any]:
    """提交结论的工具 schema。

    用 `convert_to_openai_tool` 直接把 Pydantic 模型转成工具定义，**不写一个
    永不执行的 `@tool` 假函数** —— 这里的工具只当输出格式用，没有执行语义。

    Returns:
        OpenAI 风格的工具定义
    """
    schema = convert_to_openai_tool(Verdict)
    schema["function"]["name"] = TOOL_NAME
    return schema


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


def _format_calls(message: AIMessage) -> str:
    """把响应里的工具调用渲染成 ``名字(参数)`` 逐行。

    与 `llm_trace` / `compression` 的写法一致：Python 关键字风格比原始 JSON
    好读，`!r` 保证值仍可原样还原。

    Args:
        message: AI 响应

    Returns:
        每个调用一行；没有调用时返回空串
    """
    lines = []
    for call in message.tool_calls:
        args = ", ".join(f"{key}={value!r}" for key, value in (call.get("args") or {}).items())
        lines.append(f"{call.get('name', '?')}({args})")
    return "\n".join(lines)


def _ask(model: BaseChatModel, request: list[BaseMessage]) -> AIMessage:
    """调模型，返回**原始响应**。

    解析交给 `_parse`，中间隔着一层日志 —— 响应的样子必须打出来再解析：
    「没调工具」这个失败模式只有在看得到正文时才知道是「写了段散文」还是
    「什么都没输出」，而这两种的修法完全不同。

    Args:
        model: 主模型
        request: 消息列表

    Returns:
        AI 响应
    """
    return model.bind_tools([_verdict_tool()]).invoke(request)


def _parse(message: AIMessage) -> Verdict:
    """从响应里取出结论并校验。

    Args:
        message: AI 响应

    Returns:
        校验过的结论

    Raises:
        ValueError: 没调用工具、调用次数不对，或回参不符合 schema
    """
    calls = [call for call in message.tool_calls if call.get("name") == TOOL_NAME]
    if len(calls) != 1:
        msg = (
            f"期望恰好一次 {TOOL_NAME} 调用，实际 {len(calls)} 次"
            f"（总共 {len(message.tool_calls)} 次调用）"
        )
        raise ValueError(msg)
    return Verdict.model_validate(calls[0].get("args") or {})


def _resolve_misses(group: Group, misses: list[MissArg]) -> list[PotentialMiss]:
    """给漏报信号补上文件路径。

    路径由 `hunk_id` 查出，**不让模型自己写** —— 同一个坐标只留一个来源，
    与评论的 `path` 同源规则一致。编号对不上的直接丢（记一条 WARNING）：
    锚不到 diff 上的漏报提示没有可用之处，而它出现本身就说明这次输出不可靠。

    描述为空的也丢，不记日志 —— 那是噪声不是信号。

    Args:
        group: 审查单元
        misses: 模型报的漏报信号

    Returns:
        PotentialMiss 列表
    """
    index = {item.hunk_id: item.file_path for item in group.hunks}
    resolved: list[PotentialMiss] = []
    for item in misses:
        hunk_id = item.hunk_id.strip()
        path = index.get(hunk_id)
        if path is None:
            logger.warning("反思 · 忽略未知编号的漏报 %r", item.hunk_id)
            continue
        if not item.description.strip():
            continue
        resolved.append(
            PotentialMiss(hunk_id=hunk_id, path=path, description=item.description.strip())
        )
    return resolved


def reflect(model: BaseChatModel, group: Group, comments: list[Comment]) -> ReflectionOutcome:
    """复核一组评论，去掉被 diff 推翻的那些。

    Args:
        model: 主模型。复核是**换一个视角读同一批材料**，不是换一个模型
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

    prompt = tpl.REFLECTION_USER.format(
        diff=build_diff_block(group),
        comments="\n".join(_render_comment(item_id, item) for item_id, item in numbered),
    )
    request = [SystemMessage(content=tpl.REFLECTION_SYSTEM), HumanMessage(content=prompt)]
    # 复核走的是**裸模型调用**，不经过 `ReviewTrace`（它包的是图里的模型节点），
    # 所以这里不自己打就完全看不见 —— 与压缩请求同样的理由，级别也取 INFO
    logger.info("反思 · 请求 · system\n%s", fields(("message", tpl.REFLECTION_SYSTEM)))
    logger.info("反思 · 请求 · user\n%s", fields(("message", prompt)))

    started = time.monotonic()
    try:
        message = _ask(model, request)
        # 响应先打出来**再解析**：`反思 · 失败` 那条只说「没调工具」，
        # 看不出模型到底回了什么 —— 而「写了段散文」与「什么都没输出」
        # 的修法完全不同。与压缩的「请求 / 响应」两条腿对齐
        logger.info(
            "反思 · 响应 · 耗时 %dms\n%s",
            int((time.monotonic() - started) * 1000),
            fields(
                ("content", str(message.content)),
                ("tool_calls", _format_calls(message)),
            ),
        )
        verdict = _parse(message)
    # 宽捕是**故意的**：复核失败必须 fail-open，不能让整组审查崩掉
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("反思 · 失败 · %s，已保留全部 %d 条", reason, len(comments))
        return ReflectionOutcome(kept=list(comments), failure=reason)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    by_id = dict(numbered)
    removed: list[Removal] = []
    dropped: set[str] = set()
    for item in verdict.removals:
        comment_id = item.comment_id.strip()
        target = by_id.get(comment_id)
        if target is None:
            # 编号对不上就忽略，**绝不猜** —— 猜错一次就是删掉一条无关的评论
            logger.warning("反思 · 忽略未知编号 %r（那条评论已保留）", item.comment_id)
            continue
        if not item.reasoning.strip():
            # 指不出具体哪一处 diff → 按规则应当保留。空理由是「没指认」在
            # schema 层面唯一可判定的形态，所以在这里兜一层
            logger.warning(
                "反思 · 拒绝无依据的删除 %s（ground=%s，理由为空）", comment_id, item.ground
            )
            continue
        if comment_id in dropped:
            logger.debug("反思 · 忽略重复编号 %s", comment_id)
            continue
        dropped.add(comment_id)
        removed.append(
            Removal(
                comment_id=comment_id,
                path=target.path,
                ground=item.ground,
                reasoning=item.reasoning.strip(),
            )
        )

    misses = _resolve_misses(group, verdict.misses)
    kept = [item for item_id, item in numbered if item_id not in dropped]
    logger.info(
        "反思 · 完成 · %s · 删除 %d / 保留 %d（删除率 %.0f%%）· 耗时 %dms\n%s",
        group.group_id,
        len(removed),
        len(kept),
        len(removed) / len(comments) * 100,
        elapsed_ms,
        fields(
            # 删评论是唯一有破坏性的动作，依据与指认全部留档
            (
                "removed",
                "\n".join(
                    f"{item.comment_id} {item.path} Ground {item.ground}：{item.reasoning}"
                    for item in removed
                ),
            ),
            (
                "misses",
                "\n".join(f"{item.hunk_id} {item.path}：{item.description}" for item in misses),
            ),
        ),
    )
    return ReflectionOutcome(kept=kept, removed=removed, misses=misses)
