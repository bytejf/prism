"""hunk 报告工具。

LLM 只报告原始数据（结论 + 评论），进度由代码统计。
返回值只给「剩余未审」而非「已完成」，让返回体随进度递减。

状态回写走框架的 `Command`（工具更新 state 的正路），
不再是「改一个塞在 state 里的共享可变对象」。
"""

import logging
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import (
    STATUS_DONE,
    HunkStatus,
    ReviewState,
    done_count,
)
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)


class CommentArg(BaseModel):
    """LLM 报告的单条评论。"""

    # 类 docstring 会成为 schema 里的 description，**是模型看得到的**，所以只写
    # 一句给模型看的话；下面这条设计理由用注释写，别写进 docstring。
    #
    # 为什么把字段做成显式模型而不是 list[dict]：前者把字段说明摆到模型面前，
    # 实测 severity 漏填率 55% → 0（渲染成 `[类别 · ]` 的那批评论没有了）。
    severity: Literal["high", "medium", "low"] = Field(
        description=(
            "严重程度，必填。high=会导致错误/崩溃/数据损坏；"
            "medium=有风险或明显的维护性问题；low=风格与可读性"
        )
    )
    # 取值 = 主任务里那 7 条**审查维度的原名** + 「其他」。
    #
    # **为什么用维度的原名而不是另起一套短标签**：原样复用同一套词，
    # 漂移就没法发生（`test_comment_categories_match_review_dimensions` 钉住），
    # 而且模型不用学第二套概念。代价是徽章略长（`[意图与实现的一致性 · medium]`）。
    #
    # **「其他」不是退路，是指标**：它用得越多，说明这 7 条维度越不够用 ——
    # 比如性能问题在这套维度里没有归宿，只能落「其他」。所以它满了就该加维度，
    # 而不是继续让它兜底。（OCR 的做法相反：另有一套 8 类通用标签
    # bug/security/performance/maintainability/…，和它的检查清单不是一套词。）
    category: Literal[
        "业务逻辑正确性",
        "跨文件一致性",
        "意图与实现的一致性",
        "副作用与隐式契约",
        "边界条件",
        "可维护性",
        "安全",
        "其他",
    ] = Field(description="问题类别：选最贴近的那一条审查维度；**都不贴近才用「其他」**")
    content: str = Field(description="评论正文：说清问题、影响与依据。**一条评论只讲一件事**")
    existing_code: str = Field(
        description=(
            "用于定位的代码片段：本 hunk diff 里**改动行**（`+` / `-`）的原文，"
            "逐行照抄。**抄了 diff 里没有的行，这条评论会挂不上、变成未定位**"
        )
    )
    suggestion_code: str | None = Field(
        default=None, description="建议的修改代码（可选；能给出就直接可替换的那种）"
    )


# 类 docstring 会成为 schema 里的 `items.description`，**是模型看得到的**，
# 所以只写一句给模型看的话；下面这条设计理由用注释写。
#
# **为什么没有 `complete` 字段**（反馈文档 23 号 §十一）：那个布尔表达的是
# 「这是终局结论、以后不许再报」，模型读到之后理性地攒到最后一次性报，而收尾闸
# 又按预算把它按停 —— 两边打架，产出的是假结论。去掉之后「报过 = 关注过」，
# 同一个 hunk 随时可以再报、评论累加。
class HunkReportArg(BaseModel):
    """单个 hunk 的审查结论。"""

    hunk_id: str = Field(description="hunk 编号，必须来自审查范围，如 H-01")
    comments: list[CommentArg] = Field(
        default_factory=list, description="在这个 hunk 上发现的问题；没有问题就传空列表"
    )


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py。
@tool(
    parse_docstring=True,
    description=(
        "报告 hunk 的审查结论（有问题报问题，没有问题也报一次空列表）。\n"
        "报过的 hunk 之后仍可继续补充**新**评论（与已报过的内容重复的会被忽略）。"
        "所有 hunk 都报过一次之后，才能用 end_done 结束。"
    ),
)
def report_hunks(
    reports: list[HunkReportArg],
    runtime: ToolRuntime[ReviewContext, ReviewState],
) -> Command[Any]:
    """报告 hunk 的审查结论。

    `reports` 的说明走下面的 `Args:`，**嵌套字段的说明走 `HunkReportArg` /
    `CommentArg` 的 `Field`** —— 嵌套结构只能由 Pydantic 模型表达。
    `runtime` 由框架注入、模型看不到，会被自动排除。

    Args:
        reports: hunk 报告列表
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        状态更新（新 hunk 状态表 + 新评论 + 进度文本的 ToolMessage）
    """
    state = runtime.state
    hunk_status = dict(state["hunk_status"])
    # 只把**本次改动**的 hunk 写回去（增量）。并行调用时两份增量按 key 合并，
    # 各改各的互不覆盖；写全量表会被后一份快照整个盖掉
    changed: dict[str, HunkStatus] = {}

    logger.debug("report_hunks 调用：%d 个 hunk 报告", len(reports))

    recorded: list[str] = []
    unknown: list[str] = []
    already_done: list[str] = []
    supplemented: list[str] = []
    new_comment_count = 0

    for raw in reports:
        hunk_id = raw.hunk_id.strip()
        current = hunk_status.get(hunk_id)
        if current is None:
            unknown.append(hunk_id or "<空编号>")
            continue

        comments = [_to_comment(item) for item in raw.comments]

        # 已报过的 hunk **允许继续补充**（去掉重复的之后追加），从来不是「整份忽略」。
        # 去重挡的是「反复确认同一个 hunk」那类重复评论（模型看到进度提示后想
        # 「再确认一下 H-01」是自然行为），不该挡新结论
        if current.status == STATUS_DONE:
            said = {_normalize(item.content) for item in current.comments}
            fresh = [item for item in comments if _normalize(item.content) not in said]
            if not fresh:
                already_done.append(hunk_id)
                continue
            comments = fresh
            supplemented.append(hunk_id)
        else:
            recorded.append(hunk_id)

        new_comment_count += len(comments)

        # 整体替换成新对象，不做原地修改
        updated = HunkStatus(
            hunk_id=hunk_id,
            status=STATUS_DONE,
            comments=[*current.comments, *comments],
        )
        hunk_status[hunk_id] = updated
        changed[hunk_id] = updated

    done = done_count(hunk_status)
    total = len(hunk_status)
    logger.info(
        "report_hunks：记录 %s，新增 %d 条评论，进度 %d/%d（补充已报过的 %d 个，"
        "重复报告被忽略 %d 个）",
        ", ".join(recorded) or "无",
        new_comment_count,
        done,
        total,
        len(supplemented),
        len(already_done),
    )
    if unknown:
        logger.warning("report_hunks 收到未知 hunk 编号：%s", ", ".join(unknown))
    if already_done:
        logger.warning("report_hunks 重复报告已完成的 hunk，已忽略：%s", ", ".join(already_done))

    parts: list[str] = []
    if recorded:
        parts.append(f"已记录 {len(recorded)} 个 hunk：{', '.join(recorded)}")
    if supplemented:
        parts.append(
            f"已为 {len(supplemented)} 个报过的 hunk 补充了新评论：{', '.join(supplemented)}"
        )
    if recorded or supplemented:
        parts.append(f"新增评论 {new_comment_count} 条")
    else:
        parts.append("未记录任何 hunk")
    if already_done:
        parts.append(
            f"⚠️ 以下 hunk 本轮的评论与已报过的完全相同，已忽略：{', '.join(already_done)}。"
            "（同一个 hunk 可以继续补充**新**的结论，重复内容不必再报）"
        )
    if unknown:
        parts.append(f"⚠️ 未知 hunk 编号（已忽略）：{', '.join(unknown)}")
    # 进度的**权威表述**不在这里，在那条 `<review_status>` 快照（它带着覆盖缺口与
    # 预算进度）。这里再说一遍只会让同一个数字在一轮里出现两次、还少一半信息

    # 写回去的是**增量** `changed`，不是全量 `hunk_status` —— 并行调用要靠
    # reducer 按 key 合并增量。另外 `end_done_count` 的重置挪到了
    # `review_progress`（middleware 是单写者，工具节点可以并行）：两个工具
    # 同时写同一个键会触发 LangGraph 的 InvalidUpdateError
    return Command(
        update={
            "hunk_status": changed,
            "messages": [ToolMessage(content="\n".join(parts), tool_call_id=runtime.tool_call_id)],
        }
    )


def _normalize(text: str) -> str:
    """把评论正文归一化，用来判「这条是不是已经报过了」。

    只折叠空白、去首尾 —— **不做语义相似**：那是另一个量级的东西，而且判错的
    代价（把一条真评论当重复丢掉）比留下一条重复评论大得多。

    Args:
        text: 评论正文

    Returns:
        归一化后的文本
    """
    return " ".join(text.split())


def _to_comment(arg: CommentArg) -> CommentInput:
    """把模型报的评论转成 CommentInput。

    severity 由 `Literal` 保证合法 —— 非法值在进到这里之前就被 Pydantic
    拦下、作为错误回灌给模型重试，不会静默变成空字符串。

    Args:
        arg: 模型报的评论（参数名不叫 `raw`，避免与调用点的循环变量同名，
            PyCharm 会把两者混起来推断）

    Returns:
        CommentInput 对象
    """
    return CommentInput(
        severity=arg.severity,
        category=arg.category.strip(),
        content=arg.content.strip(),
        existing_code=arg.existing_code.strip(),
        suggestion_code=_optional_text(arg.suggestion_code),
    )


def _optional_text(value: object) -> str | None:
    """把可选字段转成非空字符串或 None。

    Args:
        value: 原始值

    Returns:
        去空白后的文本；为空或 None 时返回 None
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None
