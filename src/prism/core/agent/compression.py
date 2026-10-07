"""审查历史的上下文压缩。

一个审查组跑下来，上下文会随轮次单向增长（`<review_status>` 每轮全量重发、
工具结果不断累积）。1M 窗口下大多数 PR 撞不到上限，但超大组会 —— 本模块是那道
保险：剩余空间不够时，把历史压成一份结构化摘要，让审查能继续跑到 end_done。

## 四步

1. **清理** —— 只清「可以重新获取」的工具结果（`file_read` / `code_search`），
   顺带删掉历史里的 `<review_status>`。判据只有「能不能重新获取」这一条，
   不是「有没有用」：`end_done` 被拒绝的那条返回看着没用，却是摘要里
   `Errors and Fixes` 段唯一的素材来源
2. **压缩** —— 历史拍平成一段带角色标注的 XML 文本，连同背景（审查 system +
   要求 + diff）与权威进度，交给主模型生成 7 段摘要
3. **重组** —— `RemoveMessage(REMOVE_ALL_MESSAGES)` 清空，再放回
   「A（指导 + diff）+ 摘要 + 新的 `<review_status>`」
4. **降级** —— 失败不静默：重试 → 纯截断 → 熔断，逐级记结构化告警

## 为什么不用框架的 `SummarizationMiddleware`

v1 自带那个（`langchain/agents/middleware/summarization.py`）覆盖了触发与摘要
调用，但有三处硬冲突：`keep` 默认保留最近 20 条消息（我们要全压），摘要请求只有
`{messages}` 一个变量（塞不进背景），失败直接抛（没有降级与熔断）。
三处做法倒是借了过来：拍平成 XML 文本、`REMOVE_ALL_MESSAGES` 哨兵、
拿 `usage_metadata.total_tokens` 比阈值。

## 状态是按实例的

压缩次数、连续失败数、熔断标记都是**每组审查独立**的，所以 `Compressor` 由
`build_review_agent` 每组造一份 —— 共用会把上一组的熔断带到下一组。
"""

import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from prism.core.agent.state import HunkStatus
from prism.core.agent.status import REVIEW_STATUS_TAG_OPEN, build_review_status
from prism.core.guidance import templates as tpl
from prism.models.compression import CompressionConfig
from prism.observability.llm_trace import fields

logger = logging.getLogger(__name__)

# 可以重新获取的工具：结果清空后模型能再调一次拿回来。
#
# **判据只有这一条**，不是「有没有信息价值」。按「有没有用」判会把
# `report_hunks`（回一句 ack + 进度计数）和 `end_done`（回「审查完成」或
# 「⚠️ 剩余 X 未审」）也清掉 —— 后者的拒绝消息是摘要里 `Errors and Fixes`
# 段唯一的素材来源，删了那一段永远是「（无）」
REFETCHABLE_TOOLS = frozenset({"file_read", "file_read_diff", "code_search"})

# 占位符要同时说清三件事：发生了什么、原内容是什么、怎么拿回来。
# 只说「(已清理)」模型不知道这儿原本有什么，也看不出能重新取；
# 说成「(内容已抹除)」则会被读成「这段不重要」。
PLACEHOLDER = "[内容已清理：原为 {call}，可重新获取]"

# 占位符的前缀。**判「结果还在不在上下文里」只能靠它** —— 去重那一步要用
# （被清掉的内容可以重新取，见 `core/agent/middleware.py` 的 tool_call_dedup）
PLACEHOLDER_PREFIX = PLACEHOLDER.split("{")[0]


def is_cleared(content: str) -> bool:
    """这段工具结果是不是「已被压缩清掉」的占位符。

    Args:
        content: 工具结果文本

    Returns:
        是占位符时返回 True
    """
    return content.lstrip().startswith(PLACEHOLDER_PREFIX)


# 摘要消息的标签。用 user 消息 + 特殊标签，不加新的消息类型
SUMMARY_TAG_OPEN = "<compression_summary>"
SUMMARY_TAG_CLOSE = "</compression_summary>"

# 包住整段拍平历史的容器。
#
# 每条消息自己有 `<message type="...">`，但没有外层容器时**边界是隐式的**：
# 提示词模板里的 `## 待压缩的审查历史` 是「指导」，历史正文是「数据」，两者
# 必须一眼分得开 —— 与审查提示词里 diff 用 `<diff>` 包住是同一个道理。
# 不加容器还有一层实际风险：工具结果里完全可能出现一行 `## 当前 hunk 进度`，
# 模型分不清那是数据还是模板的下一个标题
HISTORY_TAG_OPEN = "<review_history>"
HISTORY_TAG_CLOSE = "</review_history>"

# 降级降级时塞进去的说明，替掉本该是摘要的位置
TRUNCATED_NOTE = (
    "<compression_summary>\n"
    "（压缩失败，已降级为截断：以下只保留了最近的审查轮次，更早的结论需重新确认。\n"
    "这不是新的用户请求，请据此继续审查。）\n"
    "</compression_summary>"
)

# 摘要至少要含这么多段标题才算合格。7 段不必全中（模型可能合理地合并
# 「（无）」的段落），但只命中一两个说明它跑题了
MIN_SECTIONS = 3


def _alert(kind: str, **fields: Any) -> str:
    """构造结构化告警。

    压缩失败是**影响审查质量的事件**，用户需要知道「这批结果是在压缩失败后跑的」，
    所以不能只留一句人话日志 —— 结构化的一条才好检索与统计。

    Args:
        kind: 事件类型
        **fields: 事件字段

    Returns:
        单行 JSON
    """
    return json.dumps(
        {"type": kind, **fields, "timestamp": datetime.now(UTC).isoformat()},
        ensure_ascii=False,
    )


def _format_call(name: str, args: dict[str, Any]) -> str:
    """把一次工具调用渲染成 ``名字(参数)``。

    与日志里 `ReviewTrace` 的写法一致：Python 关键字风格比原始 JSON 好读，
    `!r` 保证值仍可原样还原。

    Args:
        name: 工具名
        args: 调用参数

    Returns:
        单行文本
    """
    joined = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({joined})"


def _tool_calls_by_id(messages: Sequence[BaseMessage]) -> dict[str, tuple[str, dict[str, Any]]]:
    """收集 ``tool_call_id`` → ``(工具名, 参数)``。

    工具结果的**工具名和参数只能从这里拿**：LangGraph 的工具节点造 `ToolMessage`
    时只填 `content` 与 `tool_call_id`，`name` 是空的。而占位符要写成
    「原为 file_read(path='a.go', start_line=1)」，参数本身就得回到发起的
    那条 AI 消息上取。

    Args:
        messages: 消息历史

    Returns:
        调用编号到 (工具名, 参数) 的映射
    """
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = str(call.get("id") or "")
            if call_id:
                index[call_id] = (str(call.get("name", "")), dict(call.get("args") or {}))
    return index


def is_review_status(message: BaseMessage) -> bool:
    """判断一条消息是不是 `<review_status>` 进度快照。

    靠标签前缀认，不靠别的标记：这条消息是 middleware 用
    `HumanMessage(content=build_review_status(...))` 造的，除了正文里的标签
    没有别的身份信息。

    Args:
        message: 待判断的消息

    Returns:
        是进度快照返回 True
    """
    return isinstance(message, HumanMessage) and str(message.content).lstrip().startswith(
        REVIEW_STATUS_TAG_OPEN
    )


def clean_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """压缩前清理。

    两件事：

    - 可重新获取的工具结果 → 换占位符（**保留消息本身**，API 要求
      `tool_use` / `tool_result` 成对出现，删消息会直接报错）
    - 历史里的 `<review_status>` → **整条删掉**。它是全量快照，每一版都含当时
      已完成的全部 hunk，N 轮就是 N 份高度重叠的文本。留着会让「权威进度」不唯一，
      模型可能把中间态当成确实发生过的事写进 `Completed Hunks`

    首条消息（指导 + diff）一个字不动 —— 它是审查的立身之本，也是压缩后要原样
    放回去的 A 段。

    Args:
        messages: 消息历史

    Returns:
        清理后的消息列表
    """
    calls = _tool_calls_by_id(messages)
    cleaned: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if index == 0:
            cleaned.append(message)
            continue
        if is_review_status(message):
            continue
        if isinstance(message, ToolMessage):
            name, args = calls.get(str(message.tool_call_id), ("", {}))
            if name in REFETCHABLE_TOOLS:
                cleaned.append(
                    message.model_copy(
                        update={"content": PLACEHOLDER.format(call=_format_call(name, args))}
                    )
                )
                continue
        cleaned.append(message)
    return cleaned


def render_history(messages: Sequence[BaseMessage]) -> str:
    """把消息历史拍平成一段带角色标注的文本。

    用框架的 `get_buffer_string(format="xml")`，产物形如::

        <message type="ai">
          <tool_call id="c1" name="file_read">{"path": "a.go"}</tool_call>
        </message>
        <message type="tool">package a ...</message>

    拍平是**故意的**：压缩请求只有一条 user 消息，历史在协议层面不是「消息」
    而是文本，模型不可能把它们误当成正在进行的轮次。

    外面再套一层 `<review_history>`：逐条的 `<message>` 标的是「谁说的」，
    容器标的才是「这一整段是数据」。没有容器时，模板里的 `## 待压缩的审查历史`
    与历史正文之间没有可见的边界。

    Args:
        messages: 消息历史

    Returns:
        以 `<review_history>` 包住的 XML 标注文本
    """
    body = get_buffer_string(list(messages), format="xml")
    return f"{HISTORY_TAG_OPEN}\n{body}\n{HISTORY_TAG_CLOSE}"


def last_reported_tokens(messages: Sequence[BaseMessage]) -> int | None:
    """最后一条 AI 响应报告的上下文大小。

    用 API 返回的 `usage_metadata` 而不是自己估 —— 它是权威数据，零实现成本，
    也不用去处理分词差异与消息格式开销。

    代价是**滞后一轮**：拿到的是「上一轮发出去时有多大」，本轮新加的工具结果要
    到下一轮才反映出来。这层误差靠 `reserve_tokens` 兜住，不另做增量估算。

    Args:
        messages: 消息历史

    Returns:
        总 token 数；历史里没有带 usage 的 AI 消息时返回 None
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata
        if not usage:
            return None
        return int(usage.get("total_tokens", 0))
    return None


def last_ai_id(messages: Sequence[BaseMessage]) -> str | None:
    """最近一条 AI 消息的编号。

    用来判断「自上次压缩以来有没有新信息」。**不能拿 AI 消息条数当依据**：
    降级截断会把老消息丢掉，条数变小反而更像「有新信息」，于是原地反复触发。

    Args:
        messages: 消息历史

    Returns:
        消息编号；历史里没有 AI 消息时返回 None
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.id
    return None


def round_length(messages: Sequence[BaseMessage]) -> int:
    """从开头算起，第一条消息所属的「轮」占几条。

    截断必须**按轮丢**：AI 消息和它发起的工具结果是一对，拆开会让
    `tool_call_id` 找不到对应的调用方，请求直接被 API 拒掉。

    Args:
        messages: 消息列表（首条应是 AIMessage）

    Returns:
        该轮的消息条数；列表里没有下一条 AI 消息时返回整个长度
    """
    for index, message in enumerate(messages[1:], start=1):
        if isinstance(message, AIMessage):
            return index
    return len(messages)


class Compressor:
    """一次审查的压缩器。

    状态按实例：压缩次数、连续失败数、熔断标记都是每组的，
    `build_review_agent` 每组造一份。
    """

    def __init__(self, model: BaseChatModel, config: CompressionConfig) -> None:
        """初始化。

        Args:
            model: 主模型。摘要用它生成 —— 压缩是信息搬运不是判断，不需要第二个视角
            config: 阈值与容错参数
        """
        self._config = config
        # 重试交给框架：`with_retry` 默认 3 次、指数退避。
        # 它管的是**瞬态错误**（超时、限流）；格式错误另有一层，见 `_summarize`
        self._model = model.with_retry()
        self._compressions = 0
        self._failures = 0
        self._breaker_open = False
        # 上次压缩（或降级）时最近那条 AI 消息的编号，用来判断「有没有新信息」
        self._handled_ai_id: str | None = None

    def should_compress(self, messages: Sequence[BaseMessage]) -> bool:
        """判断本轮模型调用前是否该压缩。

        Args:
            messages: 当前消息历史

        Returns:
            该压缩返回 True
        """
        if self._breaker_open:
            return False
        if self._compressions >= self._config.max_compressions:
            logger.debug("已达压缩次数上限 %d，不再压缩", self._config.max_compressions)
            return False
        if last_ai_id(messages) is not None and last_ai_id(messages) == self._handled_ai_id:
            # 压缩本身**不产生新的 AI 消息**。不挡住这一条的话，下一次
            # before_model 读到的还是同一条 AI 响应的 usage，判定仍然是「超限」，
            # 于是原地反复压缩 —— 而且每次都在烧一次摘要请求
            return False
        used = last_reported_tokens(messages)
        if used is None:
            logger.debug("拿不到 usage，跳过压缩判定")
            return False
        if used < self._config.usable_tokens:
            return False

        logger.info(
            "压缩 · 触发 · 第 %d/%d 次\n%s",
            self._compressions + 1,
            self._config.max_compressions,
            fields(
                (
                    "token",
                    f"上一轮已用 {used} · 可用上限 {self._config.usable_tokens}"
                    f" · 窗口 {self._config.context_window}"
                    f" · 固定预留 {self._config.reserve_tokens}",
                ),
            ),
        )
        return True

    def compress(
        self, messages: Sequence[BaseMessage], hunk_status: dict[str, HunkStatus]
    ) -> list[BaseMessage]:
        """压缩历史，返回替换整个消息列表的更新。

        永不返回失败 —— 摘不出来就降级为截断。调用方只管把结果塞进
        `messages` 更新里。

        Args:
            messages: 当前消息历史
            hunk_status: hunk 状态表，用来生成新的权威进度

        Returns:
            以 `RemoveMessage(REMOVE_ALL_MESSAGES)` 开头的完整消息列表
        """
        head = messages[0]
        status = build_review_status(hunk_status)
        history = clean_history(messages)[1:]
        # 不可压缩的部分：背景（A 段）+ 进度 + 压缩提示词骨架
        fixed = self._tokens([head, HumanMessage(content=status)]) + self._prompt_tokens()
        history = self._fit(history, fixed=fixed)
        # 成败都要记：降级截断同样改了上下文，不记的话下一轮会拿同一条 usage
        # 再触发一次
        self._handled_ai_id = last_ai_id(messages)

        summary = self._summarize(str(head.content), history, status)
        if summary is None:
            self._record_failure()
            return self._degrade(head, messages, hunk_status, status)

        assembled = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            head,
            HumanMessage(
                content=(
                    f"{tpl.COMPRESSION_NOTICE}\n{SUMMARY_TAG_OPEN}\n{summary}\n{SUMMARY_TAG_CLOSE}"
                )
            ),
            HumanMessage(content=status),
        ]
        self._record_success(history, summary, assembled[1:])
        return assembled

    def _prompt_tokens(self) -> int:
        """压缩提示词（system + user 骨架）自身的 token 开销。

        Returns:
            提示词文本的估算 token 数
        """
        return self._tokens(
            [
                SystemMessage(content=tpl.COMPRESSION_SYSTEM),
                HumanMessage(content=tpl.COMPRESSION_USER),
            ]
        )

    def _tokens(self, messages: Sequence[BaseMessage]) -> int:
        """估算一组消息的 token 数。

        用框架的近似计数器（`count_tokens_approximately`）。**只在装不装得下这种
        「差一点没关系」的判断上用**，触发判定仍然只看 API 给的 usage。

        Args:
            messages: 消息列表

        Returns:
            估算的 token 数
        """
        return count_tokens_approximately(list(messages))

    def _fit(self, history: list[BaseMessage], *, fixed: int) -> list[BaseMessage]:
        """丢最旧轮次，直到压缩请求装得进窗口。

        §2.9 的硬约束：压缩请求自己也得满足
        ``压缩 sys + 压缩 user + 历史 + 预留 ≤ 窗口``。极端情况下历史本身就接近
        窗口，加上提示词就发出去了 —— 那这一轮压缩必然失败。先截断是最后的保底。

        Args:
            history: 清理后的历史
            fixed: 不可压缩部分（背景 + 进度 + 提示词）的 token 数

        Returns:
            截断后的历史
        """
        budget = self._config.usable_tokens - fixed
        dropped = 0
        while history and self._tokens(history) > budget:
            history = history[round_length(history) :]
            dropped += 1
        if dropped:
            logger.warning(
                "压缩 · 截断 · 丢弃最旧 %d 轮\n%s",
                dropped,
                fields(
                    ("budget", f"历史预算 {budget} token"),
                    (
                        "alert",
                        _alert(
                            "compression_truncated",
                            reason="request_too_large",
                            dropped_rounds=dropped,
                        ),
                    ),
                ),
            )
        return history

    def _summarize(self, background: str, history: list[BaseMessage], status: str) -> str | None:
        """调模型生成摘要。

        两层重试各管各的：`with_retry` 管瞬态错误（超时、限流），本方法的循环管
        **格式错误** —— 模型回来了但没按 7 段写，值得原样再要一次，不值得重试三次。

        Args:
            background: A 段的原文（指导 + diff）
            history: 清理并截断后的历史
            status: 当前的 `<review_status>` 文本

        Returns:
            摘要正文；两次都不合格时返回 None
        """
        prompt = tpl.COMPRESSION_USER.format(
            review_system=tpl.SYSTEM,
            background=background,
            history=render_history(history),
            status=status,
        )
        request = [SystemMessage(content=tpl.COMPRESSION_SYSTEM), HumanMessage(content=prompt)]
        # 压缩走的是**裸模型调用**，不经过 `ReviewTrace`（它包的是图里的模型节点），
        # 所以这里不自己打就完全看不见 —— 调压缩提示词时只能靠猜。
        # 级别与审查请求一致取 INFO：`ReviewTrace` 在 INFO 下本来就把 system 与
        # 每条消息的**全文**打出来了，压缩请求没有理由比它更藏。
        #
        # **两条消息分两条日志打**，别塞进同一个 fields 块 —— 那样看起来像是拼成
        # 了一条消息，与实际的请求形态（system + user 两条）对不上
        logger.info("压缩 · 请求 · system\n%s", fields(("message", tpl.COMPRESSION_SYSTEM)))
        logger.info(
            "压缩 · 请求 · user · 历史 %d 条\n%s",
            len(history),
            fields(("message", prompt)),
        )

        for attempt in (1, 2):
            started = time.monotonic()
            try:
                text = str(self._model.invoke(request).text).strip()
            # 宽捕是**故意的**：压缩失败必须降级，不能让整组审查崩掉
            except Exception as exc:
                logger.warning(
                    "压缩 · 请求失败 · 第 %d 次\n%s",
                    attempt,
                    fields(("error", f"{type(exc).__name__}: {exc}")),
                )
                continue
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "压缩 · 响应 · 第 %d 次 · 耗时 %dms\n%s",
                attempt,
                elapsed_ms,
                fields(("content", text)),
            )
            if _is_summary_shaped(text):
                return text
            logger.warning(
                "压缩 · 响应不合格 · 第 %d 次\n%s",
                attempt,
                fields(
                    ("content", text),
                    ("reason", f"7 段标题只命中 {_section_hits(text)} 个，少于 {MIN_SECTIONS}"),
                ),
            )

        logger.error(
            "压缩 · 放弃 · 两次都未得到合格摘要\n%s",
            fields(("alert", _alert("compression_failed", reason="bad_response"))),
        )
        return None

    def _degrade(
        self,
        head: BaseMessage,
        messages: Sequence[BaseMessage],
        hunk_status: dict[str, HunkStatus],
        status: str,
    ) -> list[BaseMessage]:
        """降级：纯截断。

        摘不出来时不能让审查卡死在超限上 —— 丢掉最旧的轮次，把最近的工作留下，
        并让模型知道「更早的结论不可信了」。

        Args:
            head: A 段消息（指导 + diff）
            messages: 原始消息历史
            hunk_status: hunk 状态表
            status: `<review_status>` 文本

        Returns:
            以 `RemoveMessage(REMOVE_ALL_MESSAGES)` 开头的完整消息列表
        """
        note = HumanMessage(content=TRUNCATED_NOTE)
        history = clean_history(messages)[1:]
        fixed = self._tokens([head, note, HumanMessage(content=status)])
        history = self._fit(history, fixed=fixed)
        logger.warning(
            "压缩 · 降级为截断 · 保留最近 %d 条\n%s",
            len(history),
            fields(
                (
                    "alert",
                    _alert("compression_failed", reason="degraded", kept_messages=len(history)),
                ),
            ),
        )
        return [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            head,
            note,
            *history,
            HumanMessage(content=status),
        ]

    def _record_failure(self) -> None:
        """记一次失败，到阈值就熔断。"""
        self._failures += 1
        if self._failures >= self._config.max_consecutive_failures:
            self._breaker_open = True
            logger.error(
                "压缩 · 熔断 · 连续失败 %d 次，本组不再尝试\n%s",
                self._failures,
                fields(
                    ("alert", _alert("compression_circuit_open", failures=self._failures)),
                ),
            )
        else:
            logger.warning(
                "压缩 · 失败 · 连续第 %d 次\n%s",
                self._failures,
                fields(
                    (
                        "alert",
                        _alert("compression_failed", reason="error", failures=self._failures),
                    ),
                ),
            )

    def _record_success(
        self, history: list[BaseMessage], summary: str, assembled: Sequence[BaseMessage]
    ) -> None:
        """记一次成功。

        记下此刻最近的 AI 消息编号给 `should_compress` 用，并把压缩比打出来 ——
        §4.2 的校准要靠它：攒够 20 次后取「实际 E / B」的 P50。

        Args:
            history: 压缩前的历史（清理后）
            summary: 生成的摘要
            assembled: 重组之后的上下文（A 段 + 摘要 + 进度）
        """
        self._compressions += 1
        self._failures = 0
        before = self._tokens(history)
        after = self._tokens([HumanMessage(content=summary)])
        ratio = after / before if before else 0.0
        logger.info(
            "压缩 · 完成 · 第 %d/%d 次\n%s",
            self._compressions,
            self._config.max_compressions,
            fields(
                ("size", f"历史 {before} token → 摘要 {after} token"),
                ("ratio", f"{ratio:.2f}"),
                # 「压缩后还能撑几轮」是校准问题，得先有「压缩后多大」这个数
                ("context", f"压缩后上下文 {self._tokens(assembled)} token"),
            ),
        )


def _section_hits(text: str) -> int:
    """数摘要命中了几个约定的段标题。

    Args:
        text: 模型返回的正文

    Returns:
        命中的段标题数
    """
    return sum(1 for section in tpl.SUMMARY_SECTIONS if section in text)


def _is_summary_shaped(text: str) -> bool:
    """摘要的段标题命中数够不够。

    Args:
        text: 模型返回的正文

    Returns:
        命中数达到 `MIN_SECTIONS` 返回 True
    """
    return _section_hits(text) >= MIN_SECTIONS
