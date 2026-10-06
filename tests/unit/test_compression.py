"""上下文压缩的单元测试。

这里只测**部件**：清理、拍平、截断、触发判定、降级、熔断。与审查循环的集成
（压缩真的在 `before_model` 里发生、压缩后模型看到什么）放在 `test_main_loop.py`，
那边有 `ScriptedChatModel` 可复用。
"""

from dataclasses import replace
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from prism.config.settings import settings
from prism.core.agent.compression import (
    PLACEHOLDER,
    SUMMARY_TAG_OPEN,
    TRUNCATED_NOTE,
    Compressor,
    clean_history,
    last_ai_id,
    last_reported_tokens,
    render_history,
    round_length,
)
from prism.core.agent.state import STATUS_DONE, HunkStatus
from prism.core.guidance import templates as tpl

# 一段「像样的」摘要：命中 7 段里的几段，过得了格式校验
GOOD_SUMMARY = "\n".join(
    (
        "## Primary Request",
        "审查 . 包的 2 个 hunk。",
        "",
        "## Completed Hunks",
        "- H-01：无问题",
        "",
        "## Current Work",
        "正在审查 H-02。",
    )
)


class FakeSummaryModel(BaseChatModel):
    """按脚本返回**文本**的假模型，供压缩器用。

    与 `test_main_loop.py` 的 `ScriptedChatModel` 分开：那个返回构造好的
    `AIMessage`（连工具调用一起），压缩只关心一段文本。
    """

    replies: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    @override
    def _llm_type(self) -> str:
        """模型类型标识。

        Returns:
            固定字符串
        """
        return "fake-summary"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回下一个脚本响应。

        Args:
            messages: 收到的消息
            stop: 停止词（忽略）
            run_manager: 回调管理器（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            ChatResult
        """
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else "（脚本用尽）"
        if text == "!raise":
            msg = "模拟压缩请求失败"
            raise RuntimeError(msg)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def usage(total: int) -> Any:
    """构造一份 `usage_metadata`。

    返回 `Any` 而不是 `UsageMetadata`：后者是 TypedDict，`dict[str, Any]` 传不进去，
    显式写全又要四五个键 —— 这里只要 `total_tokens` 是对的。

    Args:
        total: 总 token 数

    Returns:
        usage 字典
    """
    return {"input_tokens": total - 1, "output_tokens": 1, "total_tokens": total}


def call_message(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """构造一条发起工具调用的 AI 消息。

    Args:
        name: 工具名
        args: 参数
        call_id: 调用编号

    Returns:
        AIMessage
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        id=f"ai-{call_id}",
    )


def sample_history() -> list[BaseMessage]:
    """构造一段典型的审查历史。

    Returns:
        消息列表：指导 + diff / file_read 往返 / report_hunks 往返 / 进度快照
    """
    return [
        HumanMessage(content="指导 + diff", id="m0"),
        call_message("file_read", {"path": "a.go", "start_line": 1, "end_line": 9}, "c1"),
        ToolMessage(content="500 行代码", tool_call_id="c1", id="m1"),
        call_message("report_hunks", {"reports": [{"hunk_id": "H-01"}]}, "c2"),
        ToolMessage(content="已记录 1 个 hunk：H-01", tool_call_id="c2", id="m2"),
        HumanMessage(content="<review_status>\n进度 1/2\n</review_status>", id="m3"),
    ]


# ---------- 清理 ----------


def test_clean_replaces_refetchable_and_drops_old_status() -> None:
    """测试：可重新获取的工具结果换占位符、旧进度快照整条删。"""
    cleaned = clean_history(sample_history())

    assert cleaned[0].content == "指导 + diff", "首条（指导 + diff）一个字不动"
    assert len(cleaned) == 5, "旧 <review_status> 应被整条删掉（6 → 5）"
    placeholder = str(cleaned[2].content)
    assert placeholder == PLACEHOLDER.format(
        call="file_read(path='a.go', start_line=1, end_line=9)"
    )
    assert cleaned[4].content == "已记录 1 个 hunk：H-01", "report_hunks 的返回要留着"


def test_clean_replaces_file_read_diff() -> None:
    """测试：file_read_diff 的返回也换成占位符。

    判据只有「能不能重新拿到」这一条 —— 它查的是内存里的 diff 索引，
    再调一次就能拿回原文，和 file_read / code_search 同类。
    """
    history = [
        HumanMessage(content="指导 + diff", id="m0"),
        call_message("file_read_diff", {"path": "b.go"}, "c1"),
        ToolMessage(content="diff --git a/b.go b/b.go\n@@ -1 +1 @@\n-p\n+q", tool_call_id="c1"),
    ]

    cleaned = clean_history(history)

    assert str(cleaned[2].content) == PLACEHOLDER.format(call="file_read_diff(path='b.go')")


def test_clean_keeps_end_done_rejection() -> None:
    """测试：end_done 被拒绝的返回必须保留。

    它是摘要里 `Errors and Fixes` 段唯一的素材来源 —— 按「有没有信息价值」去清，
    这一段会永远是空的。
    """
    history = [
        HumanMessage(content="指导 + diff", id="m0"),
        call_message("end_done", {}, "c1"),
        ToolMessage(
            content="⚠️ 剩余 H-03, H-05 未审，请先 report_hunks", tool_call_id="c1", id="m1"
        ),
    ]

    cleaned = clean_history(history)

    assert len(cleaned) == 3
    assert "剩余 H-03, H-05 未审" in str(cleaned[2].content)


def test_clean_keeps_compression_summary() -> None:
    """测试：上一轮的压缩摘要不会被当成进度快照删掉。

    第二次压缩时，摘要也要参与压缩（递归压缩），所以它必须留在历史里。
    """
    history = [
        HumanMessage(content="指导 + diff", id="m0"),
        HumanMessage(content=f"{SUMMARY_TAG_OPEN}\n旧摘要\n</compression_summary>", id="m1"),
    ]

    cleaned = clean_history(history)

    assert len(cleaned) == 2
    assert "旧摘要" in str(cleaned[1].content)


def test_clean_leaves_message_count_and_ids_alone() -> None:
    """测试：换内容走 model_copy，原消息对象不被就地改写。"""
    history = sample_history()
    clean_history(history)

    assert history[2].content == "500 行代码", "清理不能就地改原始消息"


# ---------- 拍平与计量 ----------


def test_render_history_wraps_in_container() -> None:
    """测试：历史拍平成带角色标注的 XML，外面再套一层容器。

    逐条的 `<message>` 标的是「谁说的」，容器标的才是「这一整段是数据」。
    没有容器时，模板里的 `## 待压缩的审查历史` 与历史正文之间没有可见边界 ——
    工具结果里出现一行 `## 当前 hunk 进度` 就分不清是数据还是模板标题。
    """
    text = render_history(sample_history()[:3])

    assert text.startswith("<review_history>")
    assert text.endswith("</review_history>")
    assert '<message type="human">指导 + diff</message>' in text
    assert '<tool_call id="c1" name="file_read">' in text
    assert '<message type="tool">500 行代码</message>' in text


def test_last_reported_tokens_uses_usage() -> None:
    """测试：读最后一条 AI 响应的 usage。"""
    messages: list[BaseMessage] = [
        AIMessage(content="", id="a1", usage_metadata=usage(2)),
        HumanMessage(content="继续", id="m1"),
        AIMessage(content="", id="a2", usage_metadata=usage(1000)),
        ToolMessage(content="结果", tool_call_id="c1", id="m2"),
    ]

    assert last_reported_tokens(messages) == 1000


def test_last_reported_tokens_none_without_usage() -> None:
    """测试：没有 usage 时返回 None（调用方据此跳过判定）。"""
    assert last_reported_tokens([AIMessage(content="无 usage", id="a1")]) is None
    assert last_reported_tokens([HumanMessage(content="只有人话", id="m1")]) is None


def test_last_ai_id_ignores_trailing_tool_messages() -> None:
    """测试：取最近一条 AI 消息的编号，工具结果不算。"""
    messages = sample_history()

    assert last_ai_id(messages) == "ai-c2"


def test_round_length_stops_at_next_ai_message() -> None:
    """测试：一轮的长度算到下一个 AI 消息之前。"""
    history = sample_history()[1:]
    # [ai-c1, tool, ai-c2, tool, status] —— 第一轮到 ai-c2 之前，共 2 条
    assert round_length(history) == 2


# ---------- 触发判定 ----------


def make_compressor(replies: list[str], *, window: int = 1_000, reserve: int = 100) -> Compressor:
    """构造一个用假模型的压缩器。

    Args:
        replies: 假模型的响应脚本
        window: 窗口大小
        reserve: 固定预留

    Returns:
        Compressor
    """
    return Compressor(
        FakeSummaryModel(replies=replies),
        replace(settings.compression, context_window=window, reserve_tokens=reserve),
    )


def with_usage(tokens: int) -> list[BaseMessage]:
    """构造一段「上一轮用掉了 tokens」的历史。

    Args:
        tokens: 上一轮报告的 total_tokens

    Returns:
        消息列表
    """
    return [
        HumanMessage(content="指导 + diff", id="m0"),
        AIMessage(content="", id="ai-1", usage_metadata=usage(tokens)),
    ]


def test_should_compress_only_above_usable() -> None:
    """测试：已用 token 达到「窗口 - 预留」才触发。"""
    compressor = make_compressor([])

    assert not compressor.should_compress(with_usage(899)), "差一个 token 不该触发"
    assert compressor.should_compress(with_usage(900))


def test_should_compress_skips_when_no_new_ai_message() -> None:
    """测试：压缩本身不产生新的 AI 消息，同一条 usage 不能反复触发。

    没有这道闸，下一次 `before_model` 读到的还是同一条响应的 usage，判定仍然是
    「超限」，于是原地反复压缩 —— 每次还白烧一次摘要请求。
    """
    compressor = make_compressor([GOOD_SUMMARY])
    messages = with_usage(5_000)

    assert compressor.should_compress(messages)
    compressor.compress(messages, {})
    assert not compressor.should_compress(messages), "同一个 AI 消息不该二次触发"


def test_should_compress_skips_when_no_usage() -> None:
    """测试：拿不到 usage 就不判定（fake 模型、非 DeepSeek provider 都可能没有）。"""
    compressor = make_compressor([])

    assert not compressor.should_compress([HumanMessage(content="指导 + diff", id="m0")])


def test_should_compress_stops_after_max_compressions() -> None:
    """测试：压缩次数上限（§3.3 的 2 次）到了就不再压。"""
    compressor = make_compressor([GOOD_SUMMARY, GOOD_SUMMARY])

    for round_no in range(2):
        # 每轮换一个 AI 消息编号，模拟「又跑了一轮」
        messages = with_usage(5_000)
        messages[-1].id = f"ai-{round_no}"
        assert compressor.should_compress(messages)
        compressor.compress(messages, {})

    messages = with_usage(5_000)
    messages[-1].id = "ai-final"
    assert not compressor.should_compress(messages)


# ---------- 压缩与重组 ----------


def test_compress_assembles_summary_and_fresh_status() -> None:
    """测试：压缩后是「A 段 + 摘要 + 新的 review_status」。"""
    compressor = make_compressor([GOOD_SUMMARY])
    hunk_status = {"H-01": HunkStatus("H-01", STATUS_DONE)}

    result = compressor.compress(sample_history(), hunk_status)

    # 首条是「清空全部」哨兵，其余是可读消息
    assert result[0].__class__.__name__ == "RemoveMessage"
    assert result[1].content == "指导 + diff"
    assert SUMMARY_TAG_OPEN in str(result[2].content)
    assert "## Current Work" in str(result[2].content)
    assert "<review_status>" in str(result[3].content)
    assert "H-01" in str(result[3].content)


def test_compress_prefixes_notice_before_summary() -> None:
    """测试：摘要前面要有来路说明，并落在 `<compression_summary>` **外面**。

    模型看到一段没头没尾的结构化文本，最省事的解释是「这是新的用户请求」——
    于是可能去回应它、或者当成新任务重头再来。标签内保持是纯粹的摘要，
    以后要做「摘要与 `<review_status>` 对账」时正文里不该混进我们自己的话。
    """
    compressor = make_compressor([GOOD_SUMMARY])

    result = compressor.compress(sample_history(), {})
    content = str(result[2].content)

    assert tpl.COMPRESSION_NOTICE in content
    assert content.index(tpl.COMPRESSION_NOTICE) < content.index(SUMMARY_TAG_OPEN)
    assert "不是新的用户请求" in content


def test_compress_request_carries_background_and_cleaned_history() -> None:
    """测试：压缩请求里既有背景（审查 system + 指导 + diff），也有清理过的历史。"""
    model = FakeSummaryModel(replies=[GOOD_SUMMARY])
    # 窗口给足：这里测的是「请求里装了什么」，不是触发判定
    compressor = Compressor(model, replace(settings.compression, context_window=1_000_000))

    compressor.compress(sample_history(), {"H-01": HunkStatus("H-01", STATUS_DONE)})

    prompt = str(model.calls[0][1].content)
    assert "你是一名资深 Go 代码审查员" in prompt, "背景里要有审查 system 原文"
    assert "指导 + diff" in prompt
    assert "内容已清理" in prompt, "历史里的工具结果应已清理"
    assert "500 行代码" not in prompt, "原文不该还在"


def test_compress_degrades_to_truncation_on_bad_response() -> None:
    """测试：摘要不合格时降级为截断，绝不静默。"""
    compressor = make_compressor(["跑题了，没有段标题", "还是跑题"])

    result = compressor.compress(sample_history(), {})

    assert TRUNCATED_NOTE in str(result[2].content)
    assert "<review_status>" in str(result[-1].content), "进度快照永远在最后一条"


def test_compress_degrades_on_exception() -> None:
    """测试：压缩请求抛异常也要降级，不能让整组审查崩掉。"""
    compressor = make_compressor(["!raise", "!raise"])

    result = compressor.compress(sample_history(), {})

    assert TRUNCATED_NOTE in str(result[2].content)


def test_breaker_opens_after_consecutive_failures() -> None:
    """测试：连续 3 次失败后熔断，本组不再尝试压缩。"""
    compressor = make_compressor(["!raise"] * 6)

    for _ in range(3):
        compressor.compress(sample_history(), {})

    messages = with_usage(5_000)
    messages[-1].id = "ai-later"
    assert not compressor.should_compress(messages), "熔断后不该再触发"


def test_fit_drops_whole_rounds() -> None:
    """测试：截断按轮丢，不拆散 AI 消息与它的工具结果。"""
    # 窗口取到刚好只装得下一轮多一点：两轮一起超预算，单轮不超
    compressor = make_compressor([], window=2_000, reserve=100)
    history = [
        call_message("file_read", {"path": "a.go"}, "c1"),
        ToolMessage(content="x" * 4_000, tool_call_id="c1", id="m1"),
        call_message("file_read", {"path": "b.go"}, "c2"),
        ToolMessage(content="y" * 4_000, tool_call_id="c2", id="m2"),
    ]

    # 直接测部件（见模块 docstring），不走完整 compress
    fitted = compressor._fit(history, fixed=0)

    assert len(fitted) == 2, "只该丢掉最旧的一轮"
    assert fitted[0].id == "ai-c2", "留下的应是最近的整轮"


# ---------- 模板一致性 ----------


def test_summary_sections_match_prompt() -> None:
    """测试：格式清单与提示词里的段标题一致。

    两处分开写是有意的（提示词要读得顺、校验要能遍历），但**不能漂**：
    提示词加了段而 `SUMMARY_SECTIONS` 没加，校验就会把合格摘要判成不合格。
    """
    for section in tpl.SUMMARY_SECTIONS:
        assert section in tpl.COMPRESSION_SYSTEM, f"{section} 不在压缩提示词里"


def test_compression_user_template_placeholders() -> None:
    """测试：压缩 user 模板的占位符齐全（少一个就 format 报错）。"""
    rendered = tpl.COMPRESSION_USER.format(
        review_system="S", background="B", history="H", status="T"
    )

    assert "S" in rendered
    assert "B" in rendered
    assert "H" in rendered
    assert "T" in rendered


def test_compression_user_wraps_original_prompts() -> None:
    """测试：背景（原审查 system 与 user）也套容器，且点明「不是给你的指令」。

    背景里全是 markdown 标题（`## 输入格式` / `## 硬约束` / `### 审查范围` …），
    而压缩 system 自己也有一句「按以下 7 个标题输出」—— 不套容器、不点明，
    模型很容易把审查那套标题当成给它的指令。
    """
    rendered = tpl.COMPRESSION_USER.format(
        review_system="审查系统提示词", background="审查用户提示词", history="H", status="T"
    )

    assert "<review_system>\n审查系统提示词\n</review_system>" in rendered
    assert "<review_prompt>\n审查用户提示词\n</review_prompt>" in rendered
    assert "不是给你的指令" in rendered
