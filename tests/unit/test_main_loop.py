"""审查循环的单元测试。

用脚本化的 fake ChatModel，不真实调用 DeepSeek。

注意：假模型必须是 **`BaseChatModel`** —— 循环现在由 `create_agent` 驱动，
框架会调用 `model.bind_tools(...).invoke(...)`，旧的自定义 ChatInvoker
Protocol 已经接不上了。
"""

from pathlib import Path
from typing import Any, cast, override

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from prism.core.agent.agent import build_tools, initial_state, run_review_agent
from prism.core.agent.middleware import end_done_reset, should_exit, update_no_report_rounds
from prism.core.agent.state import (
    EXIT_END_DONE,
    EXIT_END_DONE_WITH_GAPS,
    EXIT_MAX_ROUNDS,
    EXIT_NO_PROGRESS,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    HunkStatus,
    ReviewState,
    max_rounds_for,
    pending_ids,
)
from prism.core.agent.status import build_review_status
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.compression import CompressionConfig
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput


class ScriptedChatModel(BaseChatModel):
    """按脚本依次返回响应的假 ChatModel。

    同时记录每次模型调用收到的消息与工具，供断言使用。
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    tools_seen: list[list[BaseTool]] = Field(default_factory=list)

    @property
    @override
    def _llm_type(self) -> str:
        """模型类型标识。

        Returns:
            固定字符串
        """
        return "scripted"

    @override
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        """记录被绑定的工具，原样返回自己。

        Args:
            tools: 工具列表
            tool_choice: 选择策略（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            self
        """
        self.tools_seen.append(list(tools))
        return self

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

        Raises:
            AssertionError: 脚本已用尽（顺带防住死循环）
        """
        self.calls.append(list(messages))
        if not self.responses:
            msg = "ScriptedChatModel 的响应脚本已用尽"
            raise AssertionError(msg)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def ai_text(content: str = "（无工具调用）") -> AIMessage:
    """构造纯文本响应。

    Args:
        content: 响应文本

    Returns:
        AIMessage
    """
    return AIMessage(content=content)


def ai_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    """构造带单个 tool_call 的响应。

    Args:
        name: 工具名
        args: 工具参数
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def read_call(call_id: str = "call-1") -> AIMessage:
    """构造一次 file_read 调用。

    Args:
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    return ai_call("file_read", {"path": "a.go", "start_line": 1, "end_line": 2}, call_id)


def report_call(hunk_ids: list[str], complete: bool = True, call_id: str = "call-r") -> AIMessage:
    """构造一次 report_hunks 调用。

    Args:
        hunk_ids: 要报告的 hunk 编号
        complete: 是否标记完成
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    reports = [{"hunk_id": hunk_id, "complete": complete, "comments": []} for hunk_id in hunk_ids]
    return ai_call("report_hunks", {"reports": reports}, call_id)


def done_call(call_id: str = "call-d") -> AIMessage:
    """构造一次 end_done 调用。

    Args:
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    return ai_call("end_done", {}, call_id)


def make_group(hunk_ids: list[str], group_id: str = "group-001") -> Group:
    """构造测试用审查单元。

    Args:
        hunk_ids: hunk 编号
        group_id: 分组编号

    Returns:
        Group 对象
    """
    hunks = [
        HunkWithSymbol(
            part=Hunk(old_start=1, old_lines=1, new_start=1, new_lines=1, content="@@ -1 +1 @@"),
            file_path="a.go",
            symbol=Symbol(name="Run", kind="function", start_line=1, end_line=9),
            hunk_id=hunk_id,
            head_lines=(1, 9),
            renamed_from=None,
            deleted=False,
        )
        for hunk_id in hunk_ids
    ]
    return Group(group_id=group_id, package_path=".", files=["a.go"], hunks=hunks)


@pytest.fixture
def project_root(tmp_path: Path) -> str:
    """构造一个含 a.go 的临时项目根。

    Returns:
        项目根路径字符串
    """
    (tmp_path / "a.go").write_text("package main\n\nfunc Run() {}\n", encoding="utf-8")
    return str(tmp_path)


# ---------- 纯函数 ----------


def test_max_rounds_for_uses_documented_formula() -> None:
    """测试 max_rounds = max(30, hunk 数 × 5)。"""
    assert max_rounds_for(0) == 30
    assert max_rounds_for(6) == 30
    assert max_rounds_for(7) == 35
    assert max_rounds_for(20) == 100


def test_update_no_report_rounds_resets_on_progress_tools() -> None:
    """测试 report_hunks / end_done 重置计数。"""
    assert update_no_report_rounds(4, [{"name": "report_hunks"}]) == 0
    assert update_no_report_rounds(4, [{"name": "end_done"}]) == 0
    assert update_no_report_rounds(4, [{"name": "file_read"}, {"name": "report_hunks"}]) == 0


def test_update_no_report_rounds_increments_on_exploration() -> None:
    """测试纯探索工具与无工具调用时递增。"""
    assert update_no_report_rounds(4, [{"name": "file_read"}]) == 5
    assert update_no_report_rounds(4, [{"name": "code_search"}]) == 5
    assert update_no_report_rounds(4, []) == 5


def state_with(**fields: Any) -> ReviewState:
    """构造只含判定所需字段的最小状态。

    Args:
        fields: 需要设置的字段

    Returns:
        ReviewState（字段不完整）
    """
    return cast(ReviewState, fields)


def test_should_exit_false_by_default() -> None:
    """测试默认不退出。"""
    state = state_with(
        hunk_status={"H-01": HunkStatus("H-01", STATUS_PENDING)},
        end_done_count=0,
        no_report_rounds=0,
    )
    assert should_exit(state) is False


def test_should_exit_on_accepted_end_done() -> None:
    """测试 end_done 被接受（含带缺口）时退出。"""
    done = {"H-01": HunkStatus("H-01", STATUS_DONE)}
    assert should_exit(state_with(hunk_status=done, end_done_count=1, no_report_rounds=0)) is True

    pending = {"H-01": HunkStatus("H-01", STATUS_PENDING)}
    once = state_with(hunk_status=pending, end_done_count=1, no_report_rounds=0)
    twice = state_with(hunk_status=pending, end_done_count=2, no_report_rounds=0)
    assert should_exit(once) is False
    assert should_exit(twice) is True


def test_should_exit_on_no_progress() -> None:
    """测试连续无产出到阈值时退出。"""
    state = state_with(hunk_status={}, end_done_count=0, no_report_rounds=8)
    assert should_exit(state) is True


def test_end_done_reset_on_progress() -> None:
    """测试有新 hunk 审完时重置 end_done 计数（纠正机会绑到进展上）。

    回归用例：计数从不重置时，早先一次误调的 end_done 会把纠正机会提前
    耗尽，后面真误判时直接被接受并带缺口退出。
    """
    state = state_with(
        hunk_status={
            "H-01": HunkStatus("H-01", STATUS_DONE),
            "H-02": HunkStatus("H-02", STATUS_PENDING),
        },
        end_done_count=1,
        end_done_mark=0,
    )
    assert end_done_reset(state) == {"end_done_count": 0, "end_done_mark": 1}


def test_end_done_reset_needs_actual_progress() -> None:
    """测试「审查中」不算进展，不重置。"""
    state = state_with(
        hunk_status={"H-01": HunkStatus("H-01", STATUS_IN_PROGRESS)},
        end_done_count=1,
        end_done_mark=0,
    )
    assert end_done_reset(state) == {}


def test_end_done_reset_is_idempotent() -> None:
    """测试同一批进度只重置一次：mark 已经追上时不再重置。

    否则每轮都会把计数清零，拒绝机制形同虚设。
    """
    state = state_with(
        hunk_status={"H-01": HunkStatus("H-01", STATUS_DONE)},
        end_done_count=1,
        end_done_mark=1,
    )
    assert end_done_reset(state) == {}


def test_should_exit_honours_overridden_end_done_count() -> None:
    """测试退出判定用传入的计数覆盖状态里的旧值。

    `review_progress` 本步刚重置过 end_done 时仍按旧计数判定的话，会把
    「刚有新进展」误判成「该收尾了」，带缺口提前退出。
    """
    pending = {"H-01": HunkStatus("H-01", STATUS_PENDING)}
    state = state_with(hunk_status=pending, end_done_count=2, no_report_rounds=0)

    assert should_exit(state) is True
    assert should_exit(state, end_done_count=0) is False


def test_initial_state_builds_user_message_and_round_budget(project_root: str) -> None:
    """测试初始状态：一条 user 消息、hunk 全部 pending、轮次上限。

    system 段不在这里 —— 它交给 `create_agent(system_prompt=...)`。
    """
    max_rounds = max_rounds_for(2)
    state = initial_state(make_group(["H-01", "H-02"]), max_rounds)

    assert len(state["messages"]) == 1
    assert state["messages"][0].type == "human"
    assert state["round_count"] == 0
    assert state["max_rounds"] == 30
    assert state["no_report_rounds"] == 0
    assert state["end_done_count"] == 0
    assert pending_ids(state["hunk_status"]) == ["H-01", "H-02"]

    user_text = str(state["messages"][0].content)
    assert "### 审查范围" in user_text
    assert "<diff>" in user_text
    # diff 块在前、要求块在后 —— 模型决策前最后读到的是一段短要求
    assert user_text.index("<diff>") < user_text.index("### 审查要求")


def test_build_tools_exposes_four_tools() -> None:
    """测试 4 个工具都构造出来且名字正确。"""
    names = sorted(tool.name for tool in build_tools())
    assert names == ["code_search", "end_done", "file_read", "report_hunks"]


# ---------- review_status ----------


def test_build_review_status_reports_progress() -> None:
    """测试进度快照的内容与格式。"""
    hunk_status = {
        "H-01": HunkStatus("H-01", STATUS_DONE),
        "H-02": HunkStatus("H-02", STATUS_DONE),
        "H-03": HunkStatus("H-03", STATUS_PENDING),
    }

    status = build_review_status(hunk_status)
    assert "<review_status>" in status
    assert "### Hunk 审查进度（2/3 完成）" in status
    assert "H-01 — 无问题" in status
    assert "⏳ 待审查（1）：" in status
    assert "H-03" in status


def test_build_review_status_counts_severity() -> None:
    """测试有问题的 hunk 按 severity 汇总。"""
    hunk_status = {
        "H-01": HunkStatus(
            "H-01",
            STATUS_DONE,
            comments=[
                CommentInput(severity="high", category="c", content="x", existing_code="y"),
                CommentInput(severity="medium", category="c", content="x", existing_code="y"),
            ],
        )
    }
    status = build_review_status(hunk_status)
    assert "H-01 — 2 issues（high:1, medium:1）" in status


# ---------- 端到端循环 ----------


def test_main_loop_exits_on_end_done(project_root: str) -> None:
    """测试 end_done 正常退出。"""
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])
    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.round_count == 2
    assert result.pending_hunk_ids == []


def test_main_loop_passes_tools_to_llm(project_root: str) -> None:
    """测试每轮都把 4 个工具传给 LLM。"""
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])
    run_review_agent(make_group(["H-01"]), llm, project_root)

    assert [len(tools) for tools in llm.tools_seen] == [4, 4]


def test_main_loop_exits_on_max_rounds(project_root: str) -> None:
    """测试达到最大轮次退出。

    每轮都调 report_hunks 把 no_report_rounds 压回 0，
    保证触发的是 max_rounds 而不是 no_progress。
    轮次上限现在由框架的 ModelCallLimitMiddleware 执行。
    """
    max_rounds = max_rounds_for(1)
    llm = ScriptedChatModel(
        responses=[report_call(["H-01"], complete=False) for _ in range(max_rounds)]
    )

    result = run_review_agent(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_MAX_ROUNDS
    assert result.round_count == max_rounds


def test_main_loop_exits_on_no_progress(project_root: str) -> None:
    """测试连续无产出退出：一直只调 file_read。"""
    llm = ScriptedChatModel(responses=[read_call(call_id=f"call-{i}") for i in range(20)])

    result = run_review_agent(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_NO_PROGRESS
    assert result.round_count == 8


def test_plain_text_after_tool_call_does_not_end_loop(project_root: str) -> None:
    """测试「先调工具、再返回纯文本」时循环不会提前结束。

    回归用例：`_last_tool_calls` 曾从后往前找**任意**一条带 tool_calls 的消息。
    纯文本响应的 `tool_calls` 是空列表（假值），于是它继续往前翻、翻到了上一轮
    的调用 —— `review_pacing` 误判「这轮有工具调用」，既不注入提醒也不
    `jump_to="model"`，框架按默认的「没调工具就结束」收尾，
    而我们的三个退出条件（end_done / 轮次耗尽 / 连续无产出）一个都不满足，
    `exit_reason_of` 返回 None，整个审查以 ValueError 崩掉。

    这条路径要「纯文本紧跟在一次工具调用之后」才会走到，所以常态跑不出来 ——
    实测在真实审查里偶发过一次，表现为 `ValueError: 读不出 exit_reason`。
    """
    llm = ScriptedChatModel(
        responses=[
            report_call(["H-01"]),  # 调工具
            ai_text(),  # 纯文本 —— 旧实现在这里把循环结束掉
            report_call(["H-02"]),
            done_call(),
        ]
    )
    result = run_review_agent(make_group(["H-01", "H-02"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.round_count == 4, "纯文本那一轮被当成「有工具调用」，循环提前结束了"


def test_main_loop_exits_on_no_progress_for_plain_text(project_root: str) -> None:
    """测试纯文本输出也算无产出。

    这条同时验证了 prism 与框架默认行为的冲突点：框架遇到「模型没调工具」
    会直接结束循环，是 middleware 返回 jump_to="model" 把它拉回来的。
    """
    llm = ScriptedChatModel(responses=[ai_text() for _ in range(20)])

    result = run_review_agent(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_NO_PROGRESS
    assert result.round_count == 8


def test_no_report_rounds_reset_on_end_done_rejected(project_root: str) -> None:
    """测试 end_done 被拒绝时 no_report_rounds 被重置。

    4 轮 file_read（计数 4）→ end_done 被拒绝（重置为 0）→ 再 4 轮 file_read（计数 4）
    → end_done 接受并带缺口退出。
    若重置没生效，第 9 轮就会因计数超 8 而以 no_progress 退出。
    """
    llm = ScriptedChatModel(
        responses=[
            read_call("call-1"),
            read_call("call-2"),
            read_call("call-3"),
            read_call("call-4"),
            done_call("call-5"),
            read_call("call-6"),
            read_call("call-7"),
            read_call("call-8"),
            read_call("call-9"),
            done_call("call-10"),
        ]
    )
    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE_WITH_GAPS
    assert result.round_count == 10
    assert result.pending_hunk_ids == ["H-01"]


def test_main_loop_collects_comments(project_root: str) -> None:
    """测试评论被收集到结果里。"""
    report = ai_call(
        "report_hunks",
        {
            "reports": [
                {
                    "hunk_id": "H-01",
                    "complete": True,
                    "comments": [
                        {
                            "severity": "high",
                            "category": "concurrency",
                            "content": "goroutine 缺少退出路径",
                            "existing_code": "go func() {}()",
                        }
                    ],
                }
            ]
        },
    )
    llm = ScriptedChatModel(responses=[report, done_call()])
    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert len(result.comments) == 1
    assert result.comments[0].severity == "high"
    assert result.comments[0].category == "concurrency"


def test_main_loop_injects_review_status_after_report(project_root: str) -> None:
    """测试 report_hunks 之后追加了进度快照消息。"""
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])
    run_review_agent(make_group(["H-01"]), llm, project_root)

    # 第二轮调用 LLM 时，消息里应已含 review_status
    second_round = llm.calls[1]
    assert any("<review_status>" in str(message.content) for message in second_round)


def with_usage(message: AIMessage, tokens: int) -> AIMessage:
    """给脚本响应挂上 `usage_metadata`。

    压缩的触发只看 API 报告的 token 数，而假模型默认不带 usage —— 补上它也是
    在测试里「逼出压缩」的唯一手段（真实场景下这个字段由 API 填）。

    Args:
        message: 原始响应
        tokens: 报告的总 token 数

    Returns:
        带 usage 的响应副本
    """
    usage: Any = {"input_tokens": tokens - 1, "output_tokens": 1, "total_tokens": tokens}
    return message.model_copy(update={"usage_metadata": usage})


def test_main_loop_compresses_when_window_is_tight(project_root: str) -> None:
    """测试：逼近窗口时压缩历史，且压缩后 A 段与权威进度都还在。

    这一段脚本同时喂了两条路径：第 1 条是审查响应、第 2 条是**压缩请求**的响应
    （压缩用的是同一个模型）、第 3 条是压缩后下一轮的审查响应。
    """
    # 至少命中 3 个段标题才算合格摘要（见 compression.MIN_SECTIONS）
    summary = (
        "## Primary Request\n审查 . 包的 1 个 hunk\n\n"
        "## Completed Hunks\n- H-01：无问题\n\n"
        "## Current Work\n正在审查 H-01。"
    )
    llm = ScriptedChatModel(
        responses=[
            with_usage(report_call(["H-01"]), 5_000),
            AIMessage(content=summary),
            done_call(),
        ]
    )

    run_review_agent(
        make_group(["H-01"]),
        llm,
        project_root,
        compression=CompressionConfig(context_window=1_000, reserve_tokens=100),
    )

    assert len(llm.calls) == 3, "审查 → 压缩 → 继续审查"
    # 压缩请求：system 是压缩专用那份，user 里带着背景与清理过的历史
    request = llm.calls[1]
    assert "压缩" in str(request[0].content)
    assert "<hunk id=" in str(request[1].content), "背景里要有 diff"
    # 压缩后的下一轮：A 段 + 摘要 + 权威进度，三者都在
    after = llm.calls[2]
    joined = "\n".join(str(message.content) for message in after)
    assert "<hunk id=" in joined, "A 段（指导 + diff）必须原样保留"
    assert "<compression_summary>" in joined
    assert "<review_status>" in joined
    assert "500 行代码" not in joined


def test_main_loop_injects_reminder_on_plain_text(project_root: str) -> None:
    """测试纯文本轮次注入推进提醒。"""
    llm = ScriptedChatModel(responses=[ai_text(), report_call(["H-01"]), done_call()])
    run_review_agent(make_group(["H-01"]), llm, project_root)

    assert any("agent_reminder" in str(m.content) for m in llm.calls[1])


def test_main_loop_unknown_tool_returns_error_text(project_root: str) -> None:
    """测试调用不存在的工具时回灌错误文本而非崩溃。

    错误文本由框架的 ToolNode 生成，这里只断言
    「有错误文本回灌给模型」，不锁具体措辞。
    """
    llm = ScriptedChatModel(
        responses=[ai_call("no_such_tool", {}), report_call(["H-01"]), done_call()]
    )
    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    feedback = [str(m.content) for m in llm.calls[1] if m.type == "tool"]
    assert any("no_such_tool" in text for text in feedback), feedback


def test_main_loop_feeds_back_schema_error_and_continues(project_root: str) -> None:
    """测试工具参数不合法时，校验错误被回灌给模型，循环继续而不是崩掉。

    回归用例：report_hunks 的参数曾是 `list[dict]`，模型漏填 severity 时
    静默变成空字符串（实测 55% 的评论如此）。改成 Pydantic 后由 schema 拦下。
    """
    bad = ai_call(
        "report_hunks",
        {
            "reports": [
                {
                    "hunk_id": "H-01",
                    "complete": True,
                    "comments": [{"category": "x", "content": "y", "existing_code": "z"}],
                }
            ]
        },
    )
    llm = ScriptedChatModel(responses=[bad, report_call(["H-01"]), done_call()])
    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    feedback = [str(m.content) for m in llm.calls[1] if m.type == "tool"]
    assert any("severity" in text for text in feedback), feedback


def test_main_loop_survives_parallel_report_hunks(project_root: str) -> None:
    """测试一轮里并行调两个 report_hunks 不崩，且两个 hunk 都记上了。

    回归用例：模型一轮可以发多个工具调用，它们在**同一个 super-step** 里跑，
    返回的 Command 同时写 hunk_status。没有 reducer 时 LangGraph 直接抛
    `InvalidUpdateError: At key 'hunk_status': Can receive only one value per step`。

    实测 21 轮里 8 轮有并行调用，其中 1 轮就是两个 report_hunks，当场崩掉了
    一组审查。这里两条断言缺一不可：不崩（reducer 生效）+ 两个都 done
    （工具写的是增量而不是快照，否则后者会把前者盖掉）。
    """
    parallel = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_hunks",
                "args": {"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]},
                "id": "call-1",
                "type": "tool_call",
            },
            {
                "name": "report_hunks",
                "args": {"reports": [{"hunk_id": "H-02", "complete": True, "comments": []}]},
                "id": "call-2",
                "type": "tool_call",
            },
        ],
    )
    llm = ScriptedChatModel(responses=[parallel, done_call()])

    result = run_review_agent(make_group(["H-01", "H-02"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.pending_hunk_ids == [], "并行写入被后一份快照覆盖了"


def test_main_loop_empty_group(project_root: str) -> None:
    """测试没有 hunk 的组：end_done 立即通过。"""
    llm = ScriptedChatModel(responses=[done_call()])
    result = run_review_agent(make_group([]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.comments == []
