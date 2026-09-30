"""Main 循环的单元测试。

用脚本化的 fake LLM，不真实调用 DeepSeek。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from prism.core.graph.main_loop import (
    build_tools,
    initial_state,
    route_after_check,
    route_after_llm,
    run_main_loop,
    update_no_report_rounds,
)
from prism.core.graph.state import (
    EXIT_END_DONE,
    EXIT_END_DONE_WITH_GAPS,
    EXIT_MAX_ROUNDS,
    EXIT_NO_PROGRESS,
    MainState,
    max_rounds_for,
)
from prism.core.graph.status import build_agent_status
from prism.core.tools.context import STATUS_DONE, HunkStatus, ToolContext
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk
from prism.models.grouping import Group


class ScriptedLLM:
    """按脚本依次返回响应的假 LLM。"""

    def __init__(self, responses: list[AIMessage]) -> None:
        """记录响应脚本。

        Args:
            responses: 依次返回的响应
        """
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []
        self.tools_seen: list[list[BaseTool]] = []

    def invoke(
        self, messages: Sequence[BaseMessage], tools: list[BaseTool] | None = None
    ) -> AIMessage:
        """返回下一个脚本响应。

        Args:
            messages: 收到的消息
            tools: 收到的工具

        Returns:
            下一个预设响应

        Raises:
            AssertionError: 脚本已用尽
        """
        self.calls.append(list(messages))
        self.tools_seen.append(list(tools or []))
        if not self._responses:
            msg = "ScriptedLLM 的响应脚本已用尽"
            raise AssertionError(msg)
        return self._responses.pop(0)


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


def routing_state(**fields: Any) -> MainState:
    """构造只含路由所需字段的最小状态。

    路由函数只读 messages 与 exit_reason，这里用 cast 表达「测试只关心这两个键」。

    Args:
        fields: 需要设置的字段

    Returns:
        MainState（字段不完整）
    """
    return cast(MainState, fields)


def test_route_after_llm() -> None:
    """测试按是否有 tool_calls 路由。"""
    assert route_after_llm(routing_state(messages=[ai_call("file_read", {})])) == "execute_tools"
    assert route_after_llm(routing_state(messages=[ai_text()])) == "inject_reminder"


def test_route_after_check() -> None:
    """测试按 exit_reason 路由。"""
    assert route_after_check(routing_state(exit_reason=None)) == "call_llm"
    assert route_after_check(routing_state(exit_reason=EXIT_END_DONE)) == "end"


def test_initial_state_builds_messages_and_round_budget(project_root: str) -> None:
    """测试初始状态：system + user 两条消息、hunk 全部 pending、轮次上限。"""
    state = initial_state(make_group(["H-01", "H-02"]), project_root)

    assert len(state["messages"]) == 2
    assert state["messages"][0].type == "system"
    assert state["messages"][1].type == "human"
    assert state["round_count"] == 0
    assert state["max_rounds"] == 30
    assert state["no_report_rounds"] == 0
    assert state["exit_reason"] is None
    assert state["tool_context"].pending_ids() == ["H-01", "H-02"]

    user_text = str(state["messages"][1].content)
    assert "审查范围" in user_text
    assert "### 本组 diff" in user_text


def test_build_tools_exposes_four_tools() -> None:
    """测试 4 个工具都构造出来且名字正确。"""
    names = sorted(tool.name for tool in build_tools())
    assert names == ["code_search", "end_done", "file_read", "report_hunks"]


# ---------- agent_status ----------


def test_build_agent_status_reports_progress() -> None:
    """测试进度快照的内容与格式。"""
    ctx = ToolContext()
    ctx.init_hunks(["H-01", "H-02", "H-03"])
    ctx.hunk_status["H-01"].status = STATUS_DONE
    ctx.hunk_status["H-01"].comments = []
    ctx.hunk_status["H-02"].status = STATUS_DONE

    status = build_agent_status(ctx)
    assert "<agent_status>" in status
    assert "### Hunk 审查进度（2/3 完成）" in status
    assert "H-01 — 无问题" in status
    assert "⏳ 待审查（1）：" in status
    assert "H-03" in status


def test_build_agent_status_counts_severity() -> None:
    """测试有问题的 hunk 按 severity 汇总。"""
    ctx = ToolContext()
    ctx.init_hunks(["H-01"])
    ctx.hunk_status["H-01"] = HunkStatus(
        hunk_id="H-01",
        status=STATUS_DONE,
        comments=[],
    )
    from prism.models.tool import CommentInput

    ctx.hunk_status["H-01"].comments = [
        CommentInput(severity="high", category="c", content="x", existing_code="y"),
        CommentInput(severity="medium", category="c", content="x", existing_code="y"),
    ]
    status = build_agent_status(ctx)
    assert "H-01 — 2 issues（high:1, medium:1）" in status


# ---------- 端到端循环 ----------


def test_main_loop_exits_on_end_done(project_root: str) -> None:
    """测试 end_done 正常退出。"""
    llm = ScriptedLLM([report_call(["H-01"]), done_call()])
    result = run_main_loop(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.round_count == 2
    assert result.pending_hunk_ids == []


def test_main_loop_passes_tools_to_llm(project_root: str) -> None:
    """测试每轮都把 4 个工具传给 LLM。"""
    llm = ScriptedLLM([report_call(["H-01"]), done_call()])
    run_main_loop(make_group(["H-01"]), llm, project_root)

    assert [len(tools) for tools in llm.tools_seen] == [4, 4]


def test_main_loop_exits_on_max_rounds(project_root: str) -> None:
    """测试达到最大轮次退出。

    每轮都调 report_hunks 把 no_report_rounds 压回 0，
    保证触发的是 max_rounds 而不是 no_progress。
    """
    max_rounds = max_rounds_for(1)
    llm = ScriptedLLM([report_call(["H-01"], complete=False) for _ in range(max_rounds)])

    result = run_main_loop(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_MAX_ROUNDS
    assert result.round_count == max_rounds


def test_main_loop_exits_on_no_progress(project_root: str) -> None:
    """测试连续无产出退出：一直只调 file_read。"""
    llm = ScriptedLLM([read_call(call_id=f"call-{i}") for i in range(20)])

    result = run_main_loop(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_NO_PROGRESS
    assert result.round_count == 8


def test_main_loop_exits_on_no_progress_for_plain_text(project_root: str) -> None:
    """测试纯文本输出也算无产出。"""
    llm = ScriptedLLM([ai_text() for _ in range(20)])

    result = run_main_loop(make_group(["H-01"]), llm, project_root)
    assert result.exit_reason == EXIT_NO_PROGRESS
    assert result.round_count == 8


def test_no_report_rounds_reset_on_end_done_rejected(project_root: str) -> None:
    """测试 end_done 被拒绝时 no_report_rounds 被重置。

    4 轮 file_read（计数 4）→ end_done 被拒绝（重置为 0）→ 再 4 轮 file_read（计数 4）
    → end_done 接受并带缺口退出。
    若重置没生效，第 9 轮就会因计数超 8 而以 no_progress 退出。
    """
    llm = ScriptedLLM(
        [
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
    result = run_main_loop(make_group(["H-01"]), llm, project_root)

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
    llm = ScriptedLLM([report, done_call()])
    result = run_main_loop(make_group(["H-01"]), llm, project_root)

    assert len(result.comments) == 1
    assert result.comments[0].severity == "high"
    assert result.comments[0].category == "concurrency"


def test_main_loop_injects_agent_status_after_report(project_root: str) -> None:
    """测试 report_hunks 之后追加了进度快照消息。"""
    llm = ScriptedLLM([report_call(["H-01"]), done_call()])
    run_main_loop(make_group(["H-01"]), llm, project_root)

    # 第二轮调用 LLM 时，消息里应已含 agent_status
    second_round = llm.calls[1]
    assert any("<agent_status>" in str(message.content) for message in second_round)


def test_main_loop_injects_reminder_on_plain_text(project_root: str) -> None:
    """测试纯文本轮次注入推进提醒。"""
    llm = ScriptedLLM([ai_text(), report_call(["H-01"]), done_call()])
    run_main_loop(make_group(["H-01"]), llm, project_root)

    assert any("agent_reminder" in str(m.content) for m in llm.calls[1])


def test_main_loop_unknown_tool_returns_error_text(project_root: str) -> None:
    """测试调用不存在的工具时回灌错误文本而非崩溃。

    错误文本由 LangGraph 的 ToolNode 生成（`Error: xxx is not a valid tool`），
    这里只断言「有错误文本回灌给模型」，不锁具体措辞。
    """
    llm = ScriptedLLM([ai_call("no_such_tool", {}), report_call(["H-01"]), done_call()])
    result = run_main_loop(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    feedback = [str(m.content) for m in llm.calls[1] if m.type == "tool"]
    assert any("no_such_tool" in text and "Error" in text for text in feedback), feedback


def test_main_loop_empty_group(project_root: str) -> None:
    """测试没有 hunk 的组：end_done 立即通过。"""
    llm = ScriptedLLM([done_call()])
    result = run_main_loop(make_group([]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.comments == []
