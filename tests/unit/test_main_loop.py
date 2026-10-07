"""审查循环的单元测试。

用脚本化的 fake ChatModel，不真实调用 DeepSeek。

注意：假模型必须是 **`BaseChatModel`** —— 循环现在由 `create_agent` 驱动，
框架会调用 `model.bind_tools(...).invoke(...)`，旧的自定义 ChatInvoker
Protocol 已经接不上了。
"""

import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast, override

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as langchain_tool
from langgraph.errors import GraphRecursionError
from pydantic import Field

from prism.config.settings import settings
from prism.core.agent.agent import build_review_agent, build_tools, initial_state, run_review_agent
from prism.core.agent.budget import STOP_COST, STOP_TOOL_FAILURES
from prism.core.agent.context import ReviewContext
from prism.core.agent.middleware import (
    DUPLICATE_CALL_MESSAGE,
    end_done_reset,
    should_exit,
    update_no_report_rounds,
)
from prism.core.agent.middleware import (
    _executed_before as executed_before,
)
from prism.core.agent.middleware import (
    _round_tool_results as round_tool_results,
)
from prism.core.agent.state import (
    EXIT_BUDGET_COST,
    EXIT_BUDGET_TIME,
    EXIT_END_DONE,
    EXIT_END_DONE_WITH_GAPS,
    EXIT_ERROR,
    EXIT_NO_PROGRESS,
    EXIT_RECURSION,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    HunkStatus,
    ReviewState,
    exit_reason_of,
    pending_ids,
)
from prism.core.agent.status import build_review_status
from prism.core.diff.loader import build_diff_index
from prism.core.diff.parser import parse_diff_text
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# 一次调用的用量：输入 1000（命中 800 / 未命中 200）、输出 100。
# 数字取得好算：按默认价格（未命中 4 元/百万、命中 0.5、输出 12）一次调用
# = (200*4 + 800*0.5 + 100*12) / 1e6 = 0.0024 元
USAGE_METADATA: dict[str, Any] = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "total_tokens": 1100,
    "input_token_details": {"cache_read": 800},
}


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


def ai_call(
    name: str, args: dict[str, Any], call_id: str = "call-1", *, usage: bool = True
) -> AIMessage:
    """构造带单个 tool_call 的响应。

    Args:
        name: 工具名
        args: 工具参数
        call_id: 调用 ID
        usage: 是否带上用量元数据。默认带 —— 真实模型都会回，而预算那条路
            完全依赖它（不带就等于测「拿不到用量」那条分支）

    Returns:
        AIMessage
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=USAGE_METADATA if usage else None,
    )


def read_call(call_id: str = "call-1") -> AIMessage:
    """构造一次 file_read 调用。

    Args:
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    return ai_call("file_read", {"path": "a.go", "start_line": 1, "end_line": 2}, call_id)


def report_call(
    hunk_ids: list[str], complete: bool = True, call_id: str = "call-r", *, usage: bool = True
) -> AIMessage:
    """构造一次 report_hunks 调用。

    Args:
        hunk_ids: 要报告的 hunk 编号
        complete: 是否标记完成
        call_id: 调用 ID
        usage: 是否带用量元数据

    Returns:
        AIMessage
    """
    reports = [{"hunk_id": hunk_id, "complete": complete, "comments": []} for hunk_id in hunk_ids]
    return ai_call("report_hunks", {"reports": reports}, call_id, usage=usage)


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
    state = initial_state(make_group(["H-01", "H-02"]))

    assert len(state["messages"]) == 1
    assert state["messages"][0].type == "human"
    assert state["round_count"] == 0
    assert state["no_report_rounds"] == 0
    assert state["end_done_count"] == 0
    assert pending_ids(state["hunk_status"]) == ["H-01", "H-02"]

    user_text = str(state["messages"][0].content)
    assert "### 审查范围" in user_text
    assert "<diff>" in user_text
    # diff 块在前、要求块在后 —— 模型决策前最后读到的是一段短要求
    assert user_text.index("<diff>") < user_text.index("### 你的主任务")
    # 没有静态发现时整段省略（工作方式那句会提到段名，所以断言的是段标题）
    assert "### 静态分析提示" not in user_text
    assert "\n\n\n" not in user_text


def test_build_tools_exposes_five_tools() -> None:
    """测试 5 个工具都构造出来且名字正确。"""
    names = sorted(tool.name for tool in build_tools())
    assert names == ["code_search", "end_done", "file_read", "file_read_diff", "report_hunks"]


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
    """测试每轮都把 5 个工具传给 LLM。"""
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])
    run_review_agent(make_group(["H-01"]), llm, project_root)

    assert [len(tools) for tools in llm.tools_seen] == [5, 5]


def test_diff_index_reaches_the_tool(project_root: str) -> None:
    """测试 diff 索引从 run_review_agent 一路传到了工具手里。

    回归用例：`run_review_agent` 里少传一个 `diffs=...`，file_read_diff 只会
    回「不可用」—— 没有异常、没有日志红字，只是跨文件查询**整条静默失效**。
    这里断言渲染出来的 diff 真的出现在下一轮请求里（走的是真实的
    ReviewContext 注入路径，不是直接构造上下文调工具）。
    """
    sample = parse_diff_text(
        "diff --git a/b.go b/b.go\n--- a/b.go\n+++ b/b.go\n@@ -1 +1 @@\n-old()\n+new()\n"
    )
    llm = ScriptedChatModel(
        responses=[
            ai_call("file_read_diff", {"path": "b.go"}, "c1"),
            report_call(["H-01"]),
            done_call(),
        ]
    )

    run_review_agent(
        make_group(["H-01"]),
        llm,
        project_root,
        diff_index=build_diff_index(sample),
    )

    second_request = "\n".join(str(message.content) for message in llm.calls[1])
    assert "@@ -1,1 +1,1 @@" in second_request
    assert "+new()" in second_request


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
        compression=replace(settings.compression, context_window=1_000, reserve_tokens=100),
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


# ---------- 工具异常不许炸掉整组 ----------


@langchain_tool(description="一个必定抛异常的工具（兜底用例专用）。")
def exploding_tool(path: str) -> str:
    """必定抛异常。

    Args:
        path: 工具参数（模型看不到这个 docstring，描述走 description=）

    Returns:
        永不返回

    Raises:
        RuntimeError: 总是
    """
    msg = f"工具内部炸了：{path}"
    raise RuntimeError(msg)


@pytest.fixture
def exploding(monkeypatch: pytest.MonkeyPatch) -> None:
    """往图里多塞一个「必炸」的工具。

    四个真工具今天都自己兜住了错误，所以要验兜底只能造一个故障源。
    这条用例保的不是某一次调用，是**这一类**：以后加工具、接 MCP、
    或者哪个库换了抛法，异常不该带走整组。

    Args:
        monkeypatch: pytest 的补丁工具
    """

    def with_exploding() -> list[BaseTool]:
        return [exploding_tool, *build_tools()]

    monkeypatch.setattr("prism.core.agent.agent.build_tools", with_exploding)


def test_tool_exception_does_not_kill_the_group(
    project_root: str, exploding: None, caplog: pytest.LogCaptureFixture
) -> None:
    """测试工具抛异常时整组审查照样跑完。

    回归用例（实测过）：框架默认的兜底只认 `ToolInvocationError`，
    `_default_handle_tool_errors` 最后一行是 `raise e` —— 于是工具里任何一个
    意外异常都会穿过图、穿过 `run_review_agent`，把**整条 review 带走**：
    后面所有组不再跑、前面已经审出来的评论也一起没（CLI 最后才渲染）。
    """
    llm = ScriptedChatModel(
        responses=[
            ai_call("exploding_tool", {"path": "a.go"}, "boom"),
            report_call(["H-01"]),
            done_call(),
        ]
    )

    with caplog.at_level(logging.ERROR):
        result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE, "整组应继续跑到正常收尾"
    assert "工具 exploding_tool 抛异常" in caplog.text, "异常要按 ERROR 记下来"
    # 回给模型的话要能让它自己做决定，而不是照着同一套参数重试
    second_request = "\n".join(str(message.content) for message in llm.calls[1])
    assert "工具内部执行失败" in second_request
    assert "不要用同样的参数重试" in second_request


def test_parallel_tool_calls_all_get_a_result_when_one_fails(
    project_root: str, exploding: None
) -> None:
    """测试并行工具调用里一个炸了，另一个的结果照样回给模型。

    一轮里模型可以并发调多个工具。LangGraph 要求**每个 tool_call 都有对应的
    工具结果**，少一个就是校验错误、整组崩 —— 兜底必须补齐炸掉的那一条，
    而不是把整轮丢掉。
    """
    llm = ScriptedChatModel(
        responses=[
            ai_call(
                "exploding_tool",
                {"path": "a.go"},
                "boom",
            ).model_copy(
                update={
                    "tool_calls": [
                        {
                            "name": "exploding_tool",
                            "args": {"path": "a.go"},
                            "id": "boom",
                            "type": "tool_call",
                        },
                        {
                            "name": "file_read",
                            "args": {"path": "a.go", "start_line": 1, "end_line": 2},
                            "id": "ok",
                            "type": "tool_call",
                        },
                    ]
                }
            ),
            report_call(["H-01"]),
            done_call(),
        ]
    )

    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    second_request = "\n".join(str(message.content) for message in llm.calls[1])
    assert "工具内部执行失败" in second_request, "炸掉的那条要有回执"
    assert "     1: package main" in second_request, "没炸的那条结果也要在（a.go 内容带行号）"


# ---------- 图崩了也不能丢结果 ----------


class CrashingGraph:
    """假图：先把「审了一个 hunk」演进状态，再抛指定的异常。

    真的等 `recursion_limit` 触发要跑几百步，而这条路径的语义与步数无关 ——
    用假图才能确定性地验「异常不会带走已审出的评论」。
    """

    def __init__(self, error: Exception, *, chunks: int = 1) -> None:
        """记下要抛的异常。

        Args:
            error: 抛出的异常
            chunks: 抛异常前先吐几份状态（0 表示一开始就炸）
        """
        self._error = error
        self._chunks = chunks

    def stream(self, state: ReviewState, **_kwargs: Any) -> Iterator[ReviewState]:
        """模拟 `agent.stream(...)`：吐若干份状态之后抛异常。

        Args:
            state: 初始状态

        Yields:
            推进过的状态（第一个 hunk 标记完成并带一条评论）

        Raises:
            Exception: 构造时传入的异常
        """
        for _ in range(self._chunks):
            status = dict(state["hunk_status"])
            first = sorted(status)[0]
            status[first] = replace(
                status[first],
                status=STATUS_DONE,
                comments=[
                    CommentInput(
                        severity="high", category="bug", content="必须活下来", existing_code="x"
                    )
                ],
            )
            yield cast(ReviewState, {**state, "hunk_status": status})
        raise self._error


def install_crashing_graph(
    monkeypatch: pytest.MonkeyPatch, error: Exception, chunks: int = 1
) -> None:
    """把 `build_review_agent` 换成只返回假图的工厂。

    Args:
        monkeypatch: pytest 的 monkeypatch
        error: 假图要抛的异常
        chunks: 抛异常前先吐几份状态
    """
    monkeypatch.setattr(
        "prism.core.agent.agent.build_review_agent",
        lambda *_args, **_kwargs: CrashingGraph(error, chunks=chunks),
    )


def test_graph_recursion_keeps_partial_results(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图撞上 recursion_limit 时带缺口退出，**已审出的评论与状态都留住**。

    之前它会以 `GraphRecursionError` 穿出去，整条 review 跟着一起没（渲染在最后
    一步）—— 这条守的就是那个回归。
    """
    install_crashing_graph(monkeypatch, GraphRecursionError("Recursion limit of 25 reached"))

    result = run_review_agent(
        make_group(["H-01", "H-02"]), ScriptedChatModel(responses=[]), project_root
    )

    assert result.exit_reason == EXIT_RECURSION
    assert result.pending_hunk_ids == ["H-02"], "已经审完的 hunk 不该退回去"
    assert [c.content for c in result.hunk_comments["H-01"]] == ["必须活下来"]


def test_graph_error_keeps_partial_results(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """图抛别的异常时同样带缺口退出 —— 收口但不吞（ERROR + 堆栈照记）。"""
    install_crashing_graph(monkeypatch, RuntimeError("框架内部炸了"))

    result = run_review_agent(make_group(["H-01"]), ScriptedChatModel(responses=[]), project_root)

    assert result.exit_reason == EXIT_ERROR
    assert [c.content for c in result.hunk_comments["H-01"]] == ["必须活下来"]


def test_graph_crashed_before_any_state_still_exits_cleanly(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一份状态都没吐出来就崩：退回初始状态，仍然是干净退出而不是抛异常。"""
    install_crashing_graph(monkeypatch, RuntimeError("第一步就炸"), chunks=0)

    result = run_review_agent(make_group(["H-01"]), ScriptedChatModel(responses=[]), project_root)

    assert result.exit_reason == EXIT_ERROR
    assert result.pending_hunk_ids == ["H-01"]


def test_recursion_error_is_logged_as_error(
    project_root: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """recursion_limit 是「不该发生」的一类，日志级别必须是 ERROR。"""
    install_crashing_graph(monkeypatch, GraphRecursionError("Recursion limit of 25 reached"))

    with caplog.at_level("ERROR"):
        run_review_agent(make_group(["H-01"]), ScriptedChatModel(responses=[]), project_root)

    assert "recursion_limit" in caplog.text
    assert "不该发生" in caplog.text


# ---------- 预算 ----------


def tiny_budget(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """把单组预算压到极小，用来确定性地触发闸门。

    Args:
        monkeypatch: pytest 的 monkeypatch
        **overrides: 要覆盖的预算键（如 ``max_group_cost=0.001``）
    """
    review = replace(settings.review, **overrides)
    monkeypatch.setattr("prism.core.agent.middleware.settings", replace(settings, review=review))


def test_cost_budget_stops_the_group_and_keeps_comments(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """花费超预算 → 带缺口退出，**这一轮之前审出的评论照样留住**。

    值压得极小（1e-9 元）是刻意的：一轮的用量就够越过它，于是「闸门在第 2 轮
    之前生效」这件事与具体价格无关。
    """
    tiny_budget(monkeypatch, max_group_cost=1e-9)
    llm = ScriptedChatModel(responses=[report_call(["H-01"])])

    result = run_review_agent(make_group(["H-01", "H-02"]), llm, project_root)

    assert result.exit_reason == EXIT_BUDGET_COST
    assert result.round_count == 1, "第 2 轮之前就该停，不该再调模型"
    assert result.usage.calls == 1
    assert result.usage.cache_hit == 800, "命中与未命中要分开记，单价差一个数量级"
    assert result.pending_hunk_ids == ["H-02"]


def test_time_budget_stops_before_any_model_call(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时长超预算 → 一轮都不跑（时间从这一组开始时就在涨）。"""
    tiny_budget(monkeypatch, max_group_seconds=1e-9)
    llm = ScriptedChatModel(responses=[])

    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_BUDGET_TIME
    assert result.round_count == 0
    assert llm.calls == [], "时间闸在模型调用之前生效"
    assert result.pending_hunk_ids == ["H-01"]


def test_budget_not_hit_leaves_normal_exit(
    project_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """预算宽松时不影响正常退出 —— 闸门不能变成常驻行为。"""
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])

    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.exit_reason == EXIT_END_DONE
    assert result.usage.calls == 2


def test_missing_usage_is_counted_not_silently_zero(
    project_root: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """拿不到用量时记进 `missing` 并**警告一次** —— 否则成本闸会静默失效。

    这不是假想的降级路径：成本闸完全依赖 `usage_metadata`，而它是提供商给的。
    缺了它，钱那一档就永远不响，只剩时间闸与 recursion_limit。
    """
    tiny_budget(monkeypatch, max_group_cost=1e-9)
    llm = ScriptedChatModel(
        responses=[report_call(["H-01"], call_id="c1", usage=False), done_call()]
    )

    with caplog.at_level("WARNING"):
        result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert result.usage.missing == 1, "拿不到用量要计数，不能当成零"
    assert result.usage.calls == 1, "只有带回用量的那次调用算进成本"
    assert result.exit_reason == EXIT_END_DONE, "算不出钱就不该按钱停"
    assert "拿不到用量" in caplog.text


def test_end_done_wins_over_budget() -> None:
    """审完了就是审完了：`end_done` 已被受理时，退出原因不该报成预算。

    否则用户会看到一个**假的缺口** —— 明明全审完了，输出却说「因预算停止」。
    """
    state = initial_state(make_group(["H-01"]))
    state["end_done_count"] = 1
    state["stop_reason"] = STOP_COST
    state["hunk_status"] = {"H-01": HunkStatus(hunk_id="H-01", status=STATUS_DONE)}

    assert exit_reason_of(state) == EXIT_END_DONE


# ---------- 重复调用不执行 ----------


def tool_message(call_id: str, content: str) -> ToolMessage:
    """造一条工具结果消息。

    Args:
        call_id: 对应的调用 ID
        content: 结果内容

    Returns:
        ToolMessage
    """
    return ToolMessage(content=content, tool_call_id=call_id)


def test_executed_before_detects_a_live_repeat() -> None:
    """同样的调用 + 结果还在上下文里 → 判为重复。"""
    history = [
        ai_call("file_read", {"path": "a.go"}, "c1"),
        tool_message("c1", "     1: package main"),
    ]

    assert executed_before(history, "file_read", {"path": "a.go"}, exclude_id="c2")


def test_cleared_result_allows_a_refetch() -> None:
    """结果被压缩清成占位符 → **允许重新取**。

    这条是刻意留的口子：不然模型永远拿不回被清掉的内容，而「可重新获取」正是
    压缩清理那一步的承诺（`REFETCHABLE_TOOLS`）。
    """
    history = [
        ai_call("file_read", {"path": "a.go"}, "c1"),
        tool_message("c1", "[内容已清理：原为 file_read(path=a.go)，可重新获取]"),
    ]

    assert not executed_before(history, "file_read", {"path": "a.go"}, exclude_id="c2")


def test_different_args_are_not_a_repeat() -> None:
    """参数不同就是另一次调用（换个行范围重读是合法的）。"""
    history = [
        ai_call("file_read", {"path": "a.go", "start_line": 1, "end_line": 2}, "c1"),
        tool_message("c1", "     1: package main"),
    ]

    assert not executed_before(
        history, "file_read", {"path": "a.go", "start_line": 3, "end_line": 9}, exclude_id="c2"
    )


def test_current_call_does_not_count_itself() -> None:
    """当前这一次调用自己不算「执行过」—— 它就在历史那条 AI 消息里。"""
    history = [ai_call("file_read", {"path": "a.go"}, "c1")]

    assert not executed_before(history, "file_read", {"path": "a.go"}, exclude_id="c1")


def test_replayed_history_does_not_suppress_its_own_call() -> None:
    """同一份历史被重放（本次调用的 id 与结果都已经在里面）时，不能把自己判成重复。

    正常流程里「当前调用」的结果还没产生，所以 `alive` 那道筛子已经挡掉了它 ——
    但是一旦历史被重放（checkpoint 恢复、工具节点重跑），同一条 AI 消息会带着
    自己的结果再次出现。那时如果不按 id 排除掉本次调用，就会出现「历史里有结果，
    但这次的调用根本没执行」——模型拿到「已做过」却没有新结果。
    """
    history = [
        ai_call("file_read", {"path": "a.go"}, "c1"),
        tool_message("c1", "     1: package main"),
        ai_call("file_read", {"path": "a.go"}, "c1"),
    ]

    assert not executed_before(history, "file_read", {"path": "a.go"}, exclude_id="c1")


def test_result_not_yet_returned_is_not_a_repeat() -> None:
    """调用发出去了、结果还没回来（并行那一步）→ 不算重复。"""
    history = [ai_call("file_read", {"path": "a.go"}, "c1")]

    assert not executed_before(history, "file_read", {"path": "a.go"}, exclude_id="c9")


def test_duplicate_read_is_not_executed_end_to_end(project_root: str) -> None:
    """端到端：模型连着两轮读同一个文件的同一个范围，第二次不再执行。"""
    llm = ScriptedChatModel(
        responses=[read_call("c1"), read_call("c2"), report_call(["H-01"]), done_call("c3")],
    )
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )
    reads = {
        m.tool_call_id: str(m.content)
        for m in final["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id in {"c1", "c2"}
    }

    assert len(reads) == 2, "两次调用都要有结果（一次真读、一次说明）"
    assert "package main" in reads["c1"], "第一次真的读了文件"
    assert DUPLICATE_CALL_MESSAGE.splitlines()[0] in reads["c2"], "第二次被拦下，没再读一遍"


def test_report_hunks_is_never_deduped(project_root: str) -> None:
    """`report_hunks` 有副作用，不在去重白名单里 —— 同样的报告要能报两次。"""
    llm = ScriptedChatModel(
        responses=[
            report_call(["H-01"], call_id="c1"),
            report_call(["H-01"], call_id="c2"),
            done_call("c3"),
        ]
    )
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )
    reports = [
        str(m.content)
        for m in final["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id in {"c1", "c2"}
    ]

    assert len(reports) == 2, "两次一模一样的报告都要真的执行"
    assert not any(DUPLICATE_CALL_MESSAGE.splitlines()[0] in text for text in reports)


# ---------- 工具失败预算 ----------


def parallel_explode_call(call_id: str) -> AIMessage:
    """构造一轮里两条都必炸的调用。

    **每轮都要新建对象**：LangChain 的消息带 id，同一个对象重复塞进状态会被
    `add_messages` 按 id 去重（第一版用例就是复用了同一个对象，于是后两轮的
    消息没进历史）。

    Args:
        call_id: 这一轮的调用 ID 前缀

    Returns:
        AIMessage
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "exploding_tool",
                "args": {"path": "a"},
                "id": f"{call_id}-1",
                "type": "tool_call",
            },
            {
                "name": "exploding_tool",
                "args": {"path": "b"},
                "id": f"{call_id}-2",
                "type": "tool_call",
            },
        ],
        usage_metadata=USAGE_METADATA,
    )


def explode_call(call_id: str = "c-e") -> AIMessage:
    """构造一次必炸的工具调用。

    Args:
        call_id: 调用 ID

    Returns:
        AIMessage
    """
    return ai_call("exploding_tool", {"path": "x"}, call_id)


def test_consecutive_failures_escalate_the_message(project_root: str, exploding: None) -> None:
    """连续失败到阈值后，工具结果里追加一句更硬的提醒。"""
    llm = ScriptedChatModel(
        responses=[
            explode_call("c0"),
            explode_call("c1"),
            explode_call("c2"),
            report_call(["H-01"]),
            done_call(),
        ]
    )
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )
    results = {
        m.tool_call_id: str(m.content)
        for m in final["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id.startswith("c")
    }

    assert "连续失败" not in results["c0"], "第 1 次只是普通报错"
    assert "连续失败 3 次" in results["c2"], "到阈值才追加劝退那句"


def test_total_failures_stop_the_group(project_root: str, exploding: None) -> None:
    """累计失败到上限 → 本组带缺口退出，而不是一路烧到预算见底。"""
    # 5 次失败（默认 tool_fail_exit）之后，第 6 轮开始前就该停
    llm = ScriptedChatModel(responses=[explode_call(f"c{index}") for index in range(5)])
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )

    assert final["stop_reason"] == STOP_TOOL_FAILURES
    assert not final["hunk_status"]["H-01"].comments, "没有产出也不该被算成审过"
    assert final["round_count"] < 20, "是失败预算叫停的，不是轮次上限"


def test_success_resets_the_streak_but_not_the_total(project_root: str, exploding: None) -> None:
    """中途成功一次 → **连续**计数清零，**累计**计数不动。

    两个问题不同：连续问「这个工具是不是坏了」，累计问「这一组烧了多少」。
    """
    llm = ScriptedChatModel(
        responses=[
            explode_call("c0"),
            explode_call("c1"),
            read_call("c2"),  # 成功
            explode_call("c3"),
            explode_call("c4"),
            report_call(["H-01"]),
            done_call(),
        ]
    )
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )
    results = {
        m.tool_call_id: str(m.content)
        for m in final["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id.startswith("c")
    }

    # 累计 4 次（c0/c1/c3/c4），但中间成功过 → 连续最多 2 次，够不到提醒阈值 3
    assert "连续失败" not in results["c4"]
    assert "不再重复执行" not in results["c2"], "这次是成功读到文件"


def test_parallel_failures_each_count(project_root: str, exploding: None) -> None:
    """同一个 super-step 里的并行失败要各算一次。

    计数在同一个实例上、由多个线程改 —— 漏记的后果是闸门晚响（方向安全），
    但用例要求它真的算准。
    """
    llm = ScriptedChatModel(responses=[parallel_explode_call(f"r{index}") for index in range(3)])
    agent = build_review_agent(llm)

    final = agent.invoke(
        initial_state(make_group(["H-01"])),
        context=ReviewContext(project_root=project_root, head_rev=""),
    )

    # 3 轮 × 2 个 = 6 次失败 > 上限 5 → 第 3 轮之后停
    assert final["stop_reason"] == STOP_TOOL_FAILURES
    assert final["round_count"] == 3


# ---------- 关键日志清单 ----------


def test_key_logs_are_present_on_a_normal_group(
    project_root: str, caplog: pytest.LogCaptureFixture
) -> None:
    """一次正常审查里，这几条日志必须都在。

    它们是排查的骨架：缺哪一条，都会让人少一块信息（没有耗时就没法对账时间预算，
    没有用量就没法对账花费）。**清单写在这里，就不靠人眼** —— 改日志很容易顺手
    删掉一条，用例钉住才守得住。
    """
    llm = ScriptedChatModel(responses=[report_call(["H-01"]), done_call()])

    with caplog.at_level(logging.DEBUG):
        run_review_agent(make_group(["H-01"]), llm, project_root)

    text = caplog.text
    # 开场：组名 + 规模
    assert "开始审查 group-001" in text
    # 分组装好的提示词多大（DEBUG）—— 排查「提示词为什么这么长」时第一条就看它
    assert "审查提示词：group-001" in text
    # 退出原因写清是哪一个（不是含糊的「end_done 或连续无产出」）
    assert "退出：end_done 已被受理" in text
    # 收场：原因 + 轮次 + 评论 + 未审 + **耗时** + **用量**
    assert "结束审查 group-001：exit_reason=end_done" in text
    assert "耗时" in text
    assert "用量" in text


def test_budget_stop_is_logged_as_warning(
    project_root: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """预算闸触发时要有 WARNING，并带上用量与两个上限 —— 不然只能靠猜。"""
    tiny_budget(monkeypatch, max_group_cost=1e-9)
    llm = ScriptedChatModel(responses=[report_call(["H-01"])])

    with caplog.at_level("WARNING"):
        run_review_agent(make_group(["H-01", "H-02"]), llm, project_root)

    assert "本组超预算（cost）" in caplog.text
    assert "预算 0.000 元 / 300 秒" in caplog.text


def test_no_progress_reminder_reaches_a_model_that_keeps_calling_tools(
    project_root: str,
) -> None:
    """「一直在调工具、从不报告」到阈值时要真的收到提醒。

    这是**实测踩出来的回归**：提醒原来只写在 `review_pacing` 的「没有工具调用」
    分支里，于是「连续 8 轮读文件、一条不报」这种最该被提醒的形态一次提醒都拿不到，
    第 8 轮被 no_progress 直接砍掉（第一次真跑一条真实 PR 就撞上了）。

    提醒现在挂在 `before_model`（工具跑完、追加人类消息合法的地方）。
    """
    # 每轮读不同范围：避免被去重拦下（那会让「一直在调工具」这个前提失真）
    llm = ScriptedChatModel(
        responses=[
            ai_call("file_read", {"path": "a.go", "start_line": i, "end_line": i + 5}, f"c{i}")
            for i in range(8)
        ]
    )

    result = run_review_agent(make_group(["H-01"]), llm, project_root)

    delivered = [
        str(message.content)
        for call in llm.calls
        for message in call
        if "连续" in str(message.content)
    ]
    assert result.exit_reason == EXIT_NO_PROGRESS, "8 轮无产出仍然按原阈值收尾"
    assert any("你已经连续 5 轮未报告" in text for text in delivered), "第 5 轮的提醒没送到"
    assert any("你已经连续 7 轮未报告" in text for text in delivered), "第 7 轮的提醒没送到"
    # 每轮**只能有一条**：实测撞过「同一轮注入两条」—— 那是我改这块时补丁被应用了两次，
    # 而两条一模一样的提醒白占 token，也让日志读起来像跑了两遍
    sixth = [str(message.content) for message in llm.calls[5]]
    assert sum(1 for text in sixth if "你已经连续 5 轮未报告" in text) == 1


# ---------- 无产出轮次的性质分类 ----------


def test_round_tool_results_only_takes_the_tail() -> None:
    """只取消息尾部连续的工具结果（往前到第一条非工具消息为止）。"""
    history: list[BaseMessage] = [
        ai_call("file_read", {"path": "a.go"}, "c1"),
        ToolMessage(content="旧的", tool_call_id="c1"),
        ai_call("file_read", {"path": "b.go"}, "c2"),
        ToolMessage(content="新的 1", tool_call_id="c2"),
        ToolMessage(content="新的 2", tool_call_id="c2"),
    ]

    results = round_tool_results(history)

    assert [str(message.content) for message in results] == ["新的 2", "新的 1"]


def test_no_progress_kinds_separate_exploration_from_spinning(
    project_root: str, caplog: pytest.LogCaptureFixture
) -> None:
    """八轮无产出：要能说清是「一直在读新内容」还是「在原地打转」。

    实测那条真 PR 的八轮**全在读没读过的文件**（同参数重复调用已被去重拦下）——
    它是在推进，只是没下结论。这与「换个参数读到同样内容」是两回事，砍不砍它
    应该由这个分布来决定，而不是一律按同一个阈值。
    """
    # 每轮**逐行**读：内容必然不同（整段读的话，夹具文件短、会被截断成同样的内容，
    # 那反而变成「重复」—— 第一版用例就是这么把自己坑了的）
    llm = ScriptedChatModel(
        responses=[
            ai_call("file_read", {"path": "a.go", "start_line": i + 1, "end_line": i + 1}, f"c{i}")
            for i in range(8)
        ]
    )

    with caplog.at_level(logging.INFO):
        run_review_agent(make_group(["H-01"]), llm, project_root)

    assert "性质分布" in caplog.text
    assert "新内容" in caplog.text
    # **不再断言「没有重复轮」**：收尾闸从第 6 轮起挡下只读工具，那些拒绝消息
    # 每次一模一样 —— 于是它们被如实记成「重复」。那恰恰是想要的信号：
    # `新内容 6 / 重复 2` 读起来就是「6 轮真探索，2 轮被闸挡下」


# ---------- 收尾闸：不丢结论 ----------


def test_wrap_up_gate_blocks_exploration_but_keeps_the_exit_open(
    project_root: str, caplog: pytest.LogCaptureFixture
) -> None:
    """连续无产出逼近上限时停用探索工具，只放行 report_hunks / end_done。

    **为什么**：实测 4 条真实 PR、7 次运行，6 次是「一路读文件、从不报告」被砍 ——
    已经花掉的 token 与查到的结论一起没。软提醒它不理（发了 4 次都没用），
    所以这里换成硬信号：把路收窄，只能交结论。
    """
    # 7 轮只读探索（其中两轮撞上被挡下的拒绝），第 8 轮报告、第 9 轮结束。
    # 闸从第 5 轮起关闭 —— 留三轮：撞上拒绝、反应过来、报出来
    llm = ScriptedChatModel(
        responses=[
            *[
                ai_call(
                    "file_read", {"path": "a.go", "start_line": i + 1, "end_line": i + 1}, f"c{i}"
                )
                for i in range(7)
            ],
            report_call(["H-01"]),
            done_call(),
        ]
    )

    with caplog.at_level("WARNING"):
        result = run_review_agent(make_group(["H-01"]), llm, project_root)

    assert "停用探索工具" in caplog.text, "到阈值前两轮就该收窄"
    assert result.exit_reason == EXIT_END_DONE, "被挡下之后它仍然能报结论并正常收尾"
    assert result.pending_hunk_ids == [], "H-01 确实报上去了 —— 结论没丢"

    # 被挡下的那些调用没有真的执行：工具结果是那句说明
    blocked = [
        str(message.content)
        for call in llm.calls
        for message in call
        if "探索轮次已用完" in str(message.content)
    ]
    assert blocked, "挡下时要给模型一句能读懂的说明，而不是空结果"


def test_reminder_carries_the_budget(project_root: str) -> None:
    """提醒里要带上「连续几轮就会收尾」—— 不知道截止的工作会无限细化。"""
    llm = ScriptedChatModel(
        responses=[
            *[
                ai_call(
                    "file_read", {"path": "a.go", "start_line": i + 1, "end_line": i + 1}, f"c{i}"
                )
                for i in range(5)
            ],
            report_call(["H-01"]),
            done_call(),
        ]
    )

    run_review_agent(make_group(["H-01"]), llm, project_root)

    delivered = [
        str(message.content)
        for call in llm.calls
        for message in call
        if "agent_reminder" in str(message.content)
    ]
    assert any("本组就会收尾" in text for text in delivered), "提醒要说明预算边界"


def test_explore_reopens_after_a_report(project_root: str) -> None:
    """报告一次之后探索重新开放 —— 本组还有别的 hunk 要查。"""
    llm = ScriptedChatModel(
        responses=[
            *[
                ai_call(
                    "file_read", {"path": "a.go", "start_line": i + 1, "end_line": i + 1}, f"c{i}"
                )
                for i in range(7)
            ],
            report_call(["H-01"]),
            ai_call("file_read", {"path": "a.go", "start_line": 99, "end_line": 99}, "c-after"),
            # H-02 一直没报，第一次 end_done 会被拒（提示剩余 hunk），补一次
            done_call("d1"),
            done_call("d2"),
        ]
    )

    result = run_review_agent(make_group(["H-01", "H-02"]), llm, project_root)

    # H-02 一直没报 → 第二次 end_done 被接受，但带缺口（`_with_gaps`）。这是对的：
    # 收尾闸保证「不丢已有结论」，不承诺「审完所有 hunk」
    assert result.exit_reason == EXIT_END_DONE_WITH_GAPS
    # 报告之后那次 file_read **真的要执行了**：看它那条工具结果（按调用 id 取，
    # 不能翻整个历史 —— 之前被挡下的拒绝消息还在历史里躺着）
    after = [
        str(message.content)
        for call in llm.calls
        for message in call
        if isinstance(message, ToolMessage) and message.tool_call_id == "c-after"
    ]
    assert after, "报告之后的那次调用要有结果"
    assert "探索轮次已用完" not in after[0], "报告之后探索要重新开放"
