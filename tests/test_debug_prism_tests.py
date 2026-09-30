"""单步调试用的全链路演练 —— 在真实的 prism-tests 仓库上跑一次完整审查。

**这个文件是给 IDE 断点用的，不是断言型测试。** 每个阶段的结果都留在具名变量里，
在任意一行下断点就能看清那一步的输入与输出：

    diffs        → list[Diff]            加载+解析的结果
    filtered     → FilterResult          可审查 / 跳过（含原因）
    mapped       → list[HunkWithSymbol]  每个 hunk 的归属符号
    groups       → list[Group]           审查单元
    state        → MainState             第一条 user 消息（指导文本 + diff）
    result       → MainLoopResult        循环产出（退出原因、评论）
    comments     → list[Comment]         整理 + 排序后的评论
    output       → str                   终端渲染结果

LLM 用**脚本化的假实现**（不打真实 API）：可复现、秒级完成、断点不会卡在网络上。
工具（file_read / code_search / report_hunks / end_done）都是**真实执行**的，
所以可以一路 step into 到工具实现里。

要跑真实 LLM 看完整效果，用 CLI：

    cd ~/Projects/python/prism-tests
    prism review --base main --head feature

在 IDE 里直接 Run/Debug 这个文件即可；命令行跑时加 `-s` 才能看到 print：

    uv run pytest tests/test_debug_prism_tests.py -s
"""

import logging
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from prism.core.ast.mapper import format_location, format_symbol, map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.graph.main_loop import initial_state, run_main_loop
from prism.core.graph.state import MainState
from prism.core.grouping.packages import group_by_package
from prism.core.output.renderer import TerminalRenderer
from prism.core.output.sort import sort_comments, to_comments
from prism.models.comment import ReviewStats
from prism.models.grouping import Group

# 被审查的仓库：prism 之外的独立 git 仓库，main 是基线、feature 引入问题
PRISM_TESTS = Path(os.getenv("PRISM_TESTS_DIR", Path.home() / "Projects/python/prism-tests"))
BASE_BRANCH = "main"
HEAD_BRANCH = "feature"

pytestmark = pytest.mark.skipif(
    not (PRISM_TESTS / ".git").is_dir(),
    reason=f"未找到测试仓库 {PRISM_TESTS}，设 PRISM_TESTS_DIR 指定其他路径",
)


class ScriptedLLM:
    """按脚本返回预设响应的假 LLM。

    结构与 LLMClient 一致（只要求有 invoke），因此断点可以顺着
    main_loop.call_llm → 这里 一路走下去，只是不发网络请求。
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        """记录响应脚本。

        Args:
            responses: 依次返回的响应
        """
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], tools: list[BaseTool] | None = None) -> AIMessage:
        """返回下一个脚本响应。

        Args:
            messages: 本轮收到的完整消息历史（含指导文本、工具结果、agent_status）
            tools: 本轮暴露给 LLM 的工具

        Returns:
            下一个预设响应
        """
        self.calls.append(list(messages))
        return self._responses.pop(0)


def report_all(group: Group) -> AIMessage:
    """造一轮「把所有 hunk 一次报完」的 report_hunks 调用。

    每个 hunk 报一条 medium 评论，正文带上它的位置与归属，方便在断点里
    直接看出归属对不对。

    Args:
        group: 当前审查单元

    Returns:
        带单个 tool_call 的 AIMessage
    """
    reports = [
        {
            "hunk_id": item.hunk_id,
            "complete": True,
            "comments": [
                {
                    "severity": "medium",
                    "category": "debug-sample",
                    "content": (
                        f"[调试样例] {format_location(item)} 归属：{format_symbol(item.symbol)}"
                    ),
                    "existing_code": item.part.content.splitlines()[-1][:60],
                }
            ],
        }
        for item in group.hunks
    ]
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "report_hunks", "args": {"reports": reports}, "id": "c1", "type": "tool_call"}
        ],
    )


def end_done() -> AIMessage:
    """造一次 end_done 调用。

    Returns:
        带单个 tool_call 的 AIMessage
    """
    return AIMessage(
        content="",
        tool_calls=[{"name": "end_done", "args": {}, "id": "c2", "type": "tool_call"}],
    )


def dump_state(state: MainState) -> None:
    """打印组装好的第一条消息（system + user），便于核对送进 LLM 的内容。

    Args:
        state: initial_state 的产物
    """
    for message in state["messages"]:
        print(f"\n{'=' * 30} {message.type} {'=' * 30}")
        print(str(message.content)[:1500])


def enable_console_logging(level: int = logging.INFO) -> None:
    """让 prism 的日志与 print 实时打到控制台，不被 pytest 吞掉。

    pytest 会在 root logger 上挂自己的捕获 handler —— 此时
    `logging.basicConfig()` 会变成空操作（它只在 root 没有任何 handler 时才生效），
    所以必须 `force=True` 换掉它。加完之后不必再加 `--log-cli-level`。

    只影响 prism 命名空间与 root 的输出，不改任何业务逻辑。

    Args:
        level: 日志级别，默认 INFO
    """
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("prism").setLevel(level)


def test_debug_full_pipeline() -> None:
    """在 prism-tests 上跑一次完整链路，供断点单步查看。"""
    enable_console_logging()
    print(f"\n被审查仓库: {PRISM_TESTS}（{BASE_BRANCH} → {HEAD_BRANCH}）")

    # ── 阶段 1：加载 + 解析 diff ──────────────────────────────────────
    # 断在这里看 diffs：每个 Diff 含 path / 文件级标记 / hunks
    diffs = load_diffs(PRISM_TESTS, BASE_BRANCH, HEAD_BRANCH)
    print(f"\n[1] 加载 diff: {len(diffs)} 个文件")
    for diff in diffs:
        print(f"    {diff.path}  {len(diff.hunks)} hunk  (+{diff.insertions}/-{diff.deletions})")
    assert diffs, "prism-tests 的 main → feature 应有改动"

    # ── 阶段 2：文件过滤（三级：硬性排除 → 扩展名 → 降噪）────────────
    # 断在这里看 filtered.skipped 的每个原因
    filtered = filter_files(diffs)
    print(f"\n[2] 过滤: 可审查 {len(filtered.reviewable)}，跳过 {len(filtered.skipped)}")
    for skipped in filtered.skipped:
        print(f"    skip {skipped.path} ({skipped.reason})")

    # ── 阶段 3：hunk → 符号（-U0 改动行 ∩ tree-sitter 符号）──────────
    # 断在这里逐个看 mapped[i].symbol —— 归属对不对就在这
    mapped = map_hunks_to_symbols(PRISM_TESTS, BASE_BRANCH, HEAD_BRANCH, filtered.reviewable)
    print(f"\n[3] 归属: {len(mapped)} 个 hunk")
    for hunk in mapped:
        print(f"    {format_location(hunk)}  →  {format_symbol(hunk.symbol)}")

    # ── 阶段 4：按包分组 ─────────────────────────────────────────────
    # 断在这里看 groups —— prism-tests 按包分了 main / edge / multi 三组
    groups = group_by_package(mapped)
    print(f"\n[4] 分组: {len(groups)} 个组")
    for index, group in enumerate(groups, start=1):
        files_n, hunks_n = len(group.files), len(group.hunks)
        print(f"    Group {index}: {group.package_path} ({files_n} 文件, {hunks_n} hunk)")

    # ── 阶段 5：组装状态（system prompt + 指导文本 + diff）───────────
    # dump_state 里能看到完整的送审内容：审查范围 / 审查要求 / 报告节奏 / 结束审查
    # 想调其他组就改这个下标
    first_group = groups[0]
    state = initial_state(first_group, str(PRISM_TESTS), HEAD_BRANCH)
    ctx = state["tool_context"]
    print(
        f"\n[5] 状态组装: {len(state['messages'])} 条初始消息，{len(ctx.hunk_status)} 个待审 hunk"
    )
    dump_state(state)
    assert state["tool_context"].pending_ids() == [h.hunk_id for h in first_group.hunks]

    # ── 阶段 6：Main 循环（工具真实执行，只有 LLM 是假的）────────────
    # 断在这里可以 step into：call_llm → execute_tools → 各工具实现 → check_exit
    llm = ScriptedLLM([report_all(first_group), end_done()])
    result = run_main_loop(first_group, llm, str(PRISM_TESTS), HEAD_BRANCH)
    print(f"\n[6] 循环结束: exit_reason={result.exit_reason} 轮次={result.round_count}")
    print(f"    评论 {len(result.comments)} 条，未审 {result.pending_hunk_ids}")
    print(f"    第二轮 LLM 收到的消息数: {len(llm.calls[1])}（含工具结果与 agent_status）")
    assert result.exit_reason == "end_done"
    assert result.pending_hunk_ids == []

    # ── 阶段 7：整理 + 排序 ──────────────────────────────────────────
    # 断在这里看排序前后的顺序差异
    comments = sort_comments(to_comments([result], groups))
    print(f"\n[7] 整理排序: {len(comments)} 条评论")
    for comment in comments:
        print(f"    [{comment.severity:<7}] {comment.path:<20} {comment.content[:56]}")

    # ── 阶段 8：终端渲染 ─────────────────────────────────────────────
    stats = ReviewStats(
        base_branch=BASE_BRANCH,
        head_sha=HEAD_BRANCH,
        files_count=len(filtered.reviewable),
        comments_count=len(comments),
        duration_seconds=0.0,
    )
    output = TerminalRenderer().render(comments, stats)
    print(f"\n[8] 渲染输出（{len(output)} 字符）:\n")
    print(output)

    assert "Prism Code Review" in output
    assert "=== 统计 ===" in output
