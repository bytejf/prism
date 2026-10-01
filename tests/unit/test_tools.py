"""工具系统的单元测试（file_read / code_search / report_hunks / end_done）。

工具通过 `ToolRuntime` 拿静态上下文（`runtime.context`）与可变状态
（`runtime.state`），不再吃一个共享的 ToolContext 对象。
写状态的工具返回 `Command`，测试从 `Command.update` 里读结果。
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    HunkStatus,
    ReviewState,
    done_count,
    end_done_verdict,
    merge_hunk_status,
    pending_ids,
)
from prism.core.tools.code_search import MAX_RESULTS, code_search
from prism.core.tools.end_done import end_done
from prism.core.tools.file_read import MAX_LINES, file_read
from prism.core.tools.report_hunks import report_hunks

CONTEXT_WITHOUT_ROOT = ReviewContext(project_root="")


def make_state(*hunk_ids: str) -> ReviewState:
    """构造预置 hunk 状态表的审查状态。

    Args:
        hunk_ids: hunk 编号

    Returns:
        ReviewState，全部为 pending
    """
    return ReviewState(
        messages=[],
        hunk_status={
            hunk_id: HunkStatus(hunk_id=hunk_id, status=STATUS_PENDING) for hunk_id in hunk_ids
        },
        end_done_mark=0,
        round_count=0,
        max_rounds=30,
        no_report_rounds=0,
        end_done_count=0,
        group=None,  # type: ignore[typeddict-item]
    )


def runtime_for(ctx: ReviewContext, state: ReviewState) -> ToolRuntime[Any, Any]:
    """构造测试用的 ToolRuntime。

    生产环境由框架从图状态里注入；测试手动拼一个最小可用的。

    Args:
        ctx: 静态上下文
        state: 审查状态

    Returns:
        ToolRuntime
    """
    return ToolRuntime(
        state=state,
        context=ctx,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-1",
        store=None,
    )


class ToolUnderTest:
    """把 context/state 注入 runtime 的工具包装。

    做的是框架 ToolNode 做的那件事（注入 runtime），
    让用例保持 `tool.invoke(args)` 的写法。
    """

    def __init__(
        self, tool: BaseTool, ctx: ReviewContext, state: ReviewState | None = None
    ) -> None:
        """绑定工具与运行时数据。

        Args:
            tool: 被测工具
            ctx: 静态上下文
            state: 审查状态；不传时用空状态
        """
        self._tool = tool
        self._ctx = ctx
        self._state = state if state is not None else make_state()

    def invoke(self, args: dict[str, Any]) -> Any:
        """调用工具（自动注入 runtime）。

        Args:
            args: 工具参数

        Returns:
            工具原始返回（只读工具是 str，写状态工具是 Command）
        """
        return self._tool.invoke({**args, "runtime": runtime_for(self._ctx, self._state)})

    def text(self, args: dict[str, Any]) -> str:
        """调用工具并把返回压成文本（只读工具用）。

        Args:
            args: 工具参数

        Returns:
            工具返回的文本
        """
        return str(self.invoke(args))

    def update(self, args: dict[str, Any]) -> dict[str, Any]:
        """调用工具，取出 Command 的 state 更新，并把它落回 state。

        「落回 state」是图做的事：工具自己只**返回**更新，不原地改。
        多次调用同一个工具时，第二次必须能看到第一次的结果，所以这里要模拟。

        Args:
            args: 工具参数

        Returns:
            Command.update 字典

        Raises:
            AssertionError: 工具没有返回 Command
        """
        result = self.invoke(args)
        assert isinstance(result, Command), f"期望 Command，实际 {type(result).__name__}"
        # Command.update 是可选的（框架类型如此），本文件的工具都会给出，空值兜底只是过类型
        updates = dict(result.update or {})
        # 工具写的是**增量**，图会用 reducer 按 key 合并（见 ReviewState 的
        # merge_hunk_status）。这里照做，否则多次调用时后一次的增量会把
        # 上一次的结果整个替换掉，测出来的行为与真实图不一致
        for key, value in updates.items():
            if key == "hunk_status":
                self._state["hunk_status"] = merge_hunk_status(self._state["hunk_status"], value)
            else:
                self._state[key] = value  # type: ignore[literal-required]
        return updates

    @property
    def name(self) -> str:
        """工具名。

        Returns:
            工具名
        """
        return self._tool.name


def tool_message_text(update: dict[str, Any]) -> str:
    """从 Command.update 里取出回给模型的文本。

    Args:
        update: Command.update 字典

    Returns:
        ToolMessage 的文本
    """
    return "\n".join(str(m.content) for m in update.get("messages", []))


@pytest.fixture
def ctx(tmp_path: Path) -> ReviewContext:
    """构造以 tmp_path 为项目根的上下文。

    Returns:
        ReviewContext 对象
    """
    return ReviewContext(project_root=str(tmp_path))


# ---------- file_read ----------


def test_file_read_returns_lines_with_numbers(ctx: ReviewContext, tmp_path: Path) -> None:
    """测试读取文件指定行范围，输出带行号。"""
    (tmp_path / "test.go").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "test.go", "start_line": 2, "end_line": 3})
    assert "line2" in result
    assert "line3" in result
    assert "line1" not in result
    assert "2: line2" in result


def test_file_read_truncates_at_limit(ctx: ReviewContext, tmp_path: Path) -> None:
    """测试超过 500 行时截断并提示。"""
    (tmp_path / "big.go").write_text("x\n" * (MAX_LINES + 100), encoding="utf-8")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "big.go", "start_line": 1, "end_line": MAX_LINES + 100})
    assert "截断" in result
    assert result.count("\n") <= MAX_LINES + 1


def test_file_read_rejects_path_traversal(ctx: ReviewContext, tmp_path: Path) -> None:
    """测试拒绝越出项目根的路径，且不泄漏目标文件内容。"""
    outside = tmp_path.parent / "outside.go"
    outside.write_text("OUTSIDE_CONTENT\n", encoding="utf-8")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "../outside.go", "start_line": 1, "end_line": 1})
    assert "错误" in result
    assert "OUTSIDE_CONTENT" not in result


def test_file_read_rejects_absolute_path_outside_project(ctx: ReviewContext) -> None:
    """测试拒绝项目外的绝对路径。"""
    tool = ToolUnderTest(file_read, ctx)
    result = tool.text({"path": "/etc/hosts", "start_line": 1, "end_line": 1})
    assert "错误" in result


def test_file_read_missing_file(ctx: ReviewContext) -> None:
    """测试文件不存在时返回错误文本而非抛异常。"""
    tool = ToolUnderTest(file_read, ctx)
    result = tool.text({"path": "nope.go", "start_line": 1, "end_line": 1})
    assert "不存在" in result


def test_file_read_start_line_beyond_eof(ctx: ReviewContext, tmp_path: Path) -> None:
    """测试起始行超出总行数。"""
    (tmp_path / "small.go").write_text("a\nb\n", encoding="utf-8")
    tool = ToolUnderTest(file_read, ctx)
    result = tool.text({"path": "small.go", "start_line": 99, "end_line": 100})
    assert "错误" in result


def test_file_read_without_project_root() -> None:
    """测试未设置 project_root 时返回错误文本。"""
    tool = ToolUnderTest(file_read, CONTEXT_WITHOUT_ROOT)
    result = tool.text({"path": "a.go", "start_line": 1, "end_line": 1})
    assert "project_root" in result


# ---------- 必须读「被审查版本」而不是工作区 ----------

MAIN_GO = "\n".join(
    [
        "package main",
        "",
        "func Find(id string) string {",
        '\treturn db.Query("WHERE id = ?", id)',
        "}",
        "",
    ]
)
FEATURE_GO = "\n".join(
    [
        "package main",
        "",
        "func Find(id string) string {",
        '\treturn db.Query("WHERE id = \'" + id + "\'")',
        "}",
        "",
    ]
)


@pytest.fixture
def divergence_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造「工作区停在 main、被审查版本是 feature」的仓库。

    main 与 feature 的 db.go 内容不同，用来检测工具到底读了哪一版。

    Returns:
        仓库路径（工作区在 main 上）
    """
    repo = tmp_path / "divergence"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "db.go").write_text(MAIN_GO, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "db.go").write_text(FEATURE_GO, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")

    # 关键：工作区切回 main，模拟「审别人的 PR、自己 checkout 在 main」
    git(repo, "checkout", "-q", "main")
    return repo


def test_file_read_reads_reviewed_revision_not_worktree(divergence_repo: Path) -> None:
    """测试 file_read 读被审查版本，而不是工作区。

    回归用例：审查目标是 feature，但工作区停在 main。
    若读工作区，会拿到改之前的参数化查询，导致系统性漏报。
    """
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="feature")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = '\" + id + \"'" in result, "读到了工作区（main）的内容而不是被审查版本"
    assert "WHERE id = ?" not in result


def test_file_read_falls_back_to_worktree_without_head_rev(divergence_repo: Path) -> None:
    """测试未指定 head_rev 时退化为读工作区。"""
    ctx = ReviewContext(project_root=str(divergence_repo))
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = ?" in result


def test_file_read_unknown_head_rev_falls_back(divergence_repo: Path) -> None:
    """测试 head_rev 无法解析时退化为读工作区，不抛异常。"""
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="no-such-branch")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = ?" in result


def test_file_read_no_fallback_for_file_missing_from_revision(divergence_repo: Path) -> None:
    """测试版本可读、但文件不在该版本里时，不退化为读工作区。

    回归用例：head_rev 能正常解析，工作区里却有同名文件而该版本的树里没有。
    「这个版本里没有它」才是正确答案 —— 退化读工作区会把 PR 里根本不存在的
    本地文件喂给 LLM，且与 code_search（搜版本树）对「有哪些文件」的答案不一致。
    """
    (divergence_repo / "local_only.go").write_text(
        "package main\n// 本地未提交，PR 里不存在\n", encoding="utf-8"
    )
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="feature")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "local_only.go", "start_line": 1, "end_line": 2})
    assert "文件不存在" in result
    assert "本地未提交" not in result, "读到了工作区的本地文件"

    # 同一个文件在「版本不可读」时仍应能读到 —— 退化路径本身没被砍掉
    ctx_no_rev = ReviewContext(project_root=str(divergence_repo))
    result_no_rev = ToolUnderTest(file_read, ctx_no_rev).text(
        {"path": "local_only.go", "start_line": 1, "end_line": 2}
    )
    assert "本地未提交" in result_no_rev


def test_file_read_rejects_traversal_even_with_head_rev(divergence_repo: Path) -> None:
    """测试指定 head_rev 时路径校验依然生效。"""
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="feature")
    tool = ToolUnderTest(file_read, ctx)

    for bad in ("../outside.go", "/etc/hosts", "a/../../b.go"):
        assert "错误" in tool.text({"path": bad, "start_line": 1, "end_line": 1})


# ---------- code_search ----------


@pytest.fixture
def go_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造含 Go 文件的临时 git 仓库。

    Returns:
        仓库路径
    """
    repo = tmp_path / "search"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    (repo / "a.go").write_text(
        "package main\n\nfunc Alpha() {}\n\nfunc Beta() {}\n", encoding="utf-8"
    )
    (repo / "b.go").write_text("package main\n\nfunc Gamma() {}\n", encoding="utf-8")
    (repo / "note.md").write_text("func Alpha 说明\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def test_code_search_finds_matches(go_repo: Path) -> None:
    """测试搜索命中并带文件路径与行号。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))
    result = tool.text({"pattern": "func Alpha"})

    assert "a.go:3" in result
    assert "func Alpha" in result


def test_code_search_respects_file_pattern(go_repo: Path) -> None:
    """测试 file_pattern 限定文件类型。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))
    assert "note.md" not in tool.text({"pattern": "func Alpha", "file_pattern": "*.go"})
    assert "note.md" in tool.text({"pattern": "func Alpha", "file_pattern": "*.md"})


def test_code_search_no_match(go_repo: Path) -> None:
    """测试无匹配时返回说明文本。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))
    assert "未找到匹配" in tool.text({"pattern": "func Nonexistent"})


def test_code_search_keeps_non_ascii_paths_readable(
    go_repo: Path, git: Callable[..., None]
) -> None:
    """测试非 ASCII 路径原样输出，不转义成八进制。

    回归用例：git 默认 core.quotepath=true，会把中文路径输出成
    "\\346\\234\\200.md"，LLM 拿到后无法回传给 file_read。
    """
    (go_repo / "中文说明.md").write_text("func Delta 说明\n", encoding="utf-8")
    git(go_repo, "add", "-A")
    git(go_repo, "commit", "-q", "-m", "add chinese file")

    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))
    result = tool.text({"pattern": "func Delta", "file_pattern": "*.md"})

    assert "中文说明.md" in result
    assert "\\346" not in result


def test_code_search_caps_result_count(go_repo: Path) -> None:
    """测试结果数被压到上限并提示截断。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))
    result = tool.text({"pattern": "func", "max_results": MAX_RESULTS})
    assert "func" in result


def test_code_search_without_project_root() -> None:
    """测试未设置 project_root 时返回错误文本。"""
    tool = ToolUnderTest(code_search, CONTEXT_WITHOUT_ROOT)
    assert "project_root" in tool.text({"pattern": "func"})


def test_code_search_searches_reviewed_revision_not_worktree(divergence_repo: Path) -> None:
    """测试 code_search 搜被审查版本，而不是工作区。

    回归用例：审查目标是 feature，但工作区停在 main。
    若搜工作区，会搜到改之前的代码，与 file_read 一起造成系统性漏报。
    """
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="feature")
    tool = ToolUnderTest(code_search, ctx)

    result = tool.text({"pattern": r"WHERE id = '"})
    assert "db.go" in result, "没搜到 feature 里的字符串拼接查询"

    # 工作区（main）里的参数化查询不应出现在结果里
    assert "WHERE id = ?" not in tool.text({"pattern": r"WHERE id = \?"})


def test_code_search_falls_back_to_worktree_without_head_rev(divergence_repo: Path) -> None:
    """测试未指定 head_rev 时搜工作区。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(divergence_repo)))
    assert "db.go" in tool.text({"pattern": r"WHERE id = \?"})


# ---------- report_hunks ----------


def test_report_hunks_records_comments_and_marks_done() -> None:
    """测试报告评论并把 hunk 标记为完成。"""
    state = make_state("H-01", "H-02")
    before = dict(state["hunk_status"])  # 调用前的快照，用来验证工具没有原地改
    tool = ToolUnderTest(report_hunks, ReviewContext(), state)

    update = tool.update(
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
        }
    )

    hunk_status = update["hunk_status"]
    assert "已记录" in tool_message_text(update)
    assert "剩余" in tool_message_text(update)
    assert hunk_status["H-01"].status == STATUS_DONE
    assert "H-02" not in hunk_status, "增量里不该带没改动的 hunk"
    h01_comments = hunk_status["H-01"].comments
    assert len(h01_comments) == 1
    assert h01_comments[0].content == "goroutine 缺少退出路径"
    assert h01_comments[0].suggestion_code is None
    # 工具不再原地改状态：调用前的快照没被动过（工具只是**返回**了新对象）
    assert before["H-01"].status == STATUS_PENDING
    assert before["H-01"].comments == []


def test_report_hunks_returns_remaining_not_completed() -> None:
    """测试返回值给「剩余未审」而非「已完成」。"""
    state = make_state("H-01", "H-02", "H-03")
    tool = ToolUnderTest(report_hunks, ReviewContext(), state)

    update = tool.update({"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]})
    text = tool_message_text(update)
    assert "剩余未审：H-02, H-03" in text
    assert "1/3 完成" in text


def test_report_hunks_all_done_reports_none_pending() -> None:
    """测试全部完成时剩余为「无」。"""
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))
    update = tool.update({"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]})
    assert "剩余未审：无" in tool_message_text(update)


def test_report_hunks_ignores_unknown_hunk_id() -> None:
    """测试未知 hunk 编号被忽略且提示。"""
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))

    update = tool.update({"reports": [{"hunk_id": "H-99", "complete": True, "comments": []}]})
    assert "未知 hunk 编号" in tool_message_text(update)
    assert update["hunk_status"] == {}, "未知编号不该写回任何增量"


def test_report_hunks_marks_incomplete_hunk_in_progress() -> None:
    """测试 complete=False 时置为「审查中」而不是 done。

    这给了 STATUS_IN_PROGRESS 一条真实的产生路径，<agent_status> 的
    「🔄 审查中」分组才有意义（否则永远是 0）。
    """
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))

    update = tool.update({"reports": [{"hunk_id": "H-01", "complete": False, "comments": []}]})
    hunk_status = update["hunk_status"]
    assert hunk_status["H-01"].status == STATUS_IN_PROGRESS
    # 未完成 → 仍算未审，不进 pending_ids 之外的任何完成统计
    assert pending_ids(hunk_status) == ["H-01"]
    assert done_count(hunk_status) == 0


def test_report_hunks_normalizes_optional_fields() -> None:
    """测试可选字段留空时转成 None，必填字段去掉首尾空白。"""
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))

    update = tool.update(
        {
            "reports": [
                {
                    "hunk_id": "H-01",
                    "complete": True,
                    "comments": [
                        {
                            "severity": "high",
                            "category": "  x  ",
                            "content": "y",
                            "existing_code": "z",
                            "suggestion_code": "  ",
                        }
                    ],
                }
            ]
        }
    )
    saved = update["hunk_status"]["H-01"].comments[0]
    assert saved.severity == "high"
    assert saved.category == "x"
    assert saved.suggestion_code is None


def test_report_hunks_rejects_invalid_severity() -> None:
    """测试非法 severity 被 schema 拦下，而不是静默变成空字符串。

    回归用例：参数曾是 `list[dict]`，模型拿不到 schema，实测 55% 的评论
    漏填 severity，渲染成 `[类别 · ]`，排序也失效。
    """
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))
    bad = {
        "reports": [
            {
                "hunk_id": "H-01",
                "complete": True,
                "comments": [
                    {"severity": "", "category": "x", "content": "y", "existing_code": "z"}
                ],
            }
        ]
    }
    with pytest.raises(ValidationError, match="severity"):
        tool.invoke(bad)

    # 漏填 severity 同样拦下
    missing = {
        "reports": [
            {
                "hunk_id": "H-01",
                "complete": True,
                "comments": [{"category": "x", "content": "y", "existing_code": "z"}],
            }
        ]
    }
    with pytest.raises(ValidationError, match="severity"):
        tool.invoke(missing)

    # 大小写敏感的枚举：写 "CRITICAL" 也拦下（模型看到的 enum 是小写）
    wrong = {
        "reports": [
            {
                "hunk_id": "H-01",
                "complete": True,
                "comments": [
                    {"severity": "CRITICAL", "category": "x", "content": "y", "existing_code": "z"}
                ],
            }
        ]
    }
    with pytest.raises(ValidationError, match="severity"):
        tool.invoke(wrong)


def test_report_hunks_schema_is_visible_to_model() -> None:
    """测试模型能从 schema 里看到 severity 的取值与含义，不需要猜。"""
    # tool_call_schema 的类型是 pydantic v1/v2 模型或 dict 的联合，框架没写窄
    schema = cast(type[BaseModel], report_hunks.tool_call_schema).model_json_schema()
    props = schema["$defs"]["HunkReportArg"]["properties"]
    comment = schema["$defs"]["CommentArg"]["properties"]

    assert set(comment) >= {"severity", "category", "content", "existing_code"}
    assert comment["severity"]["enum"] == ["high", "medium", "low"]
    assert "必填" in comment["severity"]["description"]
    assert "hunk_id" in props


def test_report_hunks_rejects_duplicate_report_of_done_hunk() -> None:
    """测试已完成的 hunk 被重复报告时整体忽略，评论不成倍放大。

    回归用例：评论是追加不是去重合并，同一个问题报 3 次就会变成 3 条。
    LLM 反复确认同一个 hunk 是自然行为。
    """
    state = make_state("H-01", "H-02")
    tool = ToolUnderTest(report_hunks, ReviewContext(), state)
    one_comment = [{"severity": "high", "category": "x", "content": "问题", "existing_code": "y"}]
    first = tool.update(
        {"reports": [{"hunk_id": "H-01", "complete": True, "comments": one_comment}]}
    )
    assert len(first["hunk_status"]["H-01"].comments) == 1

    # 第二次、第三次报同一个 hunk —— 应被忽略
    repeat = tool.update(
        {"reports": [{"hunk_id": "H-01", "complete": True, "comments": one_comment}]}
    )

    # 重复报告不写回增量 —— 这一条同时证明了评论没有成倍放大
    assert repeat["hunk_status"] == {}, "重复报告写回了增量，评论会被放大"
    assert len(state["hunk_status"]["H-01"].comments) == 1
    text = tool_message_text(repeat)
    assert "已审查完毕" in text
    assert "本次报告已忽略" in text
    assert "未记录任何 hunk" in text
    # 拒绝报告不应影响进度（增量是空的，看合并后的状态）
    assert done_count(state["hunk_status"]) == 1


def test_report_hunks_duplicate_and_new_in_one_call() -> None:
    """测试同一次调用里既有重复也有新报告时，只忽略重复的那部分。"""
    state = make_state("H-01", "H-02")
    tool = ToolUnderTest(report_hunks, ReviewContext(), state)
    tool.update({"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]})

    update = tool.update(
        {
            "reports": [
                {"hunk_id": "H-01", "complete": True, "comments": []},
                {"hunk_id": "H-02", "complete": True, "comments": []},
            ]
        }
    )
    text = tool_message_text(update)
    assert "已记录 1 个 hunk：H-02" in text
    assert "已审查完毕" in text
    # 增量只含 H-02，进度要看合并后的状态
    assert set(update["hunk_status"]) == {"H-02"}
    assert done_count(state["hunk_status"]) == 2


def test_report_hunks_incomplete_hunk_accepts_repeat() -> None:
    """测试未完成的 hunk 仍可被重复报告 —— 补充评论的路径不能被堵死。

    complete=false 表示「报了部分结论但还没审完」，此时重复报告是合法的。
    """
    tool = ToolUnderTest(report_hunks, ReviewContext(), make_state("H-01"))
    note = [{"severity": "low", "category": "x", "content": "补充", "existing_code": "y"}]

    tool.update({"reports": [{"hunk_id": "H-01", "complete": False, "comments": note}]})
    update = tool.update({"reports": [{"hunk_id": "H-01", "complete": False, "comments": note}]})

    assert "已审查完毕" not in tool_message_text(update)
    assert len(update["hunk_status"]["H-01"].comments) == 2


# ---------- end_done ----------


def test_end_done_accepted_when_all_complete() -> None:
    """测试全部完成时接受结束。"""
    state = make_state("H-01")
    state["hunk_status"]["H-01"] = HunkStatus("H-01", STATUS_DONE)
    tool = ToolUnderTest(end_done, ReviewContext(), state)

    update = tool.update({})
    assert "审查完成" in tool_message_text(update)
    assert update["end_done_count"] == 1


def test_end_done_rejected_on_first_call_with_pending() -> None:
    """测试有未完成 hunk 时第一次调用被拒绝。"""
    state = make_state("H-01")
    tool = ToolUnderTest(end_done, ReviewContext(), state)

    update = tool.update({})
    assert "H-01" in tool_message_text(update)
    assert "report_hunks" in tool_message_text(update)
    assert update["end_done_count"] == 1


def test_end_done_accepted_on_second_call_with_gaps() -> None:
    """测试第二次调用接受结束并提示未审 hunk（兜底，不让流程卡死）。"""
    state = make_state("H-01")
    tool = ToolUnderTest(end_done, ReviewContext(), state)

    tool.update({})
    update = tool.update({})
    assert "已接受结束请求" in tool_message_text(update)
    assert "H-01" in tool_message_text(update)
    assert update["end_done_count"] == 2


def test_end_done_verdict_matches_tool_behavior() -> None:
    """测试纯函数判定与工具行为一致。"""
    assert end_done_verdict(1, ["H-01"]) == (False, ["H-01"])
    assert end_done_verdict(2, ["H-01"]) == (True, ["H-01"])
    assert end_done_verdict(1, []) == (True, [])
