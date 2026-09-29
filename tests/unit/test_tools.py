"""工具系统的单元测试（file_read / code_search / report_hunks / end_done）。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from prism.core.tools.code_search import MAX_RESULTS, make_code_search_tool
from prism.core.tools.context import (
    STATUS_DONE,
    STATUS_PENDING,
    HunkStatus,
    ToolContext,
)
from prism.core.tools.end_done import end_done_verdict, make_end_done_tool
from prism.core.tools.file_read import MAX_LINES, make_file_read_tool
from prism.core.tools.report_hunks import make_report_hunks_tool

CONTEXT_WITHOUT_ROOT = ToolContext()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """构造以 tmp_path 为项目根的上下文。

    Returns:
        ToolContext 对象
    """
    return ToolContext(project_root=str(tmp_path))


def make_ctx_with_hunks(*hunk_ids: str) -> ToolContext:
    """构造预置 hunk 状态表的上下文。

    Args:
        hunk_ids: hunk 编号

    Returns:
        ToolContext 对象，全部为 pending
    """
    ctx = ToolContext()
    ctx.init_hunks(list(hunk_ids))
    return ctx


# ---------- file_read ----------


def test_file_read_returns_lines_with_numbers(ctx: ToolContext, tmp_path: Path) -> None:
    """测试读取文件指定行范围，输出带行号。"""
    (tmp_path / "test.go").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "test.go", "start_line": 2, "end_line": 3})
    assert "line2" in result
    assert "line3" in result
    assert "line1" not in result
    assert "2: line2" in result


def test_file_read_truncates_at_limit(ctx: ToolContext, tmp_path: Path) -> None:
    """测试超过 500 行时截断并提示。"""
    (tmp_path / "big.go").write_text("x\n" * (MAX_LINES + 100), encoding="utf-8")
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "big.go", "start_line": 1, "end_line": MAX_LINES + 100})
    assert "截断" in result
    assert result.count("\n") <= MAX_LINES + 1


def test_file_read_rejects_path_traversal(ctx: ToolContext, tmp_path: Path) -> None:
    """测试拒绝越出项目根的路径，且不泄漏目标文件内容。"""
    outside = tmp_path.parent / "outside.go"
    outside.write_text("OUTSIDE_CONTENT\n", encoding="utf-8")
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "../outside.go", "start_line": 1, "end_line": 1})
    assert "错误" in result
    assert "OUTSIDE_CONTENT" not in result


def test_file_read_rejects_absolute_path_outside_project(ctx: ToolContext) -> None:
    """测试拒绝项目外的绝对路径。"""
    tool = make_file_read_tool(ctx)
    result = tool.invoke({"path": "/etc/hosts", "start_line": 1, "end_line": 1})
    assert "错误" in result


def test_file_read_missing_file(ctx: ToolContext) -> None:
    """测试文件不存在时返回错误文本而非抛异常。"""
    tool = make_file_read_tool(ctx)
    result = tool.invoke({"path": "nope.go", "start_line": 1, "end_line": 1})
    assert "不存在" in result


def test_file_read_start_line_beyond_eof(ctx: ToolContext, tmp_path: Path) -> None:
    """测试起始行超出总行数。"""
    (tmp_path / "small.go").write_text("a\nb\n", encoding="utf-8")
    tool = make_file_read_tool(ctx)
    result = tool.invoke({"path": "small.go", "start_line": 99, "end_line": 100})
    assert "错误" in result


def test_file_read_without_project_root() -> None:
    """测试未设置 project_root 时返回错误文本。"""
    tool = make_file_read_tool(CONTEXT_WITHOUT_ROOT)
    result = tool.invoke({"path": "a.go", "start_line": 1, "end_line": 1})
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
    ctx = ToolContext(project_root=str(divergence_repo), head_rev="feature")
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = '\" + id + \"'" in result, "读到了工作区（main）的内容而不是被审查版本"
    assert "WHERE id = ?" not in result


def test_file_read_falls_back_to_worktree_without_head_rev(divergence_repo: Path) -> None:
    """测试未指定 head_rev 时退化为读工作区。"""
    ctx = ToolContext(project_root=str(divergence_repo))
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = ?" in result


def test_file_read_unknown_head_rev_falls_back(divergence_repo: Path) -> None:
    """测试 head_rev 无法解析时退化为读工作区，不抛异常。"""
    ctx = ToolContext(project_root=str(divergence_repo), head_rev="no-such-branch")
    tool = make_file_read_tool(ctx)

    result = tool.invoke({"path": "db.go", "start_line": 3, "end_line": 4})
    assert "WHERE id = ?" in result


def test_file_read_rejects_traversal_even_with_head_rev(divergence_repo: Path) -> None:
    """测试指定 head_rev 时路径校验依然生效。"""
    ctx = ToolContext(project_root=str(divergence_repo), head_rev="feature")
    tool = make_file_read_tool(ctx)

    for bad in ("../outside.go", "/etc/hosts", "a/../../b.go"):
        assert "错误" in tool.invoke({"path": bad, "start_line": 1, "end_line": 1})


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
    tool = make_code_search_tool(ToolContext(project_root=str(go_repo)))
    result = tool.invoke({"pattern": "func Alpha"})

    assert "a.go:3" in result
    assert "func Alpha" in result


def test_code_search_respects_file_pattern(go_repo: Path) -> None:
    """测试 file_pattern 限定文件类型。"""
    tool = make_code_search_tool(ToolContext(project_root=str(go_repo)))
    assert "note.md" not in tool.invoke({"pattern": "func Alpha", "file_pattern": "*.go"})
    assert "note.md" in tool.invoke({"pattern": "func Alpha", "file_pattern": "*.md"})


def test_code_search_no_match(go_repo: Path) -> None:
    """测试无匹配时返回说明文本。"""
    tool = make_code_search_tool(ToolContext(project_root=str(go_repo)))
    assert "未找到匹配" in tool.invoke({"pattern": "func Nonexistent"})


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

    tool = make_code_search_tool(ToolContext(project_root=str(go_repo)))
    result = tool.invoke({"pattern": "func Delta", "file_pattern": "*.md"})

    assert "中文说明.md" in result
    assert "\\346" not in result


def test_code_search_caps_result_count(go_repo: Path) -> None:
    """测试结果数被压到上限并提示截断。"""
    tool = make_code_search_tool(ToolContext(project_root=str(go_repo)))
    result = tool.invoke({"pattern": "func", "max_results": MAX_RESULTS})
    assert "func" in result


def test_code_search_without_project_root() -> None:
    """测试未设置 project_root 时返回错误文本。"""
    tool = make_code_search_tool(CONTEXT_WITHOUT_ROOT)
    assert "project_root" in tool.invoke({"pattern": "func"})


def test_code_search_searches_reviewed_revision_not_worktree(divergence_repo: Path) -> None:
    """测试 code_search 搜被审查版本，而不是工作区。

    回归用例：审查目标是 feature，但工作区停在 main。
    若搜工作区，会搜到改之前的代码，与 file_read 一起造成系统性漏报。
    """
    ctx = ToolContext(project_root=str(divergence_repo), head_rev="feature")
    tool = make_code_search_tool(ctx)

    result = tool.invoke({"pattern": r"WHERE id = '"})
    assert "db.go" in result, "没搜到 feature 里的字符串拼接查询"

    # 工作区（main）里的参数化查询不应出现在结果里
    assert "WHERE id = ?" not in tool.invoke({"pattern": r"WHERE id = \?"})


def test_code_search_falls_back_to_worktree_without_head_rev(divergence_repo: Path) -> None:
    """测试未指定 head_rev 时搜工作区。"""
    tool = make_code_search_tool(ToolContext(project_root=str(divergence_repo)))
    assert "db.go" in tool.invoke({"pattern": r"WHERE id = \?"})


# ---------- report_hunks ----------


def test_report_hunks_records_comments_and_marks_done() -> None:
    """测试报告评论并把 hunk 标记为完成。"""
    ctx = make_ctx_with_hunks("H-01", "H-02")
    tool = make_report_hunks_tool(ctx)

    result = tool.invoke(
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

    assert "已记录" in result
    assert "剩余" in result
    assert ctx.hunk_status["H-01"].status == STATUS_DONE
    assert ctx.hunk_status["H-02"].status == STATUS_PENDING
    assert len(ctx.all_comments) == 1
    assert ctx.all_comments[0].content == "goroutine 缺少退出路径"
    assert ctx.all_comments[0].suggestion_code is None


def test_report_hunks_returns_remaining_not_completed() -> None:
    """测试返回值给「剩余未审」而非「已完成」。"""
    ctx = make_ctx_with_hunks("H-01", "H-02", "H-03")
    tool = make_report_hunks_tool(ctx)

    result = tool.invoke({"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]})
    assert "剩余未审：H-02, H-03" in result
    assert "1/3 完成" in result


def test_report_hunks_all_done_reports_none_pending() -> None:
    """测试全部完成时剩余为「无」。"""
    ctx = make_ctx_with_hunks("H-01")
    tool = make_report_hunks_tool(ctx)
    result = tool.invoke({"reports": [{"hunk_id": "H-01", "complete": True, "comments": []}]})
    assert "剩余未审：无" in result


def test_report_hunks_ignores_unknown_hunk_id() -> None:
    """测试未知 hunk 编号被忽略且提示。"""
    ctx = make_ctx_with_hunks("H-01")
    tool = make_report_hunks_tool(ctx)

    result = tool.invoke({"reports": [{"hunk_id": "H-99", "complete": True, "comments": []}]})
    assert "未知 hunk 编号" in result
    assert ctx.all_comments == []


def test_report_hunks_keeps_incomplete_hunk_pending() -> None:
    """测试 complete=False 时状态不置为 done。"""
    ctx = make_ctx_with_hunks("H-01")
    tool = make_report_hunks_tool(ctx)

    tool.invoke({"reports": [{"hunk_id": "H-01", "complete": False, "comments": []}]})
    assert ctx.hunk_status["H-01"].status == STATUS_PENDING


def test_report_hunks_preserves_unknown_severity() -> None:
    """测试非法 severity 原样保留（交由排序阶段处理），不改写为 low。"""
    ctx = make_ctx_with_hunks("H-01")
    tool = make_report_hunks_tool(ctx)

    tool.invoke(
        {
            "reports": [
                {
                    "hunk_id": "H-01",
                    "complete": True,
                    "comments": [
                        {
                            "severity": "CRITICAL",
                            "category": "x",
                            "content": "y",
                            "existing_code": "z",
                            "suggestion_code": "  ",
                        }
                    ],
                }
            ]
        }
    )
    assert ctx.all_comments[0].severity == "critical"
    assert ctx.all_comments[0].suggestion_code is None


# ---------- end_done ----------


def test_end_done_accepted_when_all_complete() -> None:
    """测试所有 hunk 完成时接受结束。"""
    ctx = ToolContext()
    ctx.hunk_status["H-01"] = HunkStatus(hunk_id="H-01", status=STATUS_DONE)
    tool = make_end_done_tool(ctx)

    result = tool.invoke({})
    assert "审查完成" in result
    assert ctx.end_done_count == 1


def test_end_done_rejected_on_first_call_with_pending() -> None:
    """测试有未完成 hunk 时第一次调用被拒绝。"""
    ctx = make_ctx_with_hunks("H-01", "H-02")
    tool = make_end_done_tool(ctx)

    result = tool.invoke({})
    assert "未审" in result
    assert "H-01, H-02" in result
    assert ctx.end_done_count == 1


def test_end_done_accepted_on_second_call_with_gaps() -> None:
    """测试第二次调用被接受并给出结构化告警。"""
    ctx = make_ctx_with_hunks("H-01", "H-02")
    tool = make_end_done_tool(ctx)

    tool.invoke({})
    result = tool.invoke({})

    assert "已接受结束请求" in result
    assert "H-01, H-02 未被审查" in result
    assert ctx.end_done_count == 2


def test_end_done_verdict_matches_tool_behavior() -> None:
    """测试 Main 循环与工具共用同一套判定规则。"""
    ctx = make_ctx_with_hunks("H-01")
    ctx.end_done_count = 1
    assert end_done_verdict(ctx) == (False, ["H-01"])

    ctx.end_done_count = 2
    assert end_done_verdict(ctx) == (True, ["H-01"])

    ctx.hunk_status["H-01"].status = STATUS_DONE
    assert end_done_verdict(ctx) == (True, [])
