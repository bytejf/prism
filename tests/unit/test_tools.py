"""工具系统的单元测试（file_read / file_read_diff / code_search / report_hunks / end_done）。

工具通过 `ToolRuntime` 拿静态上下文（`runtime.context`）与可变状态
（`runtime.state`），不再吃一个共享的 ToolContext 对象。
写状态的工具返回 `Command`，测试从 `Command.update` 里读结果。
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ValidationError

from prism.config.settings import settings
from prism.core.agent.agent import build_tools
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
from prism.core.diff.loader import build_diff_index
from prism.core.diff.parser import parse_diff_text
from prism.core.tools.code_search import code_search
from prism.core.tools.end_done import end_done
from prism.core.tools.file_read import file_read
from prism.core.tools.file_read_diff import file_read_diff
from prism.core.tools.paths import check_relative_path
from prism.core.tools.report_hunks import report_hunks
from prism.observability.llm_trace import _describe_tool

CONTEXT_WITHOUT_ROOT = ReviewContext(project_root="")


def test_tools_keep_runtime_injectable() -> None:
    """测试五个工具都还能拿到 `runtime` 注入，且它不出现在模型看到的 schema 里。

    回归用例：**不能改用 `args_schema=`**。显式给了它，框架就不再从函数签名
    推导，`runtime` 会从 `get_input_schema()` 里消失 —— 注入断掉，调用时报
    `TypeError: missing 1 required positional argument: 'runtime'`。
    """
    for item in build_tools():
        # get_input_schema 的返回类型是 pydantic v1/v2 模型的联合，框架没写窄
        input_schema = cast(type[BaseModel], item.get_input_schema())
        assert "runtime" in input_schema.model_fields, item.name
        parameters = convert_to_openai_tool(item)["function"]["parameters"]
        assert "runtime" not in parameters.get("properties", {}), item.name


def test_tool_descriptions_are_only_for_the_model() -> None:
    """测试描述里没有 `Args:` / `Returns:` / `Notes:` 段。

    回归用例：那三段是**写给开发者看**的，整段发给模型是纯噪声 ——
    实测 `end_done` 的描述 175 字符里有 110 个是它们。
    """
    for item in build_tools():
        description = convert_to_openai_tool(item)["function"]["description"]
        assert description, item.name
        for section in ("Args:", "Returns:", "Notes:"):
            assert section not in description, f"{item.name} 的描述里混进了 {section}"


def test_tool_parameters_are_all_described() -> None:
    """测试每个模型可见的参数都有说明 —— 参数摆到模型面前，它才不用猜。"""
    for item in build_tools():
        properties = convert_to_openai_tool(item)["function"]["parameters"].get("properties", {})
        for name, spec in properties.items():
            assert spec.get("description"), f"{item.name}.{name} 缺说明"


def test_describe_tool_logs_what_the_model_actually_gets() -> None:
    """测试日志打的是**转换之后**的 schema，与线上逐字节一致。

    回归用例：以前打 `tool_call_schema.model_json_schema()`，那是 Pydantic 转成
    OpenAI 格式**之前**的形态 —— 带 `title`、带参数顶层的 `description`、
    嵌套模型还是 `$defs` + `$ref`。而框架出去前会剥掉前两个、内联后两个，
    拿它核「模型看到了什么」会得出错的结论。
    """
    for item in build_tools():
        text = _describe_tool(item)
        assert "$defs" not in text, item.name
        assert "$ref" not in text, item.name
        assert '"title"' not in text, item.name
        # 描述整段在，且与线上一致
        assert item.description in text, item.name


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
    limit = settings.tools.file_read.max_lines
    (tmp_path / "big.go").write_text("x\n" * (limit + 100), encoding="utf-8")
    tool = ToolUnderTest(file_read, ctx)

    result = tool.text({"path": "big.go", "start_line": 1, "end_line": limit + 100})
    assert "截断" in result
    assert result.count("\n") <= limit + 1


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


# ---------- file_read_diff ----------

# 两个文件的 diff。样本走**真实解析器**生成 Diff，不手搓对象 ——
# 手搓的话解析器改了、这里却还是绿的
TWO_FILE_DIFF = """\
diff --git a/a.go b/a.go
--- a/a.go
+++ b/a.go
@@ -3,2 +3,3 @@
-old()
+new()
+extra()
diff --git a/b.go b/b.go
--- a/b.go
+++ b/b.go
@@ -9 +9 @@
-ping()
+pong()
"""

RENAMED_DIFF = """\
diff --git a/old.go b/new.go
similarity index 90%
rename from old.go
rename to new.go
--- a/old.go
+++ b/new.go
@@ -1 +1 @@
-package old
+package new
"""

NEW_FILE_DIFF = """\
diff --git a/fresh.go b/fresh.go
new file mode 100644
--- /dev/null
+++ b/fresh.go
@@ -0,0 +1,2 @@
+package fresh
+var x = 1
"""


def diff_context(text: str) -> ReviewContext:
    """由 diff 文本构造带索引的上下文。

    Args:
        text: unified diff 文本

    Returns:
        ReviewContext，diffs 为按路径建好的索引
    """
    return ReviewContext(project_root="", diffs=build_diff_index(parse_diff_text(text)))


def test_file_read_diff_returns_only_the_requested_file() -> None:
    """测试按路径取到该文件的完整 diff（含 hunk 头），且不含别的文件。

    只给一个文件的改动是本工具存在的意义：模型要判断「调用方有没有跟着改」，
    拿一堆别的文件的 diff 只会烧 token。
    """
    tool = ToolUnderTest(file_read_diff, diff_context(TWO_FILE_DIFF))

    result = tool.text({"path": "a.go"})

    assert "diff --git a/a.go b/a.go" in result
    assert "@@ -3,2 +3,3 @@" in result, "hunk 头必须给 —— 行号信息只在这里"
    assert "+extra()" in result
    assert "pong()" not in result, "不该夹带别的文件"


def test_file_read_diff_hunk_header_always_spells_out_counts() -> None:
    """测试行数为 1 时也写 `,1`，不学 git 省略。

    省略的那个数正是模型数行号的起点（`@@ -9 +9 @@` 里 9 是起始行还是行数？），
    省掉的字节换来一次误读不划算。
    """
    tool = ToolUnderTest(file_read_diff, diff_context(TWO_FILE_DIFF))

    assert "@@ -9,1 +9,1 @@" in tool.text({"path": "b.go"})


def test_file_read_diff_reports_unknown_path() -> None:
    """测试查不到时给出说明，而不是空串。

    被过滤的文件与「确实没改」在这里是同一句话：索引里只有可审查文件，
    说不清是哪种。见 file_read_diff 模块 docstring。
    """
    tool = ToolUnderTest(file_read_diff, diff_context(TWO_FILE_DIFF))

    result = tool.text({"path": "nope.go"})

    assert "未找到" in result
    assert "nope.go" in result


def test_file_read_diff_reports_missing_index(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """测试索引为空时**报错**，而不是回「没有改动」。

    回归用例：索引没装上（装配错误）时若按「查不到 = 没改」处理，每次查询都
    得到「没有改动」，模型据此判断「调用方没同步改」并报出假问题 ——
    一条不报错的错链路。这是「静默失效比丢东西更难发现」的那一类。
    """
    tool = ToolUnderTest(file_read_diff, ReviewContext(project_root=""))

    with caplog.at_level(logging.WARNING):
        result = tool.text({"path": "a.go"})

    assert "错误" in result
    assert "diffs 为空" in caplog.text


def test_file_read_diff_finds_renamed_file_by_old_path() -> None:
    """测试改名后的文件用**旧路径**也查得到。

    提示词里带着 `renamed-from="old.go"`，模型很可能拿着旧名字来查；
    查不到就会得到「没有改动」这个错答案。
    """
    tool = ToolUnderTest(file_read_diff, diff_context(RENAMED_DIFF))

    by_old = tool.text({"path": "old.go"})
    by_new = tool.text({"path": "new.go"})

    assert by_old == by_new, "两个名字指向同一份改动"
    assert "rename from old.go" in by_old, "新旧名字要在文本里对得上号"


def test_file_read_diff_marks_new_and_deleted_files() -> None:
    """测试新增 / 删除文件的两侧用 /dev/null 标出来。"""
    tool = ToolUnderTest(file_read_diff, diff_context(NEW_FILE_DIFF))

    result = tool.text({"path": "fresh.go"})

    assert "new file" in result
    assert "--- /dev/null" in result
    assert "+++ b/fresh.go" in result
    # 新增文件在 diff --git 行里仍写 a/x：git 自己就是这么写的
    assert "diff --git a/fresh.go b/fresh.go" in result


def test_file_read_diff_rejects_empty_path() -> None:
    """测试空路径给出明确错误，而不是拼出「未找到 的改动」这种半截话。"""
    tool = ToolUnderTest(file_read_diff, diff_context(TWO_FILE_DIFF))

    assert "错误" in tool.text({"path": "  "})


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
    result = tool.text({"pattern": "func", "max_results": settings.tools.code_search.max_results})
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


def test_code_search_strips_revision_prefix(divergence_repo: Path) -> None:
    """测试带 rev 搜索时结果**不带** `<rev>:` 前缀。

    回归用例：`git grep <rev>` 每条结果是 `feature:db.go:12: ...`。前缀留着的话，
    模型照抄整行里的路径去调 file_read / file_read_diff 就查不到 —— 前缀不是路径的
    一部分，而它没有任何理由知道这件事。这里顺带断言剥完之后拼出来的**真的是个
    存在的文件**：前缀少剥一个字符（如 `eature:`）同样会红。
    """
    ctx = ReviewContext(project_root=str(divergence_repo), head_rev="feature")
    tool = ToolUnderTest(code_search, ctx)

    result = tool.text({"pattern": r"WHERE id = '"})

    assert "feature:" not in result
    path, line_number, _content = result.split("\n")[0].split(":", 2)
    assert (divergence_repo / path).is_file(), f"剥出来的不是可用路径：{path!r}"
    assert line_number.isdigit()


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

    这给了 STATUS_IN_PROGRESS 一条真实的产生路径，<review_status> 的
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


# ---------- 路径安全：模型给的参数是我们的信任边界 ----------


@pytest.mark.parametrize("value", ["", "   ", "/etc/hosts", "../outside.go", "a/../../b", "a\x00b"])
def test_check_relative_path_rejects(value: str) -> None:
    """测试越界与非法字符在我们这一侧就被拦下。

    拦住它们是**为了给出统一的失败**：放到下游，`../*` 会变成 git 的英文 stderr
    （还带本机绝对路径），NUL 会让 `subprocess` 抛没人接的异常。
    """
    with pytest.raises(ValueError):
        check_relative_path(value)


@pytest.mark.parametrize(
    "value", ["a.go", "internal/agent/x.go", "*.go", ".", "a b.go", ":(top)*.go"]
)
def test_check_relative_path_accepts(value: str) -> None:
    """测试合法的路径与 pathspec 不被误杀 —— 校验不能比 git 自己更窄。"""
    check_relative_path(value)


def test_file_read_blocks_symlink_escape(tmp_path: Path) -> None:
    """测试仓库内指向外面的符号链接读不到外面。

    这是字符串校验**唯一挡不住**的一类：`link.go` 字面上是合法相对路径。
    拦住它的是读取侧的 `resolve()` + `is_relative_to`。
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("OUTSIDE_CONTENT\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "link.go").symlink_to(secret)

    tool = ToolUnderTest(file_read, ReviewContext(project_root=str(repo)))
    result = tool.text({"path": "link.go", "start_line": 1, "end_line": 1})

    assert "OUTSIDE_CONTENT" not in result
    assert "错误" in result


def test_file_read_rejects_nul_instead_of_raising() -> None:
    """测试 NUL 返回错误文本，而不是抛异常。

    实测过：NUL 在下游会让 `Path.resolve()` 抛 `ValueError: embedded null
    character` —— 工具不接、框架默认也不接，整组审查就没了。
    """
    tool = ToolUnderTest(file_read, ReviewContext(project_root="/tmp"))

    assert "错误" in tool.text({"path": "a\x00b.go", "start_line": 1, "end_line": 1})


@pytest.mark.parametrize(
    "args",
    [{"pattern": "x", "file_pattern": "../*"}, {"pattern": "x", "file_pattern": "/etc/*"}],
)
def test_code_search_blocks_escaping_pathspec(go_repo: Path, args: dict[str, str]) -> None:
    """测试越界 pathspec 由**我们**回中文错误，不再把 git 的 stderr 甩给模型。

    原来靠 git 自己拒（`fatal: outside repository at '/private/...'`）——
    挡住是挡住了，但那条消息把本机绝对路径一起送给了外部 API。
    """
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))

    result = tool.text(args)

    assert "错误" in result
    assert "outside repository" not in result, "不该把 git 的英文 stderr 透传给模型"
    assert str(go_repo) not in result, "不该把本机路径写进送给模型的文本"


@pytest.mark.parametrize(
    "args",
    [{"pattern": "a\x00b"}, {"pattern": "func", "file_pattern": "a\x00b"}],
)
def test_code_search_rejects_nul_args(go_repo: Path, args: dict[str, str]) -> None:
    """测试两个参数里的 NUL 都被拦住 —— 它们都要进 argv。

    回归用例：`subprocess.run` 的 argv 不接受 NUL，抛的是 `ValueError`，
    而框架默认只把 `ToolInvocationError` 转成工具结果，其余原样抛 ——
    实测 `run_review_agent` 直接把异常抛出来，整条 review 结束。
    """
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))

    assert "错误" in tool.text(args)


def test_code_search_keeps_dash_arguments_as_pathspec(go_repo: Path) -> None:
    """测试以 `-` 开头的参数被 `--` 挡成 pathspec，不会变成 git 的选项。"""
    tool = ToolUnderTest(code_search, ReviewContext(project_root=str(go_repo)))

    # 当选项的话 git 会去搜索引；当 pathspec 就是「没有这个文件」
    assert "未找到匹配" in tool.text({"pattern": "func Alpha", "file_pattern": "--cached"})


def test_file_read_diff_has_nothing_to_escape() -> None:
    """测试 file_read_diff 对越界路径只有「查不到」一种结局。

    它不碰文件系统（纯字典查询），所以没有需要防的东西 —— 这条用例是把
    「不需要校验」这个判断钉住，免得以后有人照着 file_read 抄一份进来。
    """
    tool = ToolUnderTest(file_read_diff, diff_context(TWO_FILE_DIFF))

    for path in ("../../etc/passwd", "/etc/passwd", "a\x00b"):
        assert "未找到" in tool.text({"path": path})
