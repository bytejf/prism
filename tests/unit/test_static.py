"""静态分析（analyzer / verify）的单元测试。

**全都不依赖真实的 golangci-lint**：二进制是外部依赖，不在 `pyproject.toml` 里，
装了才有。所以这里只喂**真实的 JSON 载荷**（从 v2.14.0 的输出里抄下来的形状）
给解析层，用 monkeypatch 造子进程的失败形态。真跑一次工具的场景在
`tests/manual/check_static.py`。
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from prism.core.static import analyzer
from prism.core.static.analyzer import (
    LINTERS,
    analyze,
    format_finding,
    head_tree,
    select_for_group,
)
from prism.core.static.verify import verify_group
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.comment import Comment
from prism.models.diff import Hunk
from prism.models.static import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    StaticFinding,
    StaticReport,
)

# ---------- 造数据 ----------


def make_hunk(
    hunk_id: str,
    file_path: str = "pkg/a/iface.go",
    *,
    new: tuple[int, int] = (10, 3),
    head_lines: tuple[int, int] | None = (5, 20),
) -> HunkWithSymbol:
    """构造测试用的带符号 hunk。

    Args:
        hunk_id: hunk 编号
        file_path: 文件路径
        new: 新侧 (起始行, 行数)
        head_lines: 归属符号在被审查版本里的行号范围

    Returns:
        HunkWithSymbol
    """
    return HunkWithSymbol(
        part=Hunk(old_start=10, old_lines=3, new_start=new[0], new_lines=new[1], content="+x"),
        file_path=file_path,
        symbol=Symbol(name="F", kind="function", start_line=1, end_line=30),
        hunk_id=hunk_id,
        head_lines=head_lines,
        renamed_from=None,
        deleted=False,
    )


def make_finding(
    finding_id: str = "",
    *,
    path: str = "pkg/a/iface.go",
    line: int = 10,
    linter: str = "gosec",
    priority: str = PRIORITY_HIGH,
    text: str = "G107: Potential HTTP request made with variable url",
    hunk_id: str | None = None,
) -> StaticFinding:
    """构造测试用的静态发现。

    Args:
        finding_id: 编号
        path: 文件路径
        line: 行号
        linter: linter 名
        priority: 工具标记
        text: 描述
        hunk_id: 关联 hunk

    Returns:
        StaticFinding
    """
    return StaticFinding(
        finding_id=finding_id,
        path=path,
        line=line,
        linter=linter,
        priority=priority,
        text=text,
        hunk_id=hunk_id,
    )


def make_comment(content: str) -> Comment:
    """构造测试用的评论（只关心正文）。

    Args:
        content: 评论正文

    Returns:
        Comment
    """
    return Comment(
        path="pkg/a/iface.go",
        severity="high",
        category="bug",
        content=content,
        existing_code="x",
        hunk_id="H-01",
        locate_level="hunk",
    )


def issue(path: str, line: int, linter: str, text: str) -> dict[str, Any]:
    """造一条 golangci-lint 的原始 issue（形状照抄 v2.14.0 的真实输出）。

    Args:
        path: 绝对路径
        line: 行号
        linter: linter 名
        text: 描述

    Returns:
        issue 字典
    """
    return {
        "FromLinter": linter,
        "Text": text,
        "Severity": "",
        "SourceLines": ["x"],
        "Pos": {"Filename": path, "Offset": 0, "Line": line, "Column": 5},
        "ExpectNoLint": False,
        "ExpectedNoLintLinter": "",
    }


def payload(*issues: dict[str, Any]) -> str:
    """包成 golangci-lint 的完整 JSON 输出。

    Args:
        issues: issue 列表

    Returns:
        JSON 文本
    """
    return json.dumps({"Issues": list(issues), "Report": {"Linters": []}})


def fake_run(stdout: str = "", returncode: int = 0, stderr: str = "") -> Any:
    """造一个假的 subprocess.run 返回值。

    Args:
        stdout: 标准输出
        returncode: 退出码
        stderr: 标准错误

    Returns:
        CompletedProcess
    """

    def _raise() -> None:
        raise AssertionError("测试里不应读 check 之外的属性")

    return subprocess.CompletedProcess(
        args=["golangci-lint"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------- 解析：路径、优先级、描述 ----------


def test_parse_keeps_only_reviewable_files(tmp_path: Path) -> None:
    """只保留可审查文件的发现 —— 没改动的文件不在审查范围内。"""
    tree = tmp_path / "tree"
    tree.mkdir()
    stdin = payload(
        issue(str(tree / "api/serve.go"), 12, "gosec", "G107: x"),
        issue(str(tree / "untouched/other.go"), 3, "gosec", "G107: y"),
    )
    findings, raw_count = analyzer._parse(json.loads(stdin), tree, ["api/serve.go"])

    assert raw_count == 2, "原始条数必须报出来，否则「全被过滤」看起来像「没发现问题」"
    assert [item.path for item in findings] == ["api/serve.go"]
    assert findings[0].line == 12


def test_parse_drops_paths_outside_the_analysed_tree(tmp_path: Path) -> None:
    """分析目录之外的路径一律丢掉，不猜相对路径。"""
    tree = tmp_path / "tree"
    tree.mkdir()
    outside = tmp_path / "elsewhere/api.go"
    outside.parent.mkdir()
    outside.write_text("package api\n")

    findings, raw_count = analyzer._parse(
        json.loads(payload(issue(str(outside), 1, "gosec", "G107: x"))),
        tree,
        ["api/serve.go", "api.go"],
    )

    # 路径不在 tree 里 → 丢；它不会因为「文件名相同」而被误当成可审查文件
    assert findings == []
    assert raw_count == 1


def test_priority_comes_from_the_linter_table() -> None:
    """优先级按 linter 名映射，不看工具自己的 Severity 字段。"""
    items = [
        issue("/t/a.go", 1, "gosec", "x"),
        issue("/t/a.go", 2, "typecheck", "x"),
        issue("/t/a.go", 3, "errcheck", "x"),
        issue("/t/a.go", 4, "unused", "x"),
        issue("/t/a.go", 5, "某个新 linter", "x"),
    ]
    priorities = [analyzer._to_finding(item, "a.go").priority for item in items]

    assert priorities == [
        PRIORITY_HIGH,
        PRIORITY_HIGH,
        PRIORITY_MEDIUM,
        PRIORITY_LOW,
        PRIORITY_MEDIUM,
    ]


def test_text_is_flattened_and_truncated() -> None:
    """描述被压成一行并截断 —— 表格单元格里放不下换行。"""
    finding = analyzer._to_finding(
        issue("/t/a.go", 1, "typecheck", "# pkg/x\na.go:14:9: undefined: Old"), "a.go"
    )
    assert "\n" not in finding.text
    assert finding.text.startswith("# pkg/x a.go:14:9:")

    long_text = analyzer._to_finding(issue("/t/a.go", 1, "gosec", "x" * 500), "a.go")
    assert len(long_text.text) == analyzer.MAX_TEXT_LENGTH + 1
    assert long_text.text.endswith("…")


# ---------- 关联 hunk ----------


def test_line_inside_hunk_wins_over_symbol_range() -> None:
    """落在改动行区间里的发现关联到那片 hunk。"""
    item = analyzer._associate([make_finding(line=12)], [make_hunk("H-02", new=(11, 3))])
    assert item[0].hunk_id == "H-02"


def test_line_outside_hunk_falls_back_to_symbol_range() -> None:
    """落在符号范围、但不在改动行上的发现，关联到该符号所在的 hunk。"""
    item = analyzer._associate(
        [make_finding(line=7)], [make_hunk("H-02", new=(11, 3), head_lines=(5, 20))]
    )
    assert item[0].hunk_id == "H-02"


def test_line_outside_everything_has_no_hunk() -> None:
    """落在符号之外（比如改了这个文件里另一个函数）就没有关联。"""
    item = analyzer._associate(
        [make_finding(line=99)], [make_hunk("H-02", new=(11, 3), head_lines=(5, 20))]
    )
    assert item[0].hunk_id is None
    assert "—" in format_finding(item[0]), "关联不上时显示占位符，不是空串"


def test_pure_deletion_hunk_matches_nothing_on_the_new_side() -> None:
    """纯删除的 hunk 新侧行数为 0，不该「命中」第 0 行之后的任何位置。"""
    deleted = make_hunk("H-04", new=(7, 0), head_lines=None)
    item = analyzer._associate([make_finding(line=7)], [deleted])
    assert item[0].hunk_id is None


def test_association_is_per_file() -> None:
    """别的文件的同号行不会误关联。"""
    hunks = [make_hunk("H-01", file_path="pkg/a/iface.go", new=(10, 5))]
    item = analyzer._associate([make_finding(path="pkg/b/impl.go", line=11)], hunks)
    assert item[0].hunk_id is None


# ---------- 排序与编号 ----------


def test_order_and_number_puts_high_first_and_keeps_ids_continuous() -> None:
    """优先级降序 → 路径 → 行号，编号全局连续。"""
    raw = [
        make_finding(path="pkg/b.go", line=1, linter="unused", priority=PRIORITY_LOW),
        make_finding(path="pkg/a.go", line=9, linter="errcheck", priority=PRIORITY_MEDIUM),
        make_finding(path="pkg/a.go", line=3, linter="gosec", priority=PRIORITY_HIGH),
    ]
    ordered = analyzer._order_and_number(raw)

    assert [item.finding_id for item in ordered] == ["S-01", "S-02", "S-03"]
    assert [item.linter for item in ordered] == ["gosec", "errcheck", "unused"]


# ---------- 按文件分桶 ----------


def test_select_for_group_buckets_by_file() -> None:
    """只挑本组文件上的发现，顺序不变（因此截断先丢低优先级）。"""
    findings = [
        make_finding("S-01", path="pkg/a.go"),
        make_finding("S-02", path="pkg/b.go"),
        make_finding("S-03", path="pkg/a.go", priority=PRIORITY_LOW),
    ]
    selected = select_for_group(["pkg/a.go"], findings)
    assert [item.finding_id for item in selected] == ["S-01", "S-03"]


# ---------- 失败降级 ----------


def _tree_with_one_go_file(tmp_path: Path) -> Path:
    """造一个最小可分析目录。

    Args:
        tmp_path: pytest 的临时目录

    Returns:
        目录路径
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "go.mod").write_text("module x\n\ngo 1.22\n", encoding="utf-8")
    (tree / "a.go").write_text("package main\n", encoding="utf-8")
    return tree


def test_missing_binary_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """二进制不存在 → 跳过并说明原因，不抛异常。"""
    tree = _tree_with_one_go_file(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(analyzer._AnalysisError, match="找不到 golangci-lint"):
        analyzer._run_linter(tree, ["."])


def test_timeout_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """超时 → 跳过。"""
    tree = _tree_with_one_go_file(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="golangci-lint", timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(analyzer._AnalysisError, match="超时"):
        analyzer._run_linter(tree, ["."])


def test_nonzero_exit_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """退出码非 0 → 跳过，并把工具的 stderr 末尾带上。

    实测量过：目标不是 Go 模块时 golangci-lint 退出码 7，stdout 里仍有一份带
    `Report.Error` 的 JSON —— 只看 JSON 会把它当成「跑成功且零发现」。
    """
    tree = _tree_with_one_go_file(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: fake_run(
            stdout=payload(), returncode=7, stderr='level=error msg="not a module"\n'
        ),
    )
    with pytest.raises(analyzer._AnalysisError, match="退出码 7"):
        analyzer._run_linter(tree, ["."])


def test_broken_json_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """stdout 不是合法 JSON → 跳过（`--show-stats` 没关就是这种）。"""
    tree = _tree_with_one_go_file(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: fake_run(stdout='{"Issues":[]}\n1 issues:\n')
    )
    with pytest.raises(analyzer._AnalysisError, match="不是合法 JSON"):
        analyzer._run_linter(tree, ["."])


def test_analyze_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """装配层：任何失败都变成带原因的 StaticReport，审查照常继续。"""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "nope")

    monkeypatch.setattr(subprocess, "run", boom)
    # head_rev 传空 → 不建 worktree，于是这条路径上唯一会被执行的外部命令就是
    # golangci-lint，测的正是「它起不来」这一种降级
    report = analyze(tmp_path, "", ["pkg/a.go"], [])

    assert isinstance(report, StaticReport)
    assert report.skipped
    assert report.findings == []


def test_empty_reviewable_skips_without_running() -> None:
    """没有可审查文件时不去起进程。"""
    report = analyze(Path("/nonexistent"), "HEAD", [], [])
    assert report.skipped is False
    assert report.findings == []


# ---------- 命令形态：这一串 flag 是实测出来的，少一个就拿不到纯 JSON ----------


def test_command_flags_are_the_ones_we_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """守住命令行的形态。

    这条是**静默失效的守卫**：少了 `--show-stats=false` 或
    `--output.text.path=stderr`，stdout 就不再是纯 JSON，而现象是「解析失败 →
    跳过静态分析」，看起来像工具没装。
    """
    seen: dict[str, list[str]] = {}

    def capture(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        return fake_run(stdout=payload())

    monkeypatch.setattr(subprocess, "run", capture)
    analyzer._run_linter(_tree_with_one_go_file(tmp_path), ["."])

    command = seen["command"]
    assert command[0] == "golangci-lint"
    assert "--no-config" in command, "不能让被审仓库的配置改变我们的结论"
    assert "--default=none" in command
    assert command[command.index("-E") + 1] == ",".join(LINTERS)
    assert "--issues-exit-code=0" in command, "退出码只表示「跑没跑成」"
    assert "--max-issues-per-linter=0" in command, "工具自己的上限会静默吃掉发现"
    assert "--max-same-issues=0" in command
    assert "--output.json.path=stdout" in command
    assert "--output.text.path=stderr" in command
    assert "--show-stats=false" in command
    assert "--path-mode=abs" in command, "相对路径的基准不可控，我们自己算"
    assert not any(flag.startswith("--new-from") for flag in command), (
        "不做 CI 那套 git blame 口径（设计 18 号：改过的文件里的发现全要）"
    )


def test_linter_list_excludes_gosimple() -> None:
    """`gosimple` 在 v2 已并入 staticcheck，留着会让整次分析直接失败。"""
    assert "gosimple" not in LINTERS
    assert "staticcheck" in LINTERS


# ---------- worktree：分析被审查版本而不是工作区 ----------


def _init_repo(repo: Path) -> None:
    """造一个有两个提交的临时仓库。

    Args:
        repo: 仓库路径
    """
    repo.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.com"),
        ("config", "user.name", "t"),
        ("add", "-A"),
        ("commit", "-qm", "base"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_head_tree_checks_out_the_reviewed_revision(tmp_path: Path) -> None:
    """head_tree 里看到的是 head 的内容，退出后目录被清掉。

    这条是「工具读被审查版本」这条原则在静态分析这一侧的守卫。
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.go").write_text("package main\n// base\n", encoding="utf-8")
    _init_repo(repo)
    # 工作区停在 base 之后的新提交上，worktree 取的是更早的那个提交
    earlier = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "a.go").write_text("package main\n// head\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "head"], check=True)

    with head_tree(repo, earlier) as tree:
        assert "// base" in (tree / "a.go").read_text(encoding="utf-8")
        assert tree != repo, "分析的是检出目录，不是工作区"
        created = tree
    assert not created.exists(), "退出时必须清掉临时检出"


def test_head_tree_without_rev_uses_the_working_tree(tmp_path: Path) -> None:
    """没给 head 时直接用工作区，不白建一次检出。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    with head_tree(repo, "") as tree:
        assert tree == repo


def test_head_tree_raises_on_bad_revision(tmp_path: Path) -> None:
    """版本不存在 → 抛内部信号，由 analyze 转成 skip_reason。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.go").write_text("package main\n", encoding="utf-8")
    _init_repo(repo)

    with pytest.raises(analyzer._AnalysisError, match="worktree"), head_tree(repo, "不存在的分支"):
        pass


def test_temp_worktrees_are_removed_from_the_repo(tmp_path: Path) -> None:
    """清理之后 git 里不留工作树登记项（否则仓库会越用越脏）。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.go").write_text("package main\n", encoding="utf-8")
    _init_repo(repo)

    with head_tree(repo, "HEAD"):
        pass
    listing = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], check=True, capture_output=True, text=True
    ).stdout
    assert listing.count("\n") == 1, f"只该剩主工作树：{listing}"


# ---------- 后置校验 ----------


def test_verify_marks_referenced_and_unhandled() -> None:
    """引用到的算「有人管」，没被引用的高优先级算「未处理」。"""
    findings = [
        make_finding("S-01", priority=PRIORITY_HIGH),
        make_finding("S-02", priority=PRIORITY_HIGH),
        make_finding("S-03", priority=PRIORITY_LOW),
    ]
    verdict = verify_group(findings, [make_comment("[S-01] 成立，会 SSRF")])

    assert verdict.referenced == ("S-01",)
    assert verdict.invalid == ()
    assert [item.finding_id for item in verdict.unhandled_high] == ["S-02"]
    assert verdict.unhandled_count == 1, "低优先级的 S-03 不进 WARNING"


def test_verify_flags_unknown_ids() -> None:
    """引用了本组没注入的编号 → 无效（哪怕它在别组存在）。"""
    findings = [make_finding("S-01"), make_finding("S-02")]
    verdict = verify_group(findings, [make_comment("[S-07] 我认为这里有问题")])

    assert verdict.referenced == ()
    assert verdict.invalid == ("S-07",)
    assert verdict.unhandled_count == 2


def test_verify_accepts_sloppy_spelling() -> None:
    """大小写与补零不是校验要抓的东西。"""
    verdict = verify_group([make_finding("S-01")], [make_comment("[s-1] 成立")])
    assert verdict.referenced == ("S-01",)


def test_verify_without_findings_is_empty() -> None:
    """这一组本来就没有静态发现时不产生任何结论。"""
    verdict = verify_group([], [make_comment("[S-01] 成立")])
    assert verdict.referenced == ()
    assert verdict.invalid == ()
    assert verdict.unhandled_count == 0


def test_verify_only_counts_kept_comments() -> None:
    """只按传进来的评论算 —— 反思删掉的评论不该让一条发现重新变成「未处理」。"""
    findings = [make_finding("S-01")]
    assert verify_group(findings, []).unhandled_count == 1
    assert verify_group(findings, [make_comment("[S-01] 成立")]).unhandled_count == 0


# ---------- 只分析改动的包 ----------


def test_packages_are_the_dirs_of_changed_files(tmp_path: Path) -> None:
    """改动的包 = 改动文件所在目录的集合（去重、排序）。"""
    tree = tmp_path / "tree"
    (tree / "internal/ls").mkdir(parents=True)
    (tree / "internal/core").mkdir(parents=True)

    packages = analyzer._packages(
        ["internal/ls/a.go", "internal/ls/b.go", "internal/core/c.go"], tree
    )

    assert packages == ["internal/core", "internal/ls"]


def test_packages_drops_missing_dirs(tmp_path: Path) -> None:
    """目录在检出里不存在就不给它（整体删除的文件）—— 交给 linter 会让整次分析失败。"""
    tree = tmp_path / "tree"
    (tree / "keep").mkdir(parents=True)

    assert analyzer._packages(["keep/a.go", "gone/b.go"], tree) == ["keep"]


def test_packages_handles_root_files(tmp_path: Path) -> None:
    """根目录下的文件 → `.`（Go 的根包）。"""
    tree = tmp_path / "tree"
    tree.mkdir()

    assert analyzer._packages(["main.go"], tree) == ["."]
    assert analyzer._package_args(["."]) == ["."]
    assert analyzer._package_args(["internal/ls"]) == ["./internal/ls"]


def test_linter_is_scoped_to_the_changed_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """命令行要带上改动包，**不能让它默认 `./...` 跑全仓**。

    实测代价：78MB 的 Go 仓库只改了 1 个文件，全仓跑 77.5 秒（占整条审查 61%）、
    597 条发现里可用 0 条。
    """
    seen: dict[str, list[str]] = {}

    def capture(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        return fake_run(stdout=payload())

    monkeypatch.setattr(subprocess, "run", capture)
    analyzer._run_linter(_tree_with_one_go_file(tmp_path), ["internal/ls", "."])

    command = seen["command"]
    assert command[-2:] == ["./internal/ls", "."], "包参数要加在最后"
