"""diff 加载的单元测试（用临时 git 仓库）。"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pygit2
import pytest

from prism.core.diff.loader import build_diff_index, load_diffs, resolve_commit
from prism.core.diff.parser import parse_diff_text


@pytest.fixture
def sample_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造含 main / feature 两个分支的临时仓库。

    main:    a.go（a() 无输出）
    feature: a.go（a() 加输出）+ 新增 b.go

    Returns:
        仓库路径
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "a.go").write_text("package main\n\nfunc a() {}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.go").write_text("package main\n\nfunc a() { println(1) }\n", encoding="utf-8")
    (repo / "b.go").write_text("package main\n\nfunc b() {}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_load_diffs_returns_changed_files(sample_repo: Path) -> None:
    """测试加载出的文件集合与新增标记。"""
    diffs = load_diffs(sample_repo, "main", "feature")
    assert {d.path for d in diffs} == {"a.go", "b.go"}

    new_file = next(d for d in diffs if d.path == "b.go")
    assert new_file.is_new
    assert new_file.old_path == "/dev/null"

    modified = next(d for d in diffs if d.path == "a.go")
    assert not modified.is_new
    assert modified.insertions == 1
    assert modified.deletions == 1


def test_load_diffs_uses_merge_base(sample_repo: Path, git: Callable[..., None]) -> None:
    """测试以 merge-base 为基准：base 分支上的新提交不进入 diff。"""
    git(sample_repo, "checkout", "-q", "main")
    (sample_repo / "c.go").write_text("package main\n\nfunc c() {}\n", encoding="utf-8")
    git(sample_repo, "add", ".")
    git(sample_repo, "commit", "-q", "-m", "main only")

    diffs = load_diffs(sample_repo, "main", "feature")
    assert {d.path for d in diffs} == {"a.go", "b.go"}


def test_load_diffs_no_change(sample_repo: Path) -> None:
    """测试同一版本比较返回空列表。"""
    assert load_diffs(sample_repo, "feature", "feature") == []


def test_resolve_commit_rejects_unknown_rev(sample_repo: Path) -> None:
    """测试无法解析的版本抛 ValueError。"""
    repo = pygit2.Repository(str(sample_repo))
    with pytest.raises(ValueError, match="无法解析版本"):
        resolve_commit(repo, "no-such-branch")


def test_load_diffs_rejects_non_repo(tmp_path: Path) -> None:
    """测试非仓库路径抛 ValueError。"""
    with pytest.raises(ValueError, match="无法打开仓库"):
        load_diffs(tmp_path / "not-a-repo" / "deep", "main", "HEAD")


@pytest.fixture
def edge_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造含二进制文件 / 删除文件 / 重命名文件的临时仓库。

    main:    a.go、del.go、old.go、logo.png
    feature: a.go 改动、del.go 删除、old.go 重命名 + 改内容、logo.png 二进制改动

    Returns:
        仓库路径
    """
    repo = tmp_path / "edge"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "a.go").write_text("package main\n", encoding="utf-8")
    (repo / "del.go").write_text("old content\n", encoding="utf-8")
    # 5 行文件，重命名后只改 1 行：相似度 58%，git 会判为 R058 重命名
    (repo / "old.go").write_text("package main\n\nfunc old() {\n\treturn\n}\n", encoding="utf-8")
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.go").write_text("package main\n\nfunc a() {}\n", encoding="utf-8")
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 16)
    git(repo, "rm", "-q", "del.go")
    git(repo, "mv", "old.go", "renamed.go")
    (repo / "renamed.go").write_text(
        "package main\n\nfunc renamed() {\n\treturn\n}\n", encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_load_diffs_detects_rename_as_one_file(edge_repo: Path) -> None:
    """测试重命名被识别为一个文件，而不是拆成「删除 + 新增」两个。"""
    diffs = load_diffs(edge_repo, "main", "feature")
    assert {d.path for d in diffs} == {"a.go", "del.go", "logo.png", "renamed.go"}

    renamed = next(d for d in diffs if d.path == "renamed.go")
    assert renamed.old_path == "old.go"
    assert renamed.new_path == "renamed.go"
    assert not renamed.is_new
    assert not renamed.is_deleted


def test_load_diffs_marks_binary(edge_repo: Path) -> None:
    """测试二进制文件被标记且不产生 hunk。"""
    diffs = load_diffs(edge_repo, "main", "feature")
    binary = next(d for d in diffs if d.path == "logo.png")
    assert binary.is_binary
    assert binary.hunks == []
    assert binary.insertions == 0
    assert binary.deletions == 0


def test_load_diffs_marks_deleted(edge_repo: Path) -> None:
    """测试删除文件：is_deleted 为真，path 取 old_path。"""
    diffs = load_diffs(edge_repo, "main", "feature")
    deleted = next(d for d in diffs if d.new_path == "/dev/null")
    assert deleted.is_deleted
    assert deleted.old_path == "del.go"
    assert deleted.path == "del.go"
    assert deleted.deletions == 1


def test_load_diffs_rename_with_content_change(edge_repo: Path) -> None:
    """测试重命名同时改内容时解析出 hunk。"""
    diffs = load_diffs(edge_repo, "main", "feature")
    renamed = next(d for d in diffs if d.path == "renamed.go")
    assert len(renamed.hunks) == 1
    assert renamed.insertions == 1
    assert renamed.deletions == 1


BLANK_LINE_BASE = """package main

func Alpha() {
	println("a")
}

// Delta 只用来把 Alpha 与 Beta 隔开，
// 让两处改动相距足够远、落进不同的 hunk。
func Delta() {
	println("d")
}

func Beta() {
	println("b")
	println("b2")
}

func Gamma() {
	println("c")
}
"""

BLANK_LINE_HEAD = """package main

func Alpha() {
	println("a")
	println("a2")
}

// Delta 只用来把 Alpha 与 Beta 隔开，
// 让两处改动相距足够远、落进不同的 hunk。
func Delta() {
	println("d")
}

func Beta() {
	println("b")
}

func Gamma() {
	println("c")
}
"""


@pytest.fixture
def blank_line_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造含空行上下文与纯删除 hunk 的临时仓库。

    feature 相对 main：Alpha 加一行、Beta 删一行、函数之间都有空行。

    Returns:
        仓库路径
    """
    repo = tmp_path / "blank"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "a.go").write_text(BLANK_LINE_BASE, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.go").write_text(BLANK_LINE_HEAD, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_hunk_header_lines_match_body_walk(blank_line_repo: Path) -> None:
    """测试 hunk 头声明的行数与正文遍历结果一致。

    这条不变量是「hunk 块范围」和「改动行号」可信的前提：
    从 new_start / old_start 各走一遍正文（上下文两侧都前进、+ 只进新文件侧、
    - 只进旧文件侧），走出的行数必须等于头声明的 new_lines / old_lines，
    末行也必须落在 start + lines - 1（闭区间）。
    """
    diffs = load_diffs(blank_line_repo, "main", "feature")
    checked = 0

    for diff in diffs:
        for hunk in diff.hunks:
            checked += 1
            # content 只含正文（不含 @@ 行）；`\ No newline at end of file`
            # 是标记行，不占文件行号
            body = [ln for ln in hunk.content.splitlines() if not ln.startswith("\\")]

            new_no, old_no = hunk.new_start, hunk.old_start
            walked_new: list[int] = []
            walked_old: list[int] = []
            for line in body:
                if line.startswith("+"):
                    walked_new.append(new_no)
                    new_no += 1
                elif line.startswith("-"):
                    walked_old.append(old_no)
                    old_no += 1
                else:
                    walked_new.append(new_no)
                    walked_old.append(old_no)
                    new_no += 1
                    old_no += 1

            assert len(walked_new) == hunk.new_lines, f"{diff.path} 新文件侧行数不符"
            assert len(walked_old) == hunk.old_lines, f"{diff.path} 旧文件侧行数不符"
            if walked_new:
                assert walked_new[-1] == hunk.new_start + hunk.new_lines - 1
            if walked_old:
                assert walked_old[-1] == hunk.old_start + hunk.old_lines - 1

    assert checked >= 2, "样例应产生至少两个 hunk"


def test_load_diffs_uses_zero_context(blank_line_repo: Path) -> None:
    """测试 load_diffs 产出的是 -U0（零上下文）diff。

    `changed_lines` 直接读头字段，成立的前提就是 hunk 里只有改动行：
    正文行数必须等于 new_lines + old_lines，且每行只能是 + 或 -（外加
    `\\ No newline` 标记）。这条断言守住这个前提。
    """
    diffs = load_diffs(blank_line_repo, "main", "feature")
    hunks = [hunk for diff in diffs for hunk in diff.hunks]
    assert hunks

    for hunk in hunks:
        body = [ln for ln in hunk.content.splitlines() if not ln.startswith("\\")]
        assert len(body) == hunk.new_lines + hunk.old_lines, f"{hunk.content} 不是 -U0"
        for line in body:
            assert line[:1] in ("+", "-"), f"-U0 下不应出现上下文行：{line!r}"


def test_load_diffs_matches_git_cli_u0(blank_line_repo: Path) -> None:
    """测试 pygit2 取到的 -U0 diff 与 git 命令行的 `git diff -U0` 一致。

    pygit2 走的是 libgit2，与 git 命令行是两个实现。这条断言把
    `repo.diff(..., context_lines=0)` 的产物与 `git diff -U0` 对比：

    - 头字段（起始行 / 行数）—— 归属判定直接依赖这些数字
    - hunk 正文 —— 它是送进 LLM 审查的内容
    """
    cli_patch = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "-U0", "main", "feature"],
        cwd=blank_line_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert cli_patch, "git diff -U0 应产生输出"

    def shape(patch: str) -> dict[str, list[tuple[int, int, int, int, str]]]:
        return {
            diff.path: [
                (h.old_start, h.old_lines, h.new_start, h.new_lines, h.content) for h in diff.hunks
            ]
            for diff in parse_diff_text(patch)
        }

    ours = {
        diff.path: [
            (h.old_start, h.old_lines, h.new_start, h.new_lines, h.content) for h in diff.hunks
        ]
        for diff in load_diffs(blank_line_repo, "main", "feature")
    }
    assert ours == shape(cli_patch)


# ---------- build_diff_index ----------

# 「A 改名成 B」与「新建一个 A」同时发生 ——
# 索引里两个 A 会撞车，这个样本就是那场撞车
RENAME_AND_REUSE = """\
diff --git a/old.go b/new.go
similarity index 90%
rename from old.go
rename to new.go
--- a/old.go
+++ b/new.go
@@ -1 +1 @@
-package old
+package new
diff --git a/old.go b/old.go
new file mode 100644
--- /dev/null
+++ b/old.go
@@ -0,0 +1 @@
+package fresh
"""


def test_build_diff_index_keys_by_repo_path() -> None:
    """测试每个文件都能按自己的路径查到。"""
    diffs = parse_diff_text(
        "diff --git a/a.go b/a.go\n--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-x()\n+y()\n"
        "diff --git a/sub/b.go b/sub/b.go\n--- a/sub/b.go\n+++ b/sub/b.go\n"
        "@@ -1 +1 @@\n-p()\n+q()\n"
    )

    index = build_diff_index(diffs)

    assert sorted(index) == ["a.go", "sub/b.go"]
    assert index["sub/b.go"].hunks[0].content == "-p()\n+q()"


def test_build_diff_index_adds_old_path_alias_for_renames() -> None:
    """测试改名的文件按旧路径也查得到，且指向同一份改动。

    提示词里带着 `renamed-from="old.go"`，模型很可能拿旧名字来查。
    """
    (diff,) = parse_diff_text(
        "diff --git a/old.go b/new.go\n--- a/old.go\n+++ b/new.go\n@@ -1 +1 @@\n-a\n+b\n"
    )

    index = build_diff_index([diff])

    assert index["old.go"] is index["new.go"] is diff


def test_build_diff_index_lets_the_real_file_win_the_alias() -> None:
    """测试「改名 + 新建同名文件」时，查那个名字拿到的是**新建的文件**。

    回归用例：别名用直接赋值写的话，别名会把规范路径盖掉 —— 模型问 old.go
    拿到的是 new.go 的改动，一个不报错的错答案。
    """
    index = build_diff_index(parse_diff_text(RENAME_AND_REUSE))

    assert index["old.go"].is_new, "old.go 应该指新建的那个文件"
    assert index["new.go"].old_path == "old.go", "new.go 仍是那份改名 diff"


def test_build_diff_index_skips_dev_null() -> None:
    """测试新增文件不会在索引里留下 `/dev/null` 这个键。

    解析器用 `/dev/null` 表示「该侧不存在」，它不是一个可查询的路径。
    """
    diffs = parse_diff_text(
        "diff --git a/fresh.go b/fresh.go\nnew file mode 100644\n--- /dev/null\n"
        "+++ b/fresh.go\n@@ -0,0 +1 @@\n+package fresh\n"
    )

    index = build_diff_index(diffs)

    assert sorted(index) == ["fresh.go"]
