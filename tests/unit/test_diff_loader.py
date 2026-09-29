"""diff 加载的单元测试（用临时 git 仓库）。"""

from collections.abc import Callable
from pathlib import Path

import pygit2
import pytest

from prism.core.diff.loader import load_diffs, resolve_commit


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
            # `\ No newline at end of file` 是标记行，不占文件行号
            body = [ln for ln in hunk.content.splitlines()[1:] if not ln.startswith("\\")]

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


def test_blank_lines_survive_in_hunk_body(blank_line_repo: Path) -> None:
    """测试空源码行的上下文行（git 输出为单个空格）在正文里保留。

    回归用例：按 line.strip() 判断有效性会把它丢掉，导致 hunk.content
    里所有空源码行消失、上述不变量失配。
    """
    diffs = load_diffs(blank_line_repo, "main", "feature")
    bodies = [hunk.content for diff in diffs for hunk in diff.hunks]

    assert any("\n \n" in body for body in bodies), "正文里应有单空格上下文行"
    for body in bodies:
        for line in body.splitlines()[1:]:
            if not line.startswith("\\"):
                assert line[:1] in (" ", "+", "-"), f"非法正文行：{line!r}"
