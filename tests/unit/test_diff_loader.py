"""diff 加载的单元测试（用临时 git 仓库）。"""

import subprocess
from pathlib import Path

import pygit2
import pytest

from prism.core.diff.loader import load_diffs, resolve_commit


def _git(repo: Path, *args: str) -> None:
    """在指定仓库内执行 git 命令。

    Args:
        repo: 仓库路径
        args: git 子命令及参数
    """
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """构造含 main / feature 两个分支的临时仓库。

    main:    a.go（a() 无输出）
    feature: a.go（a() 加输出）+ 新增 b.go

    Returns:
        仓库路径
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")

    (repo / "a.go").write_text("package main\n\nfunc a() {}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.go").write_text("package main\n\nfunc a() { println(1) }\n", encoding="utf-8")
    (repo / "b.go").write_text("package main\n\nfunc b() {}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feature")
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


def test_load_diffs_uses_merge_base(sample_repo: Path) -> None:
    """测试以 merge-base 为基准：base 分支上的新提交不进入 diff。"""
    _git(sample_repo, "checkout", "-q", "main")
    (sample_repo / "c.go").write_text("package main\n\nfunc c() {}\n", encoding="utf-8")
    _git(sample_repo, "add", ".")
    _git(sample_repo, "commit", "-q", "-m", "main only")

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
def edge_repo(tmp_path: Path) -> Path:
    """构造含二进制文件 / 删除文件 / 重命名文件的临时仓库。

    main:    a.go、del.go、old.go、logo.png
    feature: a.go 改动、del.go 删除、old.go 重命名 + 改内容、logo.png 二进制改动

    Returns:
        仓库路径
    """
    repo = tmp_path / "edge"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")

    (repo / "a.go").write_text("package main\n", encoding="utf-8")
    (repo / "del.go").write_text("old content\n", encoding="utf-8")
    # 5 行文件，重命名后只改 1 行：相似度 58%，git 会判为 R058 重命名
    (repo / "old.go").write_text("package main\n\nfunc old() {\n\treturn\n}\n", encoding="utf-8")
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.go").write_text("package main\n\nfunc a() {}\n", encoding="utf-8")
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 16)
    _git(repo, "rm", "-q", "del.go")
    _git(repo, "mv", "old.go", "renamed.go")
    (repo / "renamed.go").write_text(
        "package main\n\nfunc renamed() {\n\treturn\n}\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")
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
