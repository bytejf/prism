"""分组的单元测试。"""

from collections.abc import Callable
from pathlib import Path

import pytest

from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import ROOT_PACKAGE, group_by_package, package_path
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk


def make_hunk_with_symbol(
    file_path: str, *, hunk_id: str = "H-01", new_start: int = 1, old_start: int = 1
) -> HunkWithSymbol:
    """构造测试用 HunkWithSymbol。

    Args:
        file_path: 文件路径
        hunk_id: hunk 编号
        new_start: 新文件起始行
        old_start: 旧文件起始行

    Returns:
        HunkWithSymbol 对象
    """
    return HunkWithSymbol(
        part=Hunk(
            old_start=old_start,
            old_lines=1,
            new_start=new_start,
            new_lines=1,
            content="",
        ),
        file_path=file_path,
        symbol=Symbol(name="Run", kind="function", start_line=1, end_line=9),
        hunk_id=hunk_id,
    )


def test_group_by_package() -> None:
    """测试按包分组。"""
    hunks = [
        make_hunk_with_symbol("internal/agent/agent.go"),
        make_hunk_with_symbol("internal/agent/preview.go"),
        make_hunk_with_symbol("internal/diff/resolver.go"),
    ]
    groups = group_by_package(hunks)
    assert len(groups) == 2
    assert groups[0].package_path == "internal/agent"
    assert len(groups[0].files) == 2
    assert groups[1].package_path == "internal/diff"
    assert len(groups[1].files) == 1


def test_package_path_from_file_path() -> None:
    """测试包路径推导。"""
    assert package_path("internal/agent/agent.go") == "internal/agent"
    assert package_path("agent.go") == ROOT_PACKAGE
    assert package_path("a/b/c/d.go") == "a/b/c"


def test_root_files_form_their_own_group() -> None:
    """测试根目录下的文件单独成组。"""
    groups = group_by_package(
        [make_hunk_with_symbol("main.go"), make_hunk_with_symbol("internal/a.go")]
    )
    assert [g.package_path for g in groups] == [ROOT_PACKAGE, "internal"]


def test_group_ids_are_sequential_by_package_order() -> None:
    """测试分组 ID 按包路径升序连续分配。"""
    hunks = [
        make_hunk_with_symbol("internal/tool/registry.go"),
        make_hunk_with_symbol("internal/agent/agent.go"),
        make_hunk_with_symbol("internal/diff/resolver.go"),
    ]
    groups = group_by_package(hunks)
    assert [g.package_path for g in groups] == ["internal/agent", "internal/diff", "internal/tool"]
    assert [g.group_id for g in groups] == ["group-001", "group-002", "group-003"]


def test_hunks_sorted_by_file_then_line() -> None:
    """测试组内 hunk 按文件路径 + 行号升序。"""
    hunks = [
        make_hunk_with_symbol("pkg/b.go", hunk_id="H-01", new_start=50),
        make_hunk_with_symbol("pkg/a.go", hunk_id="H-02", new_start=90),
        make_hunk_with_symbol("pkg/a.go", hunk_id="H-03", new_start=10),
    ]
    groups = group_by_package(hunks)
    assert len(groups) == 1
    assert [(h.file_path, h.part.new_start) for h in groups[0].hunks] == [
        ("pkg/a.go", 10),
        ("pkg/a.go", 90),
        ("pkg/b.go", 50),
    ]


def test_files_are_deduplicated_and_sorted() -> None:
    """测试文件列表去重且升序。"""
    hunks = [
        make_hunk_with_symbol("pkg/b.go"),
        make_hunk_with_symbol("pkg/a.go"),
        make_hunk_with_symbol("pkg/b.go", hunk_id="H-02"),
    ]
    groups = group_by_package(hunks)
    assert groups[0].files == ["pkg/a.go", "pkg/b.go"]
    assert len(groups[0].hunks) == 3


def test_grouping_empty_input() -> None:
    """测试空输入。"""
    assert group_by_package([]) == []


# --- 真实仓库集成测试（审查前流程端到端） ---

PKG_A_GO = 'package alpha\n\nimport "fmt"\n\nfunc Run() {\n\tfmt.Println("a")\n}\n'
PKG_B_GO = 'package beta\n\nimport "fmt"\n\nfunc Run() {\n\tfmt.Println("b")\n}\n'


@pytest.fixture
def two_package_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造跨两个包的临时仓库，feature 分支在每个包里各改一个文件。

    Returns:
        仓库路径
    """
    repo = tmp_path / "multi"
    (repo / "internal/alpha").mkdir(parents=True)
    (repo / "internal/beta").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")

    (repo / "internal/alpha/a.go").write_text(PKG_A_GO, encoding="utf-8")
    (repo / "internal/beta/b.go").write_text(PKG_B_GO, encoding="utf-8")
    (repo / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "internal/alpha/a.go").write_text(
        PKG_A_GO.replace('fmt.Println("a")', 'fmt.Println("a")\n\tfmt.Println("a2")'),
        encoding="utf-8",
    )
    (repo / "internal/beta/b.go").write_text(
        PKG_B_GO.replace('fmt.Println("b")', 'fmt.Println("b")\n\tfmt.Println("b2")'),
        encoding="utf-8",
    )
    (repo / "main.go").write_text(
        "package main\n\nfunc main() {\n\tprintln(1)\n}\n", encoding="utf-8"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature")
    return repo


def test_pre_review_pipeline_groups_by_package(two_package_repo: Path) -> None:
    """测试审查前流程端到端：diff → 过滤 → hunk 映射 → 分组。"""
    reviewable = filter_files(load_diffs(two_package_repo, "main", "feature")).reviewable
    assert {d.path for d in reviewable} == {
        "internal/alpha/a.go",
        "internal/beta/b.go",
        "main.go",
    }

    mapped = map_hunks_to_symbols(two_package_repo, "main", "feature", reviewable)
    assert len(mapped) == 3

    groups = group_by_package(mapped)
    assert [g.package_path for g in groups] == [".", "internal/alpha", "internal/beta"]
    assert [g.group_id for g in groups] == ["group-001", "group-002", "group-003"]

    alpha = next(g for g in groups if g.package_path == "internal/alpha")
    assert alpha.files == ["internal/alpha/a.go"]
    assert alpha.hunks[0].symbol.name == "Run"

    root = next(g for g in groups if g.package_path == ".")
    assert root.files == ["main.go"]
