"""分组的单元测试。

两块：**按包分组**（既有契约）与**按预算装箱**（设计 18 号 §2.4 —— 一个包装不下
时拆成连续的几个组，静态发现也占额度）。
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from prism.config.settings import settings
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping import packages
from prism.core.grouping.packages import (
    ROOT_PACKAGE,
    fixed_prompt_tokens,
    group_by_package,
    group_tokens,
    package_path,
)
from prism.core.guidance.builder import build_diff_block, build_static_block
from prism.core.guidance.tokens import estimate_tokens
from prism.core.static.analyzer import select_for_group
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.static import PRIORITY_HIGH, StaticFinding


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
        head_lines=(1, 9),
        renamed_from=None,
        deleted=False,
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


# ---------- token 预算与装箱 ----------


def budget(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    """把单组预算临时改成指定值。

    模块里读的是 `settings.review.max_group_tokens`（每次调用现取），所以把模块
    上的名字换掉即可 —— `Settings` 是 frozen 的，改不了实例属性。

    Args:
        monkeypatch: pytest 的 monkeypatch
        tokens: 新的单组预算
    """
    monkeypatch.setattr(
        packages,
        "settings",
        replace(settings, review=replace(settings.review, max_group_tokens=tokens)),
    )


def pad_hunk(hunk: HunkWithSymbol, lines: int) -> HunkWithSymbol:
    """给一个 hunk 灌入指定行数的正文（用来把它的成本撑到可测的大小）。

    Args:
        hunk: 原 hunk
        lines: 正文行数

    Returns:
        新的 HunkWithSymbol
    """
    content = "\n".join(
        f"+// 填充行 {index}，用来把这一片的 token 撑起来" for index in range(lines)
    )
    part = replace(hunk.part, new_lines=lines, content=content)
    return replace(hunk, part=part)


def test_small_packages_are_not_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """预算只用来向下切，不用来向上并 —— 小包不合并。

    合并会更省（固定开销 1111 token/组），但会让 `package_path` 不再是单值，
    也让每个组的主题变糊。「一个审查单元 = 一个包」是既有契约。
    """
    budget(monkeypatch, 10**6)
    groups = group_by_package(
        [
            make_hunk_with_symbol("a/a.go"),
            make_hunk_with_symbol("b/b.go"),
            make_hunk_with_symbol("c/c.go"),
        ]
    )
    assert [item.package_path for item in groups] == ["a", "b", "c"]


def test_oversized_package_splits_into_consecutive_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个包装不下就拆成若干个连续的组，hunk 一个不丢、顺序不变。"""
    hunks = [
        pad_hunk(make_hunk_with_symbol("big/f.go", hunk_id=f"H-{index:02d}"), 40)
        for index in range(1, 7)
    ]
    budget(monkeypatch, 1500)

    groups = group_by_package(hunks)

    assert len(groups) > 1, "这个包必须被拆开，否则这条用例没测到切分"
    assert {item.package_path for item in groups} == {"big"}
    assert [item.hunk_id for group in groups for item in group.hunks] == [
        f"H-{index:02d}" for index in range(1, 7)
    ]
    assert [group.group_id for group in groups] == [
        f"group-{index:03d}" for index in range(1, len(groups) + 1)
    ]


def test_static_findings_count_toward_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """静态发现参与预算：同样的 hunk，带上发现之后会更早被切开。

    这就是设计 §2.4 那句「分组算法要把静态发现的 token 也计入预算」的可执行版本。
    """
    hunks = [
        pad_hunk(make_hunk_with_symbol("pkg/f.go", hunk_id=f"H-{index:02d}"), 30)
        for index in range(1, 5)
    ]
    # 预算要够装下两三个 hunk，否则每条都会走「单个 hunk 自己也超」那条路，
    # 切分点与发现无关，这条用例就测不到东西
    budget(monkeypatch, 2000)
    without = group_by_package(hunks)

    findings = [
        StaticFinding(
            finding_id=f"S-{index:02d}",
            path="pkg/f.go",
            line=index,
            linter="gosec",
            priority=PRIORITY_HIGH,
            text="G107: " + "x" * 300,
            hunk_id=f"H-{index:02d}",
        )
        for index in range(1, 5)
    ]
    with_findings = group_by_package(hunks, findings)

    assert len(with_findings) > len(without), "发现占的额度必须影响切分点"


def test_huge_file_is_split_hunk_by_hunk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """单个文件自己就超预算 → 按 hunk 切，并有 WARNING。

    这是唯一会拆散同一个文件的路径：一个文件改了几百行时，不拆就装不下。
    **断言的是行为（真的拆成了多组）而不是那句 WARNING 文案** —— 只断言日志的话，
    把切分改掉、日志留着，用例照样绿（变异测出来的）。
    """
    hunks = [
        pad_hunk(make_hunk_with_symbol("big/gen.go", hunk_id=f"H-{index:02d}"), 55)
        for index in range(1, 4)
    ]
    per_hunk = fixed_prompt_tokens() + group_tokens(hunks[:1], [])
    whole_file = fixed_prompt_tokens() + group_tokens(hunks, [])
    budget(monkeypatch, (per_hunk + whole_file) // 2)
    assert per_hunk <= whole_file

    with caplog.at_level("WARNING"):
        groups = group_by_package(hunks)

    assert "按 hunk 切成" in caplog.text
    assert len(groups) > 1, "文件放不下时必须拆开，否则它就是一个超预算的组"
    assert [item.hunk_id for group in groups for item in group.hunks] == ["H-01", "H-02", "H-03"]


def test_single_hunk_over_budget_forms_its_own_group(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """一个 hunk 自己就超预算时它单独成组 —— 到那一层再没有能切的东西了。"""
    hunks = [pad_hunk(make_hunk_with_symbol("pkg/f.go", hunk_id="H-01"), 400)]
    budget(monkeypatch, 100)

    with caplog.at_level("WARNING"):
        groups = group_by_package(hunks)

    assert len(groups) == 1
    assert "装不下这一片" in caplog.text


def test_every_hunk_lands_in_exactly_one_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """切分不重不漏：所有 hunk 恰好出现一次，且组内顺序单调。"""
    hunks = [
        pad_hunk(make_hunk_with_symbol(f"pkg{index % 3}/f.go", hunk_id=f"H-{index:02d}"), 25)
        for index in range(1, 13)
    ]
    budget(monkeypatch, 1200)

    groups = group_by_package(hunks)
    seen = [item.hunk_id for group in groups for item in group.hunks]

    assert sorted(seen) == sorted(item.hunk_id for item in hunks)
    assert len(seen) == len(set(seen))
    for group in groups:
        assert [item.file_path for item in group.hunks] == sorted(
            item.file_path for item in group.hunks
        )


def test_group_tokens_matches_the_rendered_prompt() -> None:
    """成本量的是真实渲染结果 —— 与真正发出去的提示词同一条代码路径。

    这条守的是「成本模型与实际渲染漂开」这一类 bug：量出来的数必须等于
    `build_diff_block` + `build_static_block` 渲染后的估算值。
    """
    hunks = [pad_hunk(make_hunk_with_symbol("pkg/f.go", hunk_id="H-01"), 10)]
    findings = [
        StaticFinding(
            finding_id="S-01",
            path="pkg/f.go",
            line=1,
            linter="gosec",
            priority=PRIORITY_HIGH,
            text="G107: x",
            hunk_id="H-01",
        )
    ]
    related = select_for_group(["pkg/f.go"], findings)
    group = Group(group_id="group-001", package_path="pkg", files=["pkg/f.go"], hunks=hunks)

    expected = estimate_tokens(build_diff_block(group, related)) + estimate_tokens(
        build_static_block(related)
    )
    assert group_tokens(hunks, findings) == expected


def test_no_group_exceeds_the_budget_when_splitting_is_possible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """切得开的时候，**没有一组该超预算**（固定开销必须算进每一次判断）。

    这条是踩出来的：文件级的判断原来只比「内容 ≤ 预算」，装箱时又比「固定开销 +
    内容 ≤ 预算」，两个口径不一致 → 干跑输出里出现带着 ⚠️ 的组，而那个文件明明
    可以拆成两组。`prism group` 的 ⚠️ 标记就是在真实仓库上把它暴露出来的。
    """
    hunks = [
        pad_hunk(make_hunk_with_symbol("pkg/f.go", hunk_id=f"H-{index:02d}"), 55)
        for index in range(1, 4)
    ]
    # 期望值**不能**用 group_cost 自己算 —— 那正是被测的那个函数，它一变期望值
    # 跟着变，变异就永远抓不住（这条用例的第一版就是这么漏的）
    one = fixed_prompt_tokens() + group_tokens(hunks[:1], [])
    all_three = fixed_prompt_tokens() + group_tokens(hunks, [])
    monkey_budget = (one + all_three) // 2
    budget(monkeypatch, monkey_budget)
    # 前提：单个放得下、三个放不下 —— 否则这条用例测不到「该拆而没拆」
    assert one <= monkey_budget < all_three

    groups = group_by_package(hunks)

    assert len(groups) > 1
    for group in groups:
        assert fixed_prompt_tokens() + group_tokens(group.hunks, []) <= monkey_budget
