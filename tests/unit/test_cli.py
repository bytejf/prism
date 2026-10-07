"""CLI 入口的单元测试。

两块：入口本身（版本号、帮助）与**全局预算**的循环 —— 后者用假协作者跑，
不碰 LLM 也不碰 git。
"""

import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prism.cli import main
from prism.cli.main import app
from prism.config.settings import settings
from prism.core.agent.state import MainLoopResult
from prism.core.reflection.reflect import ReflectionOutcome
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.cost import UsageTotals
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.static import StaticReport

runner = CliRunner()


def test_version_flag() -> None:
    """测试 `prism --version` 输出包版本号。"""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"prism {version('prism')}"


def test_no_args_prints_help() -> None:
    """测试不带参数时打印帮助并正常退出。"""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Prism" in result.stdout


# ---------- 全局预算：停掉剩下的组 ----------


def make_groups(count: int) -> list[Group]:
    """造若干个只有一个 hunk 的审查单元。

    Args:
        count: 组数

    Returns:
        Group 列表
    """
    groups = []
    for index in range(1, count + 1):
        item = HunkWithSymbol(
            part=Hunk(old_start=1, old_lines=1, new_start=1, new_lines=1, content="+x"),
            file_path=f"pkg{index}/a.go",
            symbol=Symbol(name="F", kind="function", start_line=1, end_line=9),
            hunk_id=f"H-{index:02d}",
            head_lines=(1, 9),
            renamed_from=None,
            deleted=False,
        )
        groups.append(
            Group(
                group_id=f"group-{index:03d}",
                package_path=f"pkg{index}",
                files=[item.file_path],
                hunks=[item],
            )
        )
    return groups


def install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, usage: UsageTotals) -> list[str]:
    """把 `_run_groups` 的协作者换成假的，返回「跑过的组」清单。

    Args:
        monkeypatch: pytest 的 monkeypatch
        usage: 每组回报的用量

    Returns:
        空列表，会被就地填上跑过的组编号
    """
    ran: list[str] = []

    class _FakeLLM:
        model = None

    def fake_run(group: Group, *_args: object, **_kwargs: object) -> MainLoopResult:
        ran.append(group.group_id)
        return MainLoopResult(
            group_id=group.group_id, exit_reason="end_done", round_count=1, usage=usage
        )

    monkeypatch.setattr(main, "_llm_or_exit", lambda: _FakeLLM())
    monkeypatch.setattr(main, "run_review_agent", fake_run)
    monkeypatch.setattr(main, "locate_group", lambda _group, _comments: [])
    monkeypatch.setattr(
        main,
        "reflect",
        lambda *_args, **_kwargs: ReflectionOutcome(kept=[], usage=usage),
    )
    return ran


def test_run_groups_stops_when_review_cost_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """整条 review 的花费超了 → 停掉**剩下的**组，并把未审组数报出来。

    少审几个组与「那些组没问题」在输出上长得一模一样，所以未审数必须一路传到
    渲染层（`ReviewStats.skipped_groups`）。
    """
    # 一次调用 5 万输出 token ≈ 0.6 元；一组跑两笔（审查循环 + 反思）= 1.2 元。
    # 于是第二组能跑完、第三组之前超过 2 元的整条上限
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=50_000))
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(settings, review=replace(settings.review, max_review_cost=2.0)),
    )

    comments, misses, gaps, skipped = main._run_groups(
        make_groups(4), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001", "group-002"], "第三组之前就该停"
    assert skipped == 2
    assert (comments, misses, gaps) == ([], [], [])


def test_reflection_usage_counts_toward_the_review_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**反思的用量也要算进总账** —— 组级闸看不到它（那时循环已经结束）。

    上限卡在临界值上做差分：一次调用 5 万输出 token = 0.6 元，一组两笔（审查循环
    + 反思）= 1.2 元。把上限设成 1.0：

    - 算上反思 → 第一组就跑满，停
    - 漏掉反思 → 第一组只记 0.6，还要再跑一组才停

    少了这一笔，全局预算会系统性低估「每组一次反思调用」，组数越多偏得越狠。
    """
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=50_000))
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(settings, review=replace(settings.review, max_review_cost=1.0)),
    )

    _comments, _misses, _gaps, skipped = main._run_groups(
        make_groups(4), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001"], "只有算上反思那一笔，上限才会在第一组之后就响"
    assert skipped == 3


def test_run_groups_stops_on_review_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时间预算从命令开始算起 —— 超了就该一组都不跑。"""
    ran = install_fake_pipeline(monkeypatch, UsageTotals())
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(settings, review=replace(settings.review, max_review_seconds=60.0)),
    )

    _comments, _misses, _gaps, skipped = main._run_groups(
        make_groups(3), tmp_path, "HEAD", {}, StaticReport(), time.monotonic() - 120.0
    )

    assert ran == []
    assert skipped == 3


def test_run_groups_reports_total_when_nothing_was_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """没触发闸门时照常跑完，并打一行总计。"""
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=10))

    _comments, _misses, _gaps, skipped = main._run_groups(
        make_groups(2), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001", "group-002"]
    assert skipped == 0
    assert "总计" in capsys.readouterr().out
