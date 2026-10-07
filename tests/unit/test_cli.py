"""CLI 入口的单元测试。

两块：入口本身（版本号、帮助）与**全局预算**的循环 —— 后者用假协作者跑，
不碰 LLM 也不碰 git。
"""

import threading
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
from prism.models.comment import Comment
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

    outcome = main._run_groups(
        make_groups(4), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001", "group-002"], "第三组之前就该停"
    assert outcome.skipped_groups == 2
    assert (outcome.comments, outcome.misses, outcome.gaps) == ([], [], [])


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

    outcome = main._run_groups(
        make_groups(4), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001"], "只有算上反思那一笔，上限才会在第一组之后就响"
    assert outcome.skipped_groups == 3


def test_run_groups_stops_on_review_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """时间预算从命令开始算起 —— 超了就该一组都不跑。"""
    ran = install_fake_pipeline(monkeypatch, UsageTotals())
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(settings, review=replace(settings.review, max_review_seconds=60.0)),
    )

    outcome = main._run_groups(
        make_groups(3), tmp_path, "HEAD", {}, StaticReport(), time.monotonic() - 120.0
    )

    assert ran == []
    assert outcome.skipped_groups == 3


def test_run_groups_reports_total_when_nothing_was_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """没触发闸门时照常跑完，并打一行总计 —— **stdout 与日志各一份**。

    stdout 那份给人看，日志那份给 grep 的：排查「这条命令花了多久、多少钱」时
    不该要求人重定向 stdout。
    """
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=10))

    with caplog.at_level("INFO"):
        outcome = main._run_groups(
            make_groups(2), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
        )

    assert ran == ["group-001", "group-002"]
    assert outcome.skipped_groups == 0
    assert "总计" in capsys.readouterr().out
    assert "所有审查单元跑完：2/2 组" in caplog.text


def test_unfinished_hunks_are_counted_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """没审完的 hunk 要一路传到输出 —— 「0 条评论」不能看起来像「代码很干净」。"""
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=10))

    def incomplete(group: Group, *_args: object, **_kwargs: object) -> MainLoopResult:
        ran.append(group.group_id)
        return MainLoopResult(
            group_id=group.group_id,
            exit_reason="budget_cost",
            round_count=2,
            usage=UsageTotals(calls=1, output=10),
            pending_hunk_ids=["H-01"],
        )

    monkeypatch.setattr(main, "run_review_agent", incomplete)

    outcome = main._run_groups(
        make_groups(1), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert outcome.pending_hunks == 1
    assert outcome.skipped_groups == 0
    assert "未审 hunk=1" in capsys.readouterr().out


# ---------- 并发 ----------


def set_concurrency(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    """把并发度临时改掉。

    Args:
        monkeypatch: pytest 的 monkeypatch
        value: 并发度
    """
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(settings, review=replace(settings.review, concurrency=value)),
    )


def test_concurrency_actually_runs_groups_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发度 > 1 时，两组**真的同时在跑**。

    用 `Barrier(2)` 而不是掐时间：掐时间的用例在慢机器上会假红，而屏障是精确的 ——
    两组不同时进来，`wait` 就超时抛 `BrokenBarrierError`，那一组的结果会缺席。
    """
    barrier = threading.Barrier(2, timeout=3)
    reached: list[str] = []
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=1))

    def rendezvous(group: Group, *_args: object, **_kwargs: object) -> MainLoopResult:
        ran.append(group.group_id)
        try:
            barrier.wait()
            reached.append(group.group_id)
        except threading.BrokenBarrierError:
            raise AssertionError("两组没有同时进入 —— 并发没生效") from None
        return MainLoopResult(
            group_id=group.group_id,
            exit_reason="end_done",
            round_count=1,
            usage=UsageTotals(calls=1, output=1),
        )

    monkeypatch.setattr(main, "run_review_agent", rendezvous)
    set_concurrency(monkeypatch, 2)

    outcome = main._run_groups(
        make_groups(2), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert sorted(reached) == ["group-001", "group-002"]
    assert (outcome.skipped_groups, outcome.failed_groups) == (0, 0)


def test_concurrent_results_are_ordered_by_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**汇总按组序号，不按跑完的顺序** —— 输出顺序不能跟着调度变。

    让第 1 组故意慢一点（第 2 组先跑完），断言汇总顺序仍是 1、2。
    """
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=1))

    def slow_first(group: Group, *_args: object, **_kwargs: object) -> MainLoopResult:
        if group.group_id == "group-001":
            time.sleep(0.15)
        ran.append(group.group_id)
        return MainLoopResult(
            group_id=group.group_id,
            exit_reason="end_done",
            round_count=1,
            usage=UsageTotals(calls=1, output=1),
        )

    monkeypatch.setattr(main, "run_review_agent", slow_first)

    # 每条评论的正文带上它属于哪一组，这样断言顺序才有意义
    def tag(_model: object, group: Group, *_args: object, **_kwargs: object) -> ReflectionOutcome:
        return ReflectionOutcome(
            kept=[
                Comment(
                    path="a.go",
                    severity="low",
                    category="x",
                    content=group.group_id,
                    existing_code="x",
                    hunk_id="H-01",
                    locate_level="hunk",
                )
            ]
        )

    monkeypatch.setattr(main, "reflect", tag)
    set_concurrency(monkeypatch, 2)

    outcome = main._run_groups(
        make_groups(2), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-002", "group-001"], "第 2 组确实先跑完（前提）"
    assert [item.content for item in outcome.comments] == ["group-001", "group-002"], "汇总按组序"


def test_concurrency_keeps_every_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """并发跑完的组数与串行一致（并发不改变语义）。"""
    for concurrency in (1, 3):
        ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=1))
        set_concurrency(monkeypatch, concurrency)

        outcome = main._run_groups(
            make_groups(5), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
        )

        assert ran == [f"group-{index:03d}" for index in range(1, 6)], f"并发度 {concurrency}"
        assert (outcome.skipped_groups, outcome.failed_groups) == (0, 0)


def test_concurrent_budget_stops_submitting_new_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发下超预算：**已经在跑的不打断，只是不再提交新的**。

    所以并发度会决定「超预算时多花了多少」—— 最坏多跑 `concurrency - 1` 组。
    """
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=50_000))
    set_concurrency(monkeypatch, 2)
    monkeypatch.setattr(
        "prism.cli.main.settings",
        replace(
            settings,
            review=replace(settings.review, concurrency=2, max_review_cost=1.0),
        ),
    )

    outcome = main._run_groups(
        make_groups(6), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001", "group-002"], "前两组已提交、不打断；第三组起不再提交"
    assert (outcome.skipped_groups, outcome.failed_groups) == (4, 0)


def test_one_crashed_group_does_not_kill_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """编排层某一组抛异常：记 ERROR、跳过它，别的组照跑。

    它与「因预算没跑」是两回事（一个是钱不够，一个是出错了），所以**分开计数** ——
    合成一个数会让报告说错原因（同「丢弃日志说错原因，比丢东西本身更难发现」）。
    """
    ran = install_fake_pipeline(monkeypatch, UsageTotals(calls=1, output=1))

    def boom_on_second(group: Group, *_args: object, **_kwargs: object) -> MainLoopResult:
        ran.append(group.group_id)
        if group.group_id == "group-002":
            msg = "编排层炸了"
            raise RuntimeError(msg)
        return MainLoopResult(
            group_id=group.group_id,
            exit_reason="end_done",
            round_count=1,
            usage=UsageTotals(calls=1, output=1),
        )

    monkeypatch.setattr(main, "run_review_agent", boom_on_second)
    set_concurrency(monkeypatch, 3)

    outcome = main._run_groups(
        make_groups(3), tmp_path, "HEAD", {}, StaticReport(), time.monotonic()
    )

    assert ran == ["group-001", "group-002", "group-003"], "第三组照跑"
    assert outcome.failed_groups == 1
    assert outcome.skipped_groups == 0, "出错的组不能被算成「因预算没跑」"
