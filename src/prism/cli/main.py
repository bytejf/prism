"""Prism CLI 入口。

本模块是全部可执行命令的唯一入口（`[project.scripts] prism = "prism.cli.main:app"`）。
下面是命令总览；每条命令的详细参数见各自的 `--help`，实现见本文件内的同名函数。

## 全局

    prism --version                 显示版本号并退出
    prism --help / prism            打印帮助（不带子命令时）

    **所有子命令都必须用 `--path` 指定被审查的项目**：

        prism group --path ~/Projects/foo

    `--path` 必须指向一个 git 仓库，以该仓库的 merge-base(base, head) 为基准
    比较，所以只统计 PR 实际引入的改动 —— base 分支上的新提交不会混进来。

    V1 刻意**不做「默认当前目录」**：审查是有成本的破坏性操作（花 token、
    发代码给外部 API），少敲一个参数就审错仓库的代价，比每次多敲 `--path` 大。

## 审查前流程（确定性，不调用 LLM、零 token 成本）

纯读取命令，用于在花钱调 LLM 之前确认「改了什么、审哪些、怎么分组」。
五个命令依次是流水线的下一个环节，可以逐个跑、逐个核对。

    prism diff      --path <repo> [--base main] [--head HEAD]
        加载并解析 diff，打印文件数 / hunk 数 / 增删行数。
        对应 core/diff/：pygit2 算 merge-base + 状态机解析 unified diff。

    prism filter    --path <repo> [--base main] [--head HEAD]
        按规则过滤文件，打印可审查 / 跳过清单（含跳过原因）。
        三级过滤：硬性排除（二进制 / 符号链接）→ 扩展名白名单（.go）
        → 内置降噪（*_test.go、*.pb.go）。对应 core/filter/。

    prism hunk-map  --path <repo> [--base main] [--head HEAD]
        把每个 hunk 归属到包含其改动行的符号，打印 hunk 位置与归属符号。
        只处理通过过滤的 .go 文件。对应 core/ast/：tree-sitter 提取符号
        （函数 / 方法 / 包声明 / import / const / var / type），按改动行求交。

    prism static    --path <repo> [--base main] [--head HEAD]
        跑一次 golangci-lint，打印静态发现清单（编号 / 档位 / 位置 / 关联 hunk）。
        只留可审查文件上的发现。对应 core/static/：检出 head → 跑工具 → 解析 JSON。
        **依赖外部二进制** `golangci-lint`（未安装时明确说「跳过」而不是「没发现」）。

    prism group     --path <repo> [--base main] [--head HEAD]
        按包路径（目录）分组，打印审查单元。
        这是审查前流程的最后一个环节，输出即为送进 LLM 的「审查单元」。
        对应 core/grouping/。

## 完整审查（调用 LLM，产生 token 成本）

    prism review    --path <repo> [--base main] [--head HEAD]
        串起完整链路：diff → 过滤 → hunk 映射 → 静态分析 → 分组 → LLM 审查
        → 定位 → 反思 → 排序 → 终端渲染。

        审查前三个阶段的进度打到 stdout，详细轨迹走 logging（stderr）：
        想看完整轨迹设 `PRISM_LOG_LEVEL=DEBUG`，想只看结果设 `PRISM_LOG_LEVEL=WARNING`。

        需要环境变量 `DEEPSEEK_API_KEY`，缺失时以退出码 1 明确报错。

        退出码：0 正常（含 0 条评论）；1 参数 / 环境 / git 错误。

## 典型用法

    # 先干跑一遍看清范围（不花钱），确认后再真审
    prism group  --path ~/Projects/foo --base main --head origin/some-branch
    prism review --path ~/Projects/foo --base main --head origin/some-branch

    # 排查「为什么某个文件没被审」
    prism filter --path ~/Projects/foo --base main --head origin/some-branch

    # 排查「某个 hunk 归属不准」
    prism hunk-map --path ~/Projects/foo --base main --head origin/some-branch

注意：`review` 的 `--head` 是**被审查的版本**，工具读源码时读的就是它，
不是当前工作区。所以工作区停在别的分支也能正确审查
（见 core/agent/context.py 的 head_rev）。
"""

import logging
import time
from collections.abc import Mapping
from importlib.metadata import version
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer

from prism.config.settings import settings
from prism.core.agent.agent import run_review_agent
from prism.core.agent.budget import add_usage, describe, exceeded
from prism.core.ast.mapper import format_location, format_symbol, map_hunks_to_symbols
from prism.core.diff.loader import build_diff_index, load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import (
    fixed_prompt_tokens,
    group_by_package,
    group_tokens,
    package_label,
)
from prism.core.llm.client import LLMClient
from prism.core.output.locate import locate_group
from prism.core.output.renderer import TerminalRenderer
from prism.core.output.sort import sort_comments
from prism.core.reflection.reflect import reflect
from prism.core.static.analyzer import analyze, format_finding, select_for_group
from prism.core.static.verify import verify_group
from prism.models.ast import HunkWithSymbol
from prism.models.comment import LOCATE_FAILED, Comment, ReviewStats
from prism.models.cost import UsageTotals
from prism.models.diff import Diff
from prism.models.grouping import Group
from prism.models.reflection import PotentialMiss
from prism.models.static import StaticFinding, StaticReport
from prism.observability.logging import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(help="Prism — 确定性优先的 Go 代码审查 Agent")

# 每个子命令都要声明它 —— 抽成别名，免得五处各写一遍还写歪
RepoPath = Annotated[
    Path | None,
    typer.Option("--path", help="被审查的项目路径，须是 git 仓库"),
]


def _resolve_repo(path: Path | None) -> Path:
    """校验 `--path` 并返回仓库根目录。

    没给路径就直接报错退出，**不回落当前目录**：审查会花 token、会把代码发给
    外部 API，少敲一个参数就审错仓库的代价比多敲一个大。

    Args:
        path: `--path` 传入的路径

    Returns:
        仓库根目录的绝对路径

    Raises:
        typer.Exit: 未指定路径，或该路径不是 git 仓库
    """
    if path is None:
        logger.error("缺少 --path")
        typer.echo("错误：必须用 --path 指定被审查的项目路径（例如 --path .）", err=True)
        raise typer.Exit(code=1)

    root = path.expanduser().resolve()
    # .git 可能是目录（普通仓库）也可能是文件（worktree / 子模块），exists 两者都覆盖
    if not (root / ".git").exists():
        logger.error("--path 不是 git 仓库: %s", root)
        typer.echo(f"错误：{root} 不是 git 仓库", err=True)
        raise typer.Exit(code=1)

    logger.debug("被审查仓库: %s", root)
    return root


def _version_callback(value: bool) -> None:
    """处理 `--version` 参数。

    eager 选项在回调函数体之前执行，因此这里需要自行初始化日志，
    否则 DEBUG 级别的日志无法输出。

    Args:
        value: 是否传入了 `--version`
    """
    if not value:
        return
    setup_logging()
    pkg_version = version("prism")
    logger.debug("读取包版本: %s", pkg_version)
    typer.echo(f"prism {pkg_version}")
    raise typer.Exit


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="显示版本号并退出",
    ),
) -> None:
    """Prism CLI 主入口。

    初始化日志；未指定子命令时打印帮助并正常退出。

    Args:
        ctx: Typer 上下文，用于判断是否指定了子命令
        _version: 是否传入 `--version`；值本身不读，全部由 `_version_callback`
            处理（`is_eager` 保证它在函数体之前执行）。下划线前缀表明这是
            「只为声明 CLI 参数而存在」的形参。
    """
    setup_logging()
    if ctx.invoked_subcommand is None:
        logger.debug("未指定子命令：打印帮助")
        typer.echo(ctx.get_help())
        raise typer.Exit


def _load_or_exit(root: Path, base: str, head: str) -> list[Diff]:
    """加载 diff，失败时打印错误并终止进程。

    Args:
        root: 被审查仓库的根目录
        base: base 分支 / 版本
        head: head 分支 / 版本

    Returns:
        Diff 列表
    """
    try:
        return load_diffs(root, base, head)
    except ValueError as exc:
        logger.error("diff 加载失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc


def _plural(count: int, word: str) -> str:
    """渲染带单复数的计数文本。

    Args:
        count: 数量
        word: 单数名词，如 "file"

    Returns:
        ``1 file`` 或 ``3 files``
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _map_or_exit(root: Path, base: str, head: str) -> list[HunkWithSymbol]:
    """跑完审查前流程（diff → 过滤 → hunk 映射），失败时终止进程。

    Args:
        root: 被审查仓库的根目录
        base: base 分支 / 版本
        head: head 分支 / 版本

    Returns:
        带归属符号的 hunk 列表
    """
    reviewable = filter_files(_load_or_exit(root, base, head)).reviewable
    try:
        return map_hunks_to_symbols(root, base, head, reviewable)
    except ValueError as exc:
        logger.error("hunk 映射失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def diff(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """加载并解析 diff，打印结构摘要（不调用 LLM）。

    以 merge-base(base, head) 为基准，只统计 PR 实际引入的改动。
    """
    root = _resolve_repo(path)
    logger.info("加载 diff: root=%s base=%s head=%s", root, base, head)
    diffs = _load_or_exit(root, base, head)

    typer.echo(f"Loaded {len(diffs)} files, {sum(len(d.hunks) for d in diffs)} hunks")
    for item in diffs:
        typer.echo(f"  {item.path}: {len(item.hunks)} hunks (+{item.insertions}/-{item.deletions})")


@app.command("filter")
def filter_cmd(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """按规则过滤文件，打印可审查 / 跳过清单（不调用 LLM）。

    函数名带 ``_cmd`` 后缀：命令名由 ``@app.command("filter")`` 显式给出，
    避免模块里出现一个叫 ``filter`` 的名字遮蔽内置函数。
    """
    root = _resolve_repo(path)
    logger.info("过滤文件: root=%s base=%s head=%s", root, base, head)
    diffs = _load_or_exit(root, base, head)
    result = filter_files(diffs)

    typer.echo(f"Reviewable: {_plural(len(result.reviewable), 'file')}")
    for item in result.reviewable:
        typer.echo(f"  {item.path}")

    typer.echo("")
    typer.echo(f"Skipped: {_plural(len(result.skipped), 'file')}")
    for skipped in result.skipped:
        typer.echo(f"  {skipped.path} ({skipped.reason})")


@app.command()
def hunk_map(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """把每个 hunk 归属到符号，打印映射结果（不调用 LLM）。

    只处理通过文件过滤的 .go 文件。
    """
    root = _resolve_repo(path)
    logger.info("映射 hunk: root=%s base=%s head=%s", root, base, head)
    mapped = _map_or_exit(root, base, head)

    if not mapped:
        typer.echo("没有可映射的 hunk")
        return

    blocks = [f"{format_location(item)}\n  归属：{format_symbol(item.symbol)}" for item in mapped]
    typer.echo("\n\n".join(blocks))


@app.command()
def group(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """按包路径分组并按预算装箱，打印审查单元（静态分析要跑，但不调用 LLM）。

    这是审查前流程的最后一个环节：输出即为送入 LLM 的「审查单元」。
    静态分析也在这里跑一次，因为**静态发现参与装箱预算** —— 不跑它，
    干跑出来的分组与真审时的不一样，那就失去核对的意义了。
    """
    root = _resolve_repo(path)
    logger.info("分组: root=%s base=%s head=%s", root, base, head)
    mapped = _map_or_exit(root, base, head)
    report = analyze(root, head, sorted({item.file_path for item in mapped}), mapped)
    _echo_static(report)
    groups = group_by_package(mapped, report.findings)

    if not groups:
        typer.echo("没有可审查的 hunk")
        return

    budget = settings.review.max_group_tokens
    blocks = []
    for index, item in enumerate(groups, start=1):
        tokens = group_tokens(item.hunks, report.findings)
        total = tokens + fixed_prompt_tokens()
        over = "  ⚠️ 超过预算" if total > budget else ""
        header = (
            f"Group {index}: {package_label(item)} "
            f"({_plural(len(item.files), 'file')}, {_plural(len(item.hunks), 'hunk')}, "
            f"约 {total}/{budget} token){over}"
        )
        lines = [header]
        # hunks 已按文件路径排序，同文件的连续 → groupby 一次即可
        # 变量叫 file_path 不叫 path：后者是 `--path` 参数的名字，遮蔽它两边都读不懂
        for file_path, hunks in groupby(item.hunks, key=lambda h: h.file_path):
            hunk_ids = ", ".join(h.hunk_id for h in hunks)
            lines.append(f"  - {PurePosixPath(file_path).name} ({hunk_ids})")
        blocks.append("\n".join(lines))

    typer.echo("\n\n".join(blocks))


def _echo_static(report: StaticReport) -> None:
    """把静态分析的结果打一行到 stdout。

    **「跳过」与「没有发现」必须分开说**：合成一句「没有静态发现」会让「没装
    golangci-lint」看起来像「代码很干净」—— 那正是一条不报错的错链路
    （同 `file_read_diff` 的索引为空要报错）。

    Args:
        report: 静态分析结果
    """
    if report.skipped:
        typer.echo(f"  静态分析跳过：{report.skip_reason}")
        return
    typer.echo(f"  静态发现 {len(report.findings)} 条（耗时 {report.elapsed_seconds:.1f}s）")


@app.command("static")
def static_cmd(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """跑一次静态分析，打印发现清单（不调用 LLM）。

    函数名带 ``_cmd`` 后缀：命令名由 ``@app.command("static")`` 显式给出，
    避免模块里的 ``static`` 与「静态」这个词的其他用法撞在一起。
    """
    root = _resolve_repo(path)
    logger.info("静态分析: root=%s base=%s head=%s", root, base, head)
    mapped = _map_or_exit(root, base, head)
    reviewable = sorted({item.file_path for item in mapped})

    report = analyze(root, head, reviewable, mapped)
    if report.skipped:
        # 「跳过」与「没发现」是两件事，CLI 上必须分开说 —— 合成一句话，
        # 用户会把「没装工具」读成「代码很干净」
        typer.echo(f"静态分析跳过：{report.skip_reason}")
        return
    if not report.findings:
        typer.echo(f"没有发现（本组可审查文件 {_plural(len(reviewable), 'file')}）")
        return

    for item in report.findings:
        typer.echo(format_finding(item))
    typer.echo("")
    typer.echo(
        f"共 {_plural(len(report.findings), 'finding')}（耗时 {report.elapsed_seconds:.1f}s）"
    )


def _llm_or_exit() -> LLMClient:
    """构造 LLM 客户端，失败时打印错误并终止进程。

    全部配置来自 `config/default.toml` + 环境变量里的 api_key，
    本函数不再自己拼 LLMConfig。

    Returns:
        LLMClient 实例
    """
    try:
        return LLMClient(settings.llm)
    except ValueError as exc:
        logger.error("LLM 初始化失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_groups(
    groups: list[Group],
    project_root: Path,
    head_rev: str,
    diff_index: Mapping[str, Diff],
    static_report: StaticReport,
    started: float,
) -> tuple[list[Comment], list[PotentialMiss], list[StaticFinding], int]:
    """串行跑完所有审查单元（V1 不做并发），每组跑完立刻定位与反思。

    定位与反思都放在循环体里而不是等所有组跑完：设计文档的节奏就是
    「每组审查完成 → 定位 → 反思」，而且反思是**每组一次 LLM 调用**，
    它要的素材（本组 diff + 本组评论 + 定位元数据）只有在这里才齐。

    **每组只用一次模型调用完成复核**（不是每个文件一次），失败时原样保留
    全部评论 —— 见 `core/reflection/reflect.py`。

    静态发现的**挑选在这里做一次**（挑本组文件），注入与后置校验共用同一份结果。

    **全局预算在这里执行**：每开一组之前先看总账（钱 + 时间），超了就停掉剩下的组。
    组级预算管的是「一组跑飞」，全局预算管的是「组数太多或前面几组太贵」—— 两层
    各管各的，缺一层就有一头漏。

    Args:
        groups: 审查单元列表
        project_root: 项目根目录
        head_rev: 被审查的版本，传给工具读源码用
        diff_index: `路径 → Diff` 索引，每组都要传 —— 它是**全量**的，
            组内组外都有，`file_read_diff` 靠它看别的包改了什么
        static_report: 静态分析结果（可能是被跳过的那份）
        started: 整条命令的开始时刻（`time.monotonic()`）。全局的时间预算从它算，
            于是「20 分钟」读起来就是「这条命令最长跑多久」，含静态分析

    Returns:
        (保留的评论, 疑似漏报信号, 未被引用的高优先级静态发现, 因预算未跑的组数)。
        评论含未定位的那些（带 locate_level 标记）
    """
    llm = _llm_or_exit()
    comments: list[Comment] = []
    misses: list[PotentialMiss] = []
    gaps: list[StaticFinding] = []
    total = UsageTotals()
    skipped = 0
    for index, item in enumerate(groups, start=1):
        # 开新组之前先看总账：超了就停掉剩下的组（**不是**只停这一组 ——
        # 全局闸的意思就是「整条别再花下去了」）
        stop = exceeded(
            total,
            time.monotonic() - started,
            prices=settings.cost,
            max_cost=settings.review.max_review_cost,
            max_seconds=settings.review.max_review_seconds,
        )
        if stop is not None:
            skipped = len(groups) - index + 1
            logger.warning("整条 review 超预算（%s），停掉剩余 %d 组", stop, skipped)
            typer.echo(
                f"⚠️ 整条 review 超预算（{stop}）：累计 {describe(total, settings.cost)}，"
                f"已用 {time.monotonic() - started:.0f} 秒 —— 剩余 {skipped} 组不再审查"
            )
            break

        typer.echo(
            f"Processing Group {index}: {package_label(item)} "
            f"({_plural(len(item.files), 'file')}, {_plural(len(item.hunks), 'hunk')})"
        )
        # 挑选不做二次裁剪：**预算在分组那一步就执行完了**（装不下的 hunk 顺延到
        # 下一组），这里再裁一次只会让「注入的」与「算过成本的」变成两批东西
        injected = select_for_group(item.files, static_report.findings)
        # 压缩配置直接用 settings 里那一份：它的 context_window 已经是从
        # [llm] 拼过来的，这里不再自己算一遍
        result = run_review_agent(
            item,
            llm.model,
            str(project_root),
            head_rev,
            settings.compression,
            diff_index,
            injected,
        )
        located = locate_group(item, result.hunk_comments)
        outcome = reflect(llm.model, item, located)
        # 反思的用量也要进总账：组级闸看不到它（那时循环已经结束了），而它每组一次
        group_usage = add_usage(result.usage, outcome.usage)
        total = add_usage(total, group_usage)
        comments.extend(outcome.kept)
        misses.extend(outcome.misses)
        verdict = verify_group(injected, outcome.kept)
        gaps.extend(verdict.unhandled_high)
        # 未定位数是**定位**这一步的统计，与反思删掉了几条无关，所以按
        # 定位结果算
        unlocated = sum(1 for comment in located if comment.locate_level == LOCATE_FAILED)
        static_note = (
            f", 静态发现={len(injected)}(引用{len(verdict.referenced)})" if injected else ""
        )
        typer.echo(
            f"  {result.group_id} completed: {_plural(len(outcome.kept), 'comment')} "
            f"(exit_reason={result.exit_reason}, 轮次={result.round_count}, "
            f"未定位={unlocated}, 反思删除={len(outcome.removed)}{static_note})"
        )
        typer.echo(f"    本组 {describe(group_usage, settings.cost)}")
    if not skipped:
        typer.echo(f"总计 {describe(total, settings.cost)}")
    return comments, misses, gaps, skipped


@app.command()
def review(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
) -> None:
    """完整审查：diff → 过滤 → hunk 映射 → 静态分析 → 分组 → LLM 审查 → 排序 → 渲染。"""
    started = time.monotonic()
    root = _resolve_repo(path)
    logger.info("开始审查: root=%s base=%s head=%s", root, base, head)

    typer.echo("Loading diffs...")
    diffs = _load_or_exit(root, base, head)

    typer.echo(f"Filtering files... ({_plural(len(diffs), 'file')} in diff)")
    reviewable = filter_files(diffs).reviewable

    # 索引在分组**之前**建、且建在**过滤之后**：filter 的产出既决定了哪些文件
    # 可审，也决定了 file_read_diff 能看到什么。分组不参与 —— 跨组查询要的正是
    # 别的组的改动，按组建索引就等于没建
    diff_index = build_diff_index(reviewable)

    try:
        mapped = map_hunks_to_symbols(root, base, head, reviewable)
    except ValueError as exc:
        logger.error("hunk 映射失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    # 静态分析夹在 hunk 映射与分组之间（设计 18 号 §2.3）：它的发现要按文件分桶进
    # 各组，而「关联到哪个 hunk」需要映射结果。可审查文件的路径直接取自映射结果 ——
    # 一个文件过了过滤但一个 hunk 都没活下来时，它本来也不会进任何组
    typer.echo(f"Running static analysis... ({_plural(len(reviewable), 'reviewable file')})")
    static_report = analyze(root, head, sorted({item.file_path for item in mapped}), mapped)
    _echo_static(static_report)

    typer.echo(f"Grouping by package... ({_plural(len(reviewable), 'reviewable file')})")
    groups = group_by_package(mapped, static_report.findings)

    if not groups:
        typer.echo("没有可审查的 hunk。")

    # 没有分组时不去构造 LLM 客户端：`prism review` 在空 diff 上不该要求密钥
    if groups:
        comments, misses, gaps, skipped_groups = _run_groups(
            groups, root, head, diff_index, static_report, started
        )
    else:
        comments, misses, gaps, skipped_groups = [], [], [], 0
    comments = sort_comments(comments)

    stats = ReviewStats(
        base_branch=base,
        head_sha=head,
        files_count=len(reviewable),
        comments_count=len(comments),
        duration_seconds=time.monotonic() - started,
        skipped_groups=skipped_groups,
    )
    typer.echo("")
    typer.echo(TerminalRenderer().render(comments, stats, misses, gaps))
    typer.echo(f"\nReview completed: {_plural(len(comments), 'comment')}")
