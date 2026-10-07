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
        想看完整的 LLM 轨迹设 `PRISM_LOG_LEVEL=INFO`（默认，轨迹本身就在 INFO），
        想只看结果设 `WARNING`，想要内部细节（符号表、装箱 token、提示词大小）设
        `DEBUG`。

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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer
from opentelemetry import trace

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
from prism.core.output.report import build_report, write_report
from prism.core.output.sort import sort_comments
from prism.core.reflection.reflect import reflect
from prism.core.static.analyzer import analyze, format_finding, select_for_group
from prism.core.static.verify import verify_group
from prism.models.ast import HunkWithSymbol
from prism.models.comment import LOCATE_FAILED, Comment, ReviewStats
from prism.models.cost import CostConfig, UsageTotals
from prism.models.diff import Diff
from prism.models.grouping import Group
from prism.models.reflection import PotentialMiss
from prism.models.static import StaticFinding, StaticReport
from prism.observability import semconv as sc
from prism.observability.logging import setup_logging
from prism.observability.tracing import parse_trace_attrs, set_attributes, setup_tracing

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

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
    # 追踪没配就完全不初始化：OTel 默认 tracer 是 no-op，埋点处不需要分支
    setup_tracing()
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

    typer.echo(f"文件 {len(diffs)} 个，hunk {sum(len(d.hunks) for d in diffs)} 个")
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

    typer.echo(f"可审查：{len(result.reviewable)} 个文件")
    for item in result.reviewable:
        typer.echo(f"  {item.path}")

    typer.echo("")
    typer.echo(f"跳过：{len(result.skipped)} 个文件")
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
            f"第 {index} 组：{package_label(item)}"
            f"（{len(item.files)} 个文件，{len(item.hunks)} 个 hunk，"
            f"约 {total}/{budget} token）{over}"
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
        typer.echo(f"没有发现（本组可审查文件 {len(reviewable)} 个）")
        return

    for item in report.findings:
        typer.echo(format_finding(item))
    typer.echo("")
    typer.echo(f"共 {len(report.findings)} 条发现（耗时 {report.elapsed_seconds:.1f}s）")


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


@dataclass(frozen=True)
class _GroupOutcome:
    """一个审查单元跑完之后交回给调度方的全部结果。

    做成一个值对象是为了让**串行与并发用同一条实现**：单组怎么跑（定位 → 反思 →
    校验）只有一份代码，调度方只决定「怎么调用它、什么时候打印」。这样
    `concurrency = 1` 与并发在行为上的差别就只剩「谁先跑完」。

    Attributes:
        index: 组在列表里的序号（从 1 起）。**汇总时按它排序** —— 并发跑完的顺序
            是不确定的，输出顺序不能跟着它变
        group_id: 审查单元编号
        comments: 本组保留的评论
        misses: 疑似漏报信号
        gaps: 没被引用的高优先级静态发现
        usage: 本组用量（审查循环 + 反思）
        exit_reason: 退出原因
        round_count: 实际轮次
        unlocated: 未定位评论数
        removed: 被反思删掉的评论数
        injected: 注入本组的静态发现数
        referenced: 被引用的静态发现数
        pending_hunks: 没审完的 hunk 数
        elapsed_seconds: 本组从开始审查到反思结束的墙钟耗时。**含反思** ——
            预算闸判的就是这个量级的耗时，报不全等于没法对账
    """

    index: int
    group_id: str
    comments: list[Comment]
    misses: list[PotentialMiss]
    gaps: list[StaticFinding]
    usage: UsageTotals
    exit_reason: str
    round_count: int
    unlocated: int
    removed: int
    injected: int
    referenced: int
    pending_hunks: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _RunResult:
    """`_run_groups` 的产出。

    做成值对象而不是元组：字段已经到七个（评论、漏报、静态缺口、三个计数、用量），
    再往下加一个元组就没法读了，而调用方要的是「有哪些东西」而不是「第几个」。

    Attributes:
        comments: 保留的评论（含未定位）
        misses: 疑似漏报信号
        gaps: 没被引用的高优先级静态发现
        skipped_groups: 因预算没跑的组数
        failed_groups: 异常退出的组数
        pending_hunks: 没审完的 hunk 数
        usage: 累计用量（审查循环 + 反思）
    """

    comments: list[Comment]
    misses: list[PotentialMiss]
    gaps: list[StaticFinding]
    skipped_groups: int
    failed_groups: int
    pending_hunks: int
    usage: UsageTotals


def _run_one_group(
    index: int,
    item: Group,
    llm: LLMClient,
    project_root: Path,
    head_rev: str,
    diff_index: Mapping[str, Diff],
    static_report: StaticReport,
) -> _GroupOutcome:
    """跑一个审查单元：审查 → 定位 → 反思 → 后置校验。

    **这是串行与并发共用的那一份实现**，里面不打印任何东西（打印归调度方，
    否则并发时多线程会交错着写终端）。

    Args:
        index: 组在列表里的序号（从 1 起）
        item: 审查单元
        llm: LLM 客户端
        project_root: 项目根目录
        head_rev: 被审查的版本
        diff_index: `路径 → Diff` 索引（**全量**，组内组外都有）
        static_report: 静态分析结果

    Returns:
        本组的全部结果
    """
    started = time.monotonic()
    # 挑选不做二次裁剪：**预算在分组那一步就执行完了**（装不下的 hunk 顺延到
    # 下一组），这里再裁一次只会让「注入的」与「算过成本的」变成两批东西
    injected = select_for_group(item.files, static_report.findings)
    # 压缩配置直接用 settings 里那一份：它的 context_window 已经是从 [llm] 拼过来的
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
    verdict = verify_group(injected, outcome.kept)
    return _GroupOutcome(
        index=index,
        group_id=result.group_id,
        comments=list(outcome.kept),
        misses=list(outcome.misses),
        gaps=list(verdict.unhandled_high),
        # 反思的用量也要进总账：组级闸看不到它（那时循环已经结束了），而它每组一次
        usage=add_usage(result.usage, outcome.usage),
        exit_reason=result.exit_reason,
        round_count=result.round_count,
        # 未定位数是**定位**这一步的统计，与反思删掉了几条无关，所以按定位结果算
        unlocated=sum(1 for comment in located if comment.locate_level == LOCATE_FAILED),
        removed=len(outcome.removed),
        injected=len(injected),
        referenced=len(verdict.referenced),
        pending_hunks=len(result.pending_hunk_ids),
        elapsed_seconds=time.monotonic() - started,
    )


def _echo_group(item: Group, index: int) -> None:
    """打印「开始跑这一组」。

    Args:
        item: 审查单元
        index: 组序号（从 1 起）
    """
    typer.echo(
        f"第 {index} 组：{package_label(item)}"
        f"（{len(item.files)} 个文件，{len(item.hunks)} 个 hunk）"
    )


def _echo_outcome(found: _GroupOutcome, costs: CostConfig) -> None:
    """打印一组的产出与花费。

    没审完的 hunk 必须报：不报的话「0 条评论」看起来就像「这段代码很干净」。

    Args:
        found: 本组结果
        costs: 价格表
    """
    static_note = f"，静态发现={found.injected}(引用{found.referenced})" if found.injected else ""
    gap_note = f"，⚠️ 未审 hunk={found.pending_hunks}" if found.pending_hunks else ""
    typer.echo(
        f"  {found.group_id} 完成：{len(found.comments)} 条评论"
        f"（exit_reason={found.exit_reason}，轮次={found.round_count}，"
        f"未定位={found.unlocated}，反思删除={found.removed}{static_note}{gap_note}）"
    )
    typer.echo(f"    本组 {describe(found.usage, costs)} · {found.elapsed_seconds:.1f} 秒")


def _run_groups(
    groups: list[Group],
    project_root: Path,
    head_rev: str,
    diff_index: Mapping[str, Diff],
    static_report: StaticReport,
    started: float,
) -> _RunResult:
    """跑完所有审查单元，每组跑完立刻定位与反思。

    定位与反思都放在单组里而不是等所有组跑完：设计文档的节奏就是「每组审查完成
    → 定位 → 反思」，而反思是**每组一次 LLM 调用**，它要的素材（本组 diff +
    本组评论 + 定位元数据）只有在那里才齐。

    **并发是「同时开几组」，不是「怎么汇总」**：`[review].concurrency`（1..8，
    默认 1 即串行）决定同时在跑几组，汇总一律**按组序号**排序 —— 并发跑完的顺序
    不确定，输出顺序不能跟着它变。串行与并发走的是同一个 `_run_one_group`，
    差别只在调度：`concurrency = 1` 时是普通循环，一点线程开销都没有。

    **全局预算在这里执行**：每开一组之前先看总账（钱 + 时间），超了就停掉剩下的组。
    并发下这条的语义要说清：**已经在跑的那几组不打断**（打断会丢掉它们已经审出的
    评论），只是**不再提交新的组**。所以并发度会决定「超预算时多花了多少」——
    最坏是多跑 `concurrency - 1` 组。

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
        `_RunResult`（评论、漏报、静态缺口、三个计数、累计用量）。
        评论含未定位的那些（带 locate_level 标记）
    """
    llm = _llm_or_exit()
    concurrency = settings.review.concurrency
    if concurrency > 1:
        logger.info("并发审查：同时最多 %d 组", concurrency)
    outcomes, failed = _dispatch(
        groups, llm, project_root, head_rev, diff_index, static_report, started
    )

    comments: list[Comment] = []
    misses: list[PotentialMiss] = []
    gaps: list[StaticFinding] = []
    total = UsageTotals()
    pending_hunks = 0
    for found in outcomes:
        comments.extend(found.comments)
        misses.extend(found.misses)
        gaps.extend(found.gaps)
        total = add_usage(total, found.usage)
        pending_hunks += found.pending_hunks
    skipped = len(groups) - len(outcomes) - failed
    # 总计也要进日志：stdout 那份是给人看的，日志这份是给 grep 的（排查「这条命令
    # 花了多久、多少钱」时不该要求人重定向 stdout）
    # 措辞是「审查单元跑完」而不是「review 结束」：这时候排序与渲染还没做，
    # 日志里写「结束」会让后面那几条看起来像是发生在收尾之后
    logger.info(
        "所有审查单元跑完：%d/%d 组（未跑 %d，异常 %d），%s，共 %.1f 秒",
        len(outcomes),
        len(groups),
        skipped,
        failed,
        describe(total, settings.cost),
        time.monotonic() - started,
    )
    if not skipped and not failed:
        typer.echo(f"总计 {describe(total, settings.cost)}")
    return _RunResult(
        comments=comments,
        misses=misses,
        gaps=gaps,
        skipped_groups=skipped,
        failed_groups=failed,
        pending_hunks=pending_hunks,
        usage=total,
    )


def _dispatch(
    groups: list[Group],
    llm: LLMClient,
    project_root: Path,
    head_rev: str,
    diff_index: Mapping[str, Diff],
    static_report: StaticReport,
    started: float,
) -> tuple[list[_GroupOutcome], int]:
    """按并发度把组跑完，**按组序号**返回结果。

    串行（`concurrency = 1`）走普通循环：不建线程池、不碰锁 —— 默认路径上一点
    额外开销都不该有。并发走 `ThreadPoolExecutor`：**线程而不是进程**，因为这活
    几乎全是等待（HTTP 调用 + git 子进程），GIL 在等待时会释放。

    Args:
        groups: 审查单元列表
        llm: LLM 客户端
        project_root: 项目根目录
        head_rev: 被审查的版本
        diff_index: `路径 → Diff` 索引
        static_report: 静态分析结果
        started: 整条命令的开始时刻

    Returns:
        (跑过的组的结果（按序号升序）, 异常退出的组数)。
        **因预算没跑的**组两个都不在里面 —— 它由调用方按「组数 − 跑过 − 失败」算
    """
    concurrency = settings.review.concurrency
    work = partial(
        _run_one_group,
        llm=llm,
        project_root=project_root,
        head_rev=head_rev,
        diff_index=diff_index,
        static_report=static_report,
    )
    if concurrency <= 1:
        results: list[_GroupOutcome] = []
        for index, item in enumerate(groups, start=1):
            if _review_budget_stop(results, started, remaining=len(groups) - index + 1):
                break
            _echo_group(item, index)
            found = work(index, item)
            results.append(found)
            _echo_outcome(found, settings.cost)
        return results, 0

    return _dispatch_concurrent(groups, work, started, concurrency)


def _dispatch_concurrent(
    groups: list[Group],
    work: Callable[..., _GroupOutcome],
    started: float,
    concurrency: int,
) -> tuple[list[_GroupOutcome], int]:
    """并发跑组：池子填满就等，跑完一组先看总账、再决定要不要补下一组。

    **预算每次提交前都查**（不是只在池子空了才查）：只在池子空时查的写法，会在
    池子里一直有活的时候无限续提交 —— 那等于闸门不生效。

    Args:
        groups: 审查单元列表
        work: 跑一组的函数
        started: 整条命令的开始时刻
        concurrency: 并发度

    Returns:
        (跑过的组的结果（按序号升序）, 异常退出的组数)
    """
    finished: list[_GroupOutcome] = []
    failed = 0
    stopped = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        running: dict[Future[_GroupOutcome], int] = {}
        next_index = 1
        while next_index <= len(groups) or running:
            while not stopped and next_index <= len(groups) and len(running) < concurrency:
                if _review_budget_stop(finished, started, remaining=len(groups) - next_index + 1):
                    stopped = True
                    break
                item = groups[next_index - 1]
                _echo_group(item, next_index)
                running[pool.submit(work, next_index, item)] = next_index
                next_index += 1
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                del running[future]
                found = _result_or_none(future)
                if found is None:
                    failed += 1
                    continue
                finished.append(found)
                _echo_outcome(found, settings.cost)
    finished.sort(key=lambda item: item.index)
    return finished, failed


def _result_or_none(future: Future[_GroupOutcome]) -> _GroupOutcome | None:
    """取一个已完成的 future 的结果，异常时记 ERROR 并放弃这一组。

    单组内部的异常已经在 `_run_graph` 与 `reflect` 里各自收口过；能漏到这里的
    是编排层自己的错（比如定位/校验那一步）。**收口但不吞**：ERROR + 完整堆栈，
    然后继续跑别的组 —— 一组出问题不该让整条 review 归零。

    Args:
        future: 已完成的 future

    Returns:
        结果；出异常时返回 None
    """
    try:
        return future.result()
    except Exception as exc:
        logger.error("某一组异常退出，跳过它的结果：%s", exc, exc_info=exc)
        return None


def _review_budget_stop(finished: Sequence[_GroupOutcome], started: float, remaining: int) -> bool:
    """整条 review 的预算闸：超了就打印一句并返回 True。

    只统计**已经跑完**的组：并发下正在跑的那几组拦不住（打断它们会丢掉已经审出的
    评论），所以这里决定的是「还要不要再开新的」。

    Args:
        finished: 已跑完的组
        started: 整条命令的开始时刻
        remaining: 还没提交的组数（打印给人看）

    Returns:
        该停时返回 True
    """
    total = UsageTotals()
    for found in finished:
        total = add_usage(total, found.usage)
    stop = exceeded(
        total,
        time.monotonic() - started,
        prices=settings.cost,
        max_cost=settings.review.max_review_cost,
        max_seconds=settings.review.max_review_seconds,
    )
    if stop is None:
        return False
    logger.warning("整条 review 超预算（%s），不再开新的组", stop)
    typer.echo(
        f"⚠️ 整条 review 超预算（{stop}）：累计 {describe(total, settings.cost)}，"
        f"已用 {time.monotonic() - started:.0f} 秒 —— 剩余 {remaining} 组不再审查"
    )
    return True


@app.command()
def review(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    path: RepoPath = None,
    trace_attr: Annotated[
        list[str] | None,
        typer.Option(
            "--trace-attr",
            help="附加到根 span 的属性（key=value，可重复）。pr_url / instance_id "
            "这类只有调用方知道的信息由它带进来",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json-out",
            help="把机器可读的结果写到这个文件（终端渲染照常）。bench 驱动靠它拿评论",
        ),
    ] = None,
) -> None:
    """完整审查：diff → 过滤 → hunk 映射 → 静态分析 → 分组 → LLM 审查 → 排序 → 渲染。

    开了追踪时（`PRISM_TRACE_FILE` / `PRISM_TRACE_DIR`）会额外写一份 OTLP JSON：
    根 span 是这次审查，子 span 是每个审查组、每次 LLM 调用、每次工具执行。
    """
    try:
        extra = parse_trace_attrs(trace_attr or [])
    except ValueError as exc:
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    with tracer.start_as_current_span("prism_review") as span:
        set_attributes(span, {sc.PRISM_BASE_REV: base, sc.PRISM_HEAD_REV: head, **extra})
        stats = _review(base, head, path, json_out)
        set_attributes(
            span,
            {
                sc.PRISM_SKIPPED_GROUPS: stats.skipped_groups,
                sc.PRISM_FAILED_GROUPS: stats.failed_groups,
                sc.PRISM_PENDING_HUNKS: stats.pending_hunks,
                sc.PRISM_DURATION_MS: int(stats.duration_seconds * 1000),
            },
        )


def _review(base: str, head: str, path: Path | None, json_out: Path | None = None) -> ReviewStats:
    """`prism review` 的主体（根 span 由调用方开）。

    Args:
        base: base 分支 / 版本
        head: head 分支 / 版本
        path: 被审查仓库路径
        json_out: 机器可读结果的输出路径；None 表示不写

    Returns:
        审查统计（根 span 的属性从它取）
    """
    started = time.monotonic()
    root = _resolve_repo(path)
    logger.info("开始审查: root=%s base=%s head=%s", root, base, head)

    typer.echo("加载 diff…")
    diffs = _load_or_exit(root, base, head)

    typer.echo(f"过滤文件…（diff 里 {len(diffs)} 个文件）")
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
    typer.echo(f"静态分析…（{len(reviewable)} 个可审查文件）")
    static_report = analyze(root, head, sorted({item.file_path for item in mapped}), mapped)
    _echo_static(static_report)

    typer.echo(f"按包分组…（{len(reviewable)} 个可审查文件）")
    groups = group_by_package(mapped, static_report.findings)

    if not groups:
        typer.echo("没有可审查的 hunk。")

    # 没有分组时不去构造 LLM 客户端：`prism review` 在空 diff 上不该要求密钥
    if groups:
        outcome = _run_groups(groups, root, head, diff_index, static_report, started)
    else:
        outcome = _RunResult(
            comments=[],
            misses=[],
            gaps=[],
            skipped_groups=0,
            failed_groups=0,
            pending_hunks=0,
            usage=UsageTotals(),
        )
    comments = sort_comments(outcome.comments)

    stats = ReviewStats(
        base_branch=base,
        head_sha=head,
        files_count=len(reviewable),
        comments_count=len(comments),
        duration_seconds=time.monotonic() - started,
        skipped_groups=outcome.skipped_groups,
        pending_hunks=outcome.pending_hunks,
        failed_groups=outcome.failed_groups,
    )
    typer.echo("")
    typer.echo(TerminalRenderer().render(comments, stats, outcome.misses, outcome.gaps))
    typer.echo(f"\n审查完成：{len(comments)} 条评论")
    if json_out is not None:
        write_report(
            json_out,
            build_report(
                base=base,
                head=head,
                project_root=str(root),
                stats=stats,
                usage=outcome.usage,
                prices=settings.cost,
                comments=comments,
                misses=outcome.misses,
                static_gaps=outcome.gaps,
            ),
        )
    return stats
