"""Prism CLI 入口。

本模块是全部可执行命令的唯一入口（`[project.scripts] prism = "prism.cli.main:app"`）。
下面是命令总览；每条命令的详细参数见各自的 `--help`，实现见本文件内的同名函数。

## 全局

    prism --version                 显示版本号并退出
    prism --help / prism            打印帮助（不带子命令时）

    所有命令都在「当前工作目录」这个 git 仓库里运行，以 merge-base(base, head)
    为基准比较，所以只统计 PR 实际引入的改动 —— base 分支上的新提交不会混进来。

## 审查前流程（确定性，不调用 LLM、零 token 成本）

纯读取命令，用于在花钱调 LLM 之前确认「改了什么、审哪些、怎么分组」。
四个命令依次是流水线的下一个环节，可以逐个跑、逐个核对。

    prism diff      [--base main] [--head HEAD]
        加载并解析 diff，打印文件数 / hunk 数 / 增删行数。
        对应 core/diff/：pygit2 算 merge-base + 状态机解析 unified diff。

    prism filter    [--base main] [--head HEAD]
        按规则过滤文件，打印可审查 / 跳过清单（含跳过原因）。
        三级过滤：硬性排除（二进制 / 符号链接）→ 扩展名白名单（.go）
        → 内置降噪（*_test.go、*.pb.go）。对应 core/filter/。

    prism hunk-map  [--base main] [--head HEAD]
        把每个 hunk 归属到包含其改动行的符号，打印 hunk 位置与归属符号。
        只处理通过过滤的 .go 文件。对应 core/ast/：tree-sitter 提取符号
        （函数 / 方法 / 包声明 / import / const / var / type），按改动行求交。

    prism group     [--base main] [--head HEAD]
        按包路径（目录）分组，打印审查单元。
        这是审查前流程的最后一个环节，输出即为送进 LLM 的「审查单元」。
        对应 core/grouping/。

## 完整审查（调用 LLM，产生 token 成本）

    prism review    [--base main] [--head HEAD] [--model M] [--base-url URL]
        串起完整链路：diff → 过滤 → hunk 映射 → 分组 → LLM 审查 → 排序 → 终端渲染。

        审查前三个阶段的进度打到 stdout，详细轨迹走 logging（stderr）：
        想看完整轨迹设 `PRISM_LOG_LEVEL=DEBUG`，想只看结果设 `PRISM_LOG_LEVEL=WARNING`。

        需要环境变量 `DEEPSEEK_API_KEY`，缺失时以退出码 1 明确报错。

        退出码：0 正常（含 0 条评论）；1 参数 / 环境 / git 错误。

## 典型用法

    # 先干跑一遍看清范围（不花钱），确认后再真审
    prism group  --base main --head origin/some-branch
    prism review --base main --head origin/some-branch

    # 排查「为什么某个文件没被审」
    prism filter --base main --head origin/some-branch

    # 排查「某个 hunk 归属不准」
    prism hunk-map --base main --head origin/some-branch

注意：`review` 的 `--head` 是**被审查的版本**，工具读源码时读的就是它，
不是当前工作区。所以工作区停在别的分支也能正确审查
（见 core/tools/context.py 的 head_rev）。
"""

import logging
import time
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import typer

from prism.core.ast.mapper import format_location, format_symbol, map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.graph.main_loop import run_main_loop
from prism.core.graph.state import MainLoopResult
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.core.output.renderer import TerminalRenderer
from prism.core.output.sort import sort_comments, to_comments
from prism.models.ast import HunkWithSymbol
from prism.models.comment import ReviewStats
from prism.models.diff import Diff
from prism.models.grouping import Group
from prism.models.llm import LLMConfig
from prism.observability.logging import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(help="Prism — 确定性优先的 Go 代码审查 Agent")


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
    version_: bool = typer.Option(
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
        version_: 是否传入 `--version`（由 `_version_callback` 处理）
    """
    setup_logging()
    if ctx.invoked_subcommand is None:
        logger.debug("未指定子命令：打印帮助")
        typer.echo(ctx.get_help())
        raise typer.Exit


def _load_or_exit(base: str, head: str) -> list[Diff]:
    """加载 diff，失败时打印错误并终止进程。

    Args:
        base: base 分支 / 版本
        head: head 分支 / 版本

    Returns:
        Diff 列表
    """
    try:
        return load_diffs(Path.cwd(), base, head)
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


def _map_or_exit(base: str, head: str) -> list[HunkWithSymbol]:
    """跑完审查前流程（diff → 过滤 → hunk 映射），失败时终止进程。

    Args:
        base: base 分支 / 版本
        head: head 分支 / 版本

    Returns:
        带归属符号的 hunk 列表
    """
    reviewable = filter_files(_load_or_exit(base, head)).reviewable
    try:
        return map_hunks_to_symbols(Path.cwd(), base, head, reviewable)
    except ValueError as exc:
        logger.error("hunk 映射失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def diff(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
) -> None:
    """加载并解析 diff，打印结构摘要（不调用 LLM）。

    以 merge-base(base, head) 为基准，只统计 PR 实际引入的改动。
    """
    logger.info("加载 diff: base=%s head=%s", base, head)
    diffs = _load_or_exit(base, head)

    typer.echo(f"Loaded {len(diffs)} files, {sum(len(d.hunks) for d in diffs)} hunks")
    for item in diffs:
        typer.echo(f"  {item.path}: {len(item.hunks)} hunks (+{item.insertions}/-{item.deletions})")


@app.command()
def filter(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
) -> None:
    """按规则过滤文件，打印可审查 / 跳过清单（不调用 LLM）。"""
    logger.info("过滤文件: base=%s head=%s", base, head)
    diffs = _load_or_exit(base, head)
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
) -> None:
    """把每个 hunk 归属到符号，打印映射结果（不调用 LLM）。

    只处理通过文件过滤的 .go 文件。
    """
    logger.info("映射 hunk: base=%s head=%s", base, head)
    mapped = _map_or_exit(base, head)

    if not mapped:
        typer.echo("没有可映射的 hunk")
        return

    blocks = [f"{format_location(item)}\n  归属：{format_symbol(item.symbol)}" for item in mapped]
    typer.echo("\n\n".join(blocks))


@app.command()
def group(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
) -> None:
    """按包路径分组，打印审查单元（不调用 LLM）。

    这是审查前流程的最后一个环节：输出即为送入 LLM 的「审查单元」。
    """
    logger.info("分组: base=%s head=%s", base, head)
    groups = group_by_package(_map_or_exit(base, head))

    if not groups:
        typer.echo("没有可审查的 hunk")
        return

    blocks = []
    for index, item in enumerate(groups, start=1):
        header = (
            f"Group {index}: {item.package_path} "
            f"({_plural(len(item.files), 'file')}, {_plural(len(item.hunks), 'hunk')})"
        )
        lines = [header]
        for path in item.files:
            hunk_ids = ", ".join(h.hunk_id for h in item.hunks if h.file_path == path)
            lines.append(f"  - {PurePosixPath(path).name} ({hunk_ids})")
        blocks.append("\n".join(lines))

    typer.echo("\n\n".join(blocks))


def _llm_or_exit(model: str | None = None, base_url: str | None = None) -> LLMClient:
    """构造 LLM 客户端，失败时打印错误并终止进程。

    Args:
        model: 覆盖默认模型名（可选）
        base_url: 覆盖默认 base URL（可选），用于切到其他 OpenAI 兼容端点

    Returns:
        LLMClient 实例
    """
    try:
        config = LLMConfig.from_env()
        if model:
            config.model = model
        if base_url:
            config.base_url = base_url
        return LLMClient(config)
    except ValueError as exc:
        logger.error("LLM 初始化失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc


def _run_groups(
    groups: list[Group],
    project_root: Path,
    head_rev: str,
    model: str | None = None,
    base_url: str | None = None,
) -> list[MainLoopResult]:
    """串行跑完所有审查单元（V1 不做并发）。

    Args:
        groups: 审查单元列表
        project_root: 项目根目录
        head_rev: 被审查的版本，传给工具读源码用
        model: 覆盖默认模型名（可选）
        base_url: 覆盖默认 base URL（可选）

    Returns:
        各组的循环产出
    """
    llm = _llm_or_exit(model, base_url)
    results: list[MainLoopResult] = []
    for index, item in enumerate(groups, start=1):
        typer.echo(
            f"Processing Group {index}: {item.package_path} "
            f"({_plural(len(item.files), 'file')}, {_plural(len(item.hunks), 'hunk')})"
        )
        result = run_main_loop(item, llm, str(project_root), head_rev)
        results.append(result)
        typer.echo(
            f"  {result.group_id} completed: {_plural(len(result.comments), 'comment')} "
            f"(exit_reason={result.exit_reason}, 轮次={result.round_count})"
        )
    return results


@app.command()
def review(
    base: str = typer.Option("main", "--base", help="base 分支 / 版本"),
    head: str = typer.Option("HEAD", "--head", help="head 分支 / 版本"),
    model: str = typer.Option("", "--model", help="覆盖默认模型名，如 deepseek-chat"),
    base_url: str = typer.Option("", "--base-url", help="覆盖默认 base URL，切到其他兼容端点"),
) -> None:
    """完整审查：diff → 过滤 → hunk 映射 → 分组 → LLM 审查 → 排序 → 终端渲染。"""
    started = time.monotonic()
    root = Path.cwd()
    logger.info("开始审查: base=%s head=%s root=%s", base, head, root)

    typer.echo("Loading diffs...")
    diffs = _load_or_exit(base, head)

    typer.echo(f"Filtering files... ({_plural(len(diffs), 'file')} in diff)")
    reviewable = filter_files(diffs).reviewable

    typer.echo(f"Grouping by package... ({_plural(len(reviewable), 'reviewable file')})")
    try:
        groups = group_by_package(map_hunks_to_symbols(root, base, head, reviewable))
    except ValueError as exc:
        logger.error("hunk 映射失败: %s", exc)
        typer.echo(f"错误：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not groups:
        typer.echo("没有可审查的 hunk。")

    results = _run_groups(groups, root, head, model or None, base_url or None) if groups else []
    comments = sort_comments(to_comments(results, groups))

    stats = ReviewStats(
        base_branch=base,
        head_sha=head,
        files_count=len(reviewable),
        comments_count=len(comments),
        duration_seconds=time.monotonic() - started,
    )
    typer.echo("")
    typer.echo(TerminalRenderer().render(comments, stats))
    typer.echo(f"\nReview completed: {_plural(len(comments), 'comment')}")
