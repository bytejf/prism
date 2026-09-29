"""Prism CLI 入口。

V1 只提供 `--version`，后续模块逐步加入 `diff` / `filter` / `hunk-map` /
`group` / `review` 等子命令。
"""

import logging
from importlib.metadata import version
from pathlib import Path

import typer

from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.models.diff import Diff
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

    typer.echo(f"Reviewable: {len(result.reviewable)} files")
    for item in result.reviewable:
        typer.echo(f"  {item.path}")

    typer.echo("")
    typer.echo(f"Skipped: {len(result.skipped)} files")
    for skipped in result.skipped:
        typer.echo(f"  {skipped.path} ({skipped.reason})")
