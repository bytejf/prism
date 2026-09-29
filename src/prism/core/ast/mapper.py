"""hunk → 函数映射。

用 tree-sitter 提取源码里的函数 / 方法行号范围，与 hunk 行号范围求交，
把每个 hunk 归属到重叠的符号上。
"""

import logging
from pathlib import Path

from prism.core.ast.extractor import KIND_FUNCTION, KIND_METHOD, extract_symbols
from prism.core.diff.loader import read_blob, resolve_range
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Diff, Hunk

logger = logging.getLogger(__name__)

# hunk 编号前缀与补零宽度
HUNK_ID_PREFIX = "H-"
HUNK_ID_WIDTH = 2

# 无法归属到任何函数时的展示文本
TOP_LEVEL_LABEL = "<顶层>"

# 符号类型的中文展示
KIND_LABELS: dict[str, str] = {
    KIND_FUNCTION: "函数",
    KIND_METHOD: "方法",
}


def map_hunks_to_symbols(
    repo_path: str | Path,
    base: str,
    head: str,
    diffs: list[Diff],
) -> list[HunkWithSymbol]:
    """把每个 hunk 归属到与它行号范围重叠的函数 / 方法。

    删除的文件用 merge-base 侧源码，其余用 head 侧源码。
    hunk 跨多个函数时归属到全部重叠函数；无重叠时 symbols 为空（顶层）。
    编号按文件路径排序后全局连续分配，保证同一输入产生同一编号。

    Args:
        repo_path: 仓库路径
        base: base 版本（分支名 / SHA）
        head: head 版本（分支名 / SHA）
        diffs: 待映射的 diff 列表

    Returns:
        HunkWithSymbol 列表，按文件路径、文件内 hunk 顺序排列

    Raises:
        ValueError: 仓库无法打开，或版本无法解析
    """
    repo, diff_base, head_commit = resolve_range(repo_path, base, head)

    # 同一文件只解析一次：(路径, 是否用旧版本) → 符号列表
    cache: dict[tuple[str, bool], list[Symbol]] = {}
    results: list[HunkWithSymbol] = []
    counter = 0

    for diff in sorted(diffs, key=lambda item: item.path):
        use_old = diff.is_deleted
        key = (diff.path, use_old)
        if key not in cache:
            source = read_blob(repo, diff_base if use_old else head_commit, diff.path)
            if source is None:
                logger.warning("文件 %s 在目标版本中不存在，其 hunk 归属为空", diff.path)
                cache[key] = []
            else:
                cache[key] = extract_symbols(source)

        symbols = cache[key]
        for hunk in diff.hunks:
            counter += 1
            results.append(
                HunkWithSymbol(
                    hunk=hunk,
                    file_path=diff.path,
                    symbols=_overlapping_symbols(symbols, hunk, use_old),
                    hunk_id=f"{HUNK_ID_PREFIX}{counter:0{HUNK_ID_WIDTH}d}",
                )
            )

    unmatched = sum(1 for item in results if not item.symbols)
    logger.info("映射完成：%d 个 hunk，其中 %d 个未归属到函数", len(results), unmatched)
    return results


def hunk_line_range(hunk: Hunk, use_old_side: bool) -> tuple[int, int]:
    """取 hunk 在指定一侧的行号闭区间（1-based）。

    Args:
        hunk: 待定位的 hunk
        use_old_side: True 取旧文件行号，False 取新文件行号

    Returns:
        (起始行, 结束行)，均为闭区间端点
    """
    if use_old_side:
        start, length = hunk.old_start, hunk.old_lines
    else:
        start, length = hunk.new_start, hunk.new_lines
        if length == 0:
            # 纯删除的 hunk：new_start 是插入锚点，被删内容在新文件中
            # 本应位于 new_start + 1 起、共 old_lines 行的位置
            start, length = hunk.new_start + 1, hunk.old_lines
    return start, start + max(length, 1) - 1


def format_location(item: HunkWithSymbol) -> str:
    """渲染 hunk 的位置文本。

    Args:
        item: 带符号的 hunk

    Returns:
        ``H-01 internal/agent/agent.go:245-260`` 形式；单行 hunk 只显示一个行号
    """
    start, end = hunk_line_range(item.hunk, False)
    span = str(start) if start == end else f"{start}-{end}"
    return f"{item.hunk_id} {item.file_path}:{span}"


def format_symbols(symbols: list[Symbol]) -> str:
    """渲染 hunk 的归属符号文本。

    Args:
        symbols: 归属的符号列表

    Returns:
        ``(*Agent).Run（方法）`` 形式；多个符号用 ``, `` 连接；无符号时为 ``<顶层>``
    """
    if not symbols:
        return TOP_LEVEL_LABEL
    return ", ".join(
        f"{symbol.name}（{KIND_LABELS.get(symbol.kind, symbol.kind)}）" for symbol in symbols
    )


def _overlapping_symbols(symbols: list[Symbol], hunk: Hunk, use_old: bool) -> list[Symbol]:
    """取出与 hunk 行号范围有交集的符号。

    Args:
        symbols: 文件内的候选符号
        hunk: 待归属的 hunk
        use_old: 是否使用旧文件行号

    Returns:
        重叠的符号列表，按源码顺序
    """
    start, end = hunk_line_range(hunk, use_old)
    return [symbol for symbol in symbols if symbol.start_line <= end and symbol.end_line >= start]
