"""hunk → 符号映射。

用 tree-sitter 提取源码里的符号行号范围，与 hunk 的**改动行**求交，
把每个 hunk 归属到包含其改动行的符号上。

diff 用 `-U0` 生成（见 `loader.CONTEXT_LINES`），hunk 里只有改动行，
所以行号直接来自头字段。归属判定用改动行而不是 hunk 范围，是因为
按整段范围求交会把改动其实不在里面的相邻符号也挂上；用改动行后
只有真正包含改动的符号才会命中。
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pygit2

from prism.core.ast.extractor import (
    KIND_CONST,
    KIND_FUNCTION,
    KIND_IMPORT,
    KIND_INTERFACE,
    KIND_METHOD,
    KIND_PACKAGE,
    KIND_STRUCT,
    KIND_TYPE,
    KIND_VAR,
    extract_symbols,
)
from prism.core.diff.loader import read_blob, resolve_range
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.diff import Diff, Hunk

logger = logging.getLogger(__name__)

# hunk 编号前缀与补零宽度
HUNK_ID_PREFIX = "H-"
HUNK_ID_WIDTH = 2

# 无法归属到任何符号时的展示文本
TOP_LEVEL_LABEL = "<顶层>"

# 符号类型的中文展示
KIND_LABELS: dict[str, str] = {
    KIND_PACKAGE: "包声明",
    KIND_FUNCTION: "函数",
    KIND_METHOD: "方法",
    KIND_IMPORT: "import 块",
    KIND_CONST: "const 块",
    KIND_VAR: "var 块",
    KIND_STRUCT: "结构体",
    KIND_INTERFACE: "接口",
    KIND_TYPE: "类型",
}

# 注释行的行首字符（去除缩进后）
_COMMENT_PREFIXES = ("//", "/*", "*")


@dataclass
class _FileFacts:
    """某个文件在某一侧的源码事实。

    Attributes:
        lines: 源码按行切分的结果（0-based 下标，行号 = 下标 + 1）
        symbols: 该文件的符号表
    """

    lines: list[str]
    symbols: list[Symbol]


def map_hunks_to_symbols(
    repo_path: str | Path,
    base: str,
    head: str,
    diffs: list[Diff],
) -> list[HunkWithSymbol]:
    """把每个 hunk 归属到包含其改动行的符号。

    有新增行时用新文件侧（新文件源码 + 新增行号），纯删除时用旧文件侧
    （merge-base 源码 + 删除行号）—— 后者与文档「删除用旧文件的 AST」一致，
    也让删除文件、纯删除 hunk 统一走同一条路径。
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

    # 同一文件同一侧只读一次：(路径, 是否旧版本) → 源码事实
    cache: dict[tuple[str, bool], _FileFacts] = {}
    results: list[HunkWithSymbol] = []
    counter = 0

    for diff in sorted(diffs, key=lambda item: item.path):
        for hunk in diff.hunks:
            counter += 1
            results.append(
                HunkWithSymbol(
                    hunk=hunk,
                    file_path=diff.path,
                    symbols=_attribute(cache, repo, diff_base, head_commit, diff.path, hunk),
                    hunk_id=f"{HUNK_ID_PREFIX}{counter:0{HUNK_ID_WIDTH}d}",
                )
            )

    unmatched = sum(1 for item in results if not item.symbols)
    logger.info("映射完成：%d 个 hunk，其中 %d 个未归属到符号", len(results), unmatched)
    return results


def changed_lines(hunk: Hunk) -> tuple[list[int], list[int]]:
    """取出 hunk 的改动行号。

    diff 由 `loader.CONTEXT_LINES = 0`（等价 `git diff -U0`）生成，hunk 里
    只有改动行、没有上下文行，因此头字段本身就是改动行的位置与数量：
    新文件侧是 ``[new_start, new_start + new_lines - 1]``，旧文件侧同理，
    该侧行数为 0 时表示这一侧没有改动行。

    Args:
        hunk: 待解析的 hunk

    Returns:
        (新增行号列表, 删除行号列表)，均为 1-based
    """
    added = list(range(hunk.new_start, hunk.new_start + hunk.new_lines)) if hunk.new_lines else []
    removed = list(range(hunk.old_start, hunk.old_start + hunk.old_lines)) if hunk.old_lines else []
    return added, removed


def display_range(hunk: Hunk) -> tuple[int, int]:
    """取 hunk 用于展示的行号闭区间。

    优先新文件侧；纯删除（新文件侧为空）时退到旧文件侧。

    Args:
        hunk: 待展示的 hunk

    Returns:
        (起始行, 结束行)
    """
    if hunk.new_lines:
        return hunk.new_start, hunk.new_start + hunk.new_lines - 1
    return hunk.old_start, hunk.old_start + max(hunk.old_lines, 1) - 1


def format_location(item: HunkWithSymbol) -> str:
    """渲染 hunk 的位置文本。

    Args:
        item: 带符号的 hunk

    Returns:
        ``H-01 internal/agent/agent.go:245-260`` 形式；单行 hunk 只显示一个行号
    """
    start, end = display_range(item.hunk)
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


def _attribute(
    cache: dict[tuple[str, bool], _FileFacts],
    repo: pygit2.Repository,
    base_commit: pygit2.Commit,
    head_commit: pygit2.Commit,
    path: str,
    hunk: Hunk,
) -> list[Symbol]:
    """把单个 hunk 归属到符号。

    有新增行 → 新文件侧；纯删除 → 旧文件侧；两侧都没有改动行 → 空。
    改动行是注释时先「落」到注释块下方的第一个非注释行（见 _landing_line）。

    Args:
        cache: 文件事实缓存
        repo: pygit2 仓库
        base_commit: merge-base 侧的 commit
        head_commit: head 侧的 commit
        path: 文件路径
        hunk: 待归属的 hunk

    Returns:
        命中的符号列表
    """
    added, removed = changed_lines(hunk)
    if added:
        facts = _file_facts(cache, repo, head_commit, path, side=False)
        return _containing(facts.symbols, [_landing_line(facts.lines, ln) for ln in added])
    if removed:
        facts = _file_facts(cache, repo, base_commit, path, side=True)
        return _containing(facts.symbols, [_landing_line(facts.lines, ln) for ln in removed])
    return []


def _landing_line(lines: list[str], line_no: int) -> int:
    """把注释行「落」到它下方第一个非注释行。

    注释在语法树里是独立节点，不挂在它描述的声明下面 —— 「改了一行函数文档
    注释」在纯语法视角下不归属任何符号。Go 的约定是「紧贴声明上方、不隔空行的
    连续注释块 = 该声明的文档注释」，所以向下找到注释块结束后的第一个非注释行，
    就能落到那条声明上。

    非注释行原样返回；注释块一直到文件末尾则也原样返回（无从归属）。

    Args:
        lines: 源码按行切分的结果，行号 = 下标 + 1
        line_no: 待归位的行号（1-based）

    Returns:
        归位后的行号（1-based）
    """
    if line_no < 1 or line_no > len(lines) or not _is_comment_line(lines[line_no - 1]):
        return line_no

    index = line_no  # 下一个待检查行的 0-based 下标
    while index < len(lines) and _is_comment_line(lines[index]):
        index += 1
    return index + 1 if index < len(lines) else line_no


def _is_comment_line(line: str) -> bool:
    """判断一行是否是注释行。

    Args:
        line: 源码的一行

    Returns:
        去缩进后以 ``//`` / ``/*`` / ``*`` 开头则视为注释
    """
    return line.strip().startswith(_COMMENT_PREFIXES)


def _file_facts(
    cache: dict[tuple[str, bool], _FileFacts],
    repo: pygit2.Repository,
    commit: pygit2.Commit,
    path: str,
    side: bool,
) -> _FileFacts:
    """取某个文件在指定侧的源码事实（行文本 + 符号表），带缓存。

    Args:
        cache: 缓存字典
        repo: pygit2 仓库
        commit: 目标 commit
        path: 文件路径
        side: True 表示旧版本侧

    Returns:
        该文件的 _FileFacts；文件不存在时为空事实
    """
    key = (path, side)
    if key not in cache:
        source = read_blob(repo, commit, path)
        if source is None:
            logger.debug("文件 %s 在该版本中不存在，符号表为空", path)
            cache[key] = _FileFacts(lines=[], symbols=[])
        else:
            cache[key] = _FileFacts(lines=source.splitlines(), symbols=extract_symbols(source))
    return cache[key]


def _containing(symbols: list[Symbol], lines: list[int]) -> list[Symbol]:
    """取出包含任一给定行的符号。

    Args:
        symbols: 候选符号
        lines: 改动行号列表（已归位）

    Returns:
        命中的符号列表，保持入参顺序
    """
    return [s for s in symbols if any(s.start_line <= line <= s.end_line for line in lines)]
