"""hunk → 符号映射。

用 tree-sitter 提取源码里的符号行号范围，与 hunk 的**改动行**求交，
把每个 hunk 归属到包含其改动行的符号上。

diff 用 `-U0` 生成（见 `loader.CONTEXT_LINES`），hunk 里只有改动行，
所以行号直接来自头字段。归属判定用改动行而不是 hunk 范围，是因为
按整段范围求交会把改动其实不在里面的相邻符号也挂上；用改动行后
只有真正包含改动的符号才会命中。
"""

import logging
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


def map_hunks_to_symbols(
    repo_path: str | Path,
    base: str,
    head: str,
    diffs: list[Diff],
) -> list[HunkWithSymbol]:
    """把每个 hunk 归属到包含其改动行的符号。

    先按实体边界把 hunk 拆开（`split_by_symbol`），再逐个子 hunk 归属 ——
    这样每个结果要么属于唯一一个符号，要么不属于任何符号（真·顶层改动），
    不会出现「一个 hunk 挂着一串符号」这种含糊状态。
    有新增行时用新文件侧（新文件源码 + 新增行号），纯删除时用旧文件侧
    （merge-base 源码 + 删除行号）—— 与文档「删除用旧文件的 AST」一致。
    编号按文件路径排序、拆分后全局连续分配，保证同一输入产生同一编号。

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

    # 同一文件同一侧只读一次：(路径, 是否旧版本) → 符号表
    cache: dict[tuple[str, bool], list[Symbol]] = {}
    results: list[HunkWithSymbol] = []
    counter = 0

    for diff in sorted(diffs, key=lambda item: item.path):
        for hunk in diff.hunks:
            for piece in _split_hunk(hunk, cache, repo, diff_base, head_commit, diff.path):
                counter += 1
                results.append(
                    HunkWithSymbol(
                        hunk=piece,
                        file_path=diff.path,
                        symbols=_attribute(cache, repo, diff_base, head_commit, diff.path, piece),
                        hunk_id=f"{HUNK_ID_PREFIX}{counter:0{HUNK_ID_WIDTH}d}",
                    )
                )

    unmatched = sum(1 for item in results if not item.symbols)
    logger.info("映射完成：%d 个 hunk，其中 %d 个未归属到符号", len(results), unmatched)
    return results


def _split_hunk(
    hunk: Hunk,
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    base_commit: pygit2.Commit,
    head_commit: pygit2.Commit,
    path: str,
) -> list[Hunk]:
    """按实体边界拆分单个 hunk。

    两侧符号表都给：`+` 行按新文件行号、`-` 行按旧文件行号各自查自己的表。

    Args:
        hunk: 待拆分的 hunk
        cache: 文件符号表缓存
        repo: pygit2 仓库
        base_commit: merge-base 侧的 commit
        head_commit: head 侧的 commit
        path: 文件路径

    Returns:
        子 hunk 列表
    """
    return split_by_symbol(
        hunk,
        new_symbols=_file_symbols(cache, repo, head_commit, path, side=False),
        old_symbols=_file_symbols(cache, repo, base_commit, path, side=True),
    )


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


def split_by_symbol(hunk: Hunk, new_symbols: list[Symbol], old_symbols: list[Symbol]) -> list[Hunk]:
    """按实体边界把 hunk 拆成若干只属于单一符号的子 hunk。

    逐行遍历正文，算出每行的归属；归属变化处开一个新子 hunk。
    **不属于任何实体的行（空行、独立注释）不触发切分**，直接留在当前子 hunk 里
    —— 即「空隙行归给相邻子 hunk」，这样每个子 hunk 都恰好对应一个实体。
    （实体之间一定有这种空隙行，所以不能按行号机械对半切。）

    `+` 行用**新文件侧**符号表、`-` 行用**旧文件侧**符号表 —— 两侧行号是
    两套坐标，混用会把删除行的旧行号拿去查新文件的符号表，拆出跨实体的碎片。

    子 hunk 的行号范围与正文都是从原 hunk 切出来的，两者拼接回原样：
    行号范围切分后并集等于原范围，正文切片的拼接等于原正文。

    Args:
        hunk: 待拆分的 hunk
        new_symbols: 新文件侧的符号表
        old_symbols: 旧文件侧的符号表

    Returns:
        子 hunk 列表；只有一个归属时原样返回 [hunk]
    """
    body = hunk.content.splitlines()
    if not body:
        return [hunk]

    # 第一遍：找出切分点与每段的归属。
    # 按**名字**比较而不是对象身份：新旧两侧符号表是两次提取的结果，同名
    # 实体会是两个不同对象；用身份比较会把「一行改成另一行」（`-旧` + `+新`）
    # 这种同一实体的改动误切成两半。也不能比整个对象：改动会让函数范围变化，
    # 同名实体的行号在新旧两侧本来就可能不同。
    cuts: list[int] = []
    current_name: str | None = None

    old_no, new_no = hunk.old_start, hunk.new_start
    for index, line in enumerate(body):
        if line.startswith("+"):
            owner = _owner_at(new_symbols, new_no)
            new_no += 1
        elif line.startswith("-"):
            owner = _owner_at(old_symbols, old_no)
            old_no += 1
        else:
            # -U0 下不该出现上下文行；防御性处理，两侧都前进
            owner, old_no, new_no = None, old_no + 1, new_no + 1

        if index == 0:
            cuts.append(0)
        elif owner is not None and current_name is not None and owner.name != current_name:
            cuts.append(index)

        # 空隙行（owner 为 None）不改 current_name，等于归给当前子 hunk
        if owner is not None:
            current_name = owner.name

    if len(cuts) == 1:
        return [hunk]

    # 第二遍：按切分点切出子 hunk，并用原 hunk 的游标算出各自的行号范围
    bounds = [*cuts, len(body)]
    return [_sub_hunk(hunk, body, bounds[i], bounds[i + 1] - 1) for i in range(len(cuts))]


def _sub_hunk(hunk: Hunk, body: list[str], start: int, end: int) -> Hunk:
    """由原 hunk 的一段正文切出子 hunk。

    Args:
        hunk: 原 hunk（提供起始行号）
        body: 原 hunk 的正文行
        start: 该段在正文中的起始下标（含）
        end: 该段在正文中的结束下标（含）

    Returns:
        子 hunk
    """
    before = body[:start]
    segment = body[start : end + 1]
    return Hunk(
        old_start=hunk.old_start + sum(1 for line in before if line.startswith("-")),
        old_lines=sum(1 for line in segment if line.startswith("-")),
        new_start=hunk.new_start + sum(1 for line in before if line.startswith("+")),
        new_lines=sum(1 for line in segment if line.startswith("+")),
        content="\n".join(segment),
    )


def _owner_at(symbols: list[Symbol], line_no: int) -> Symbol | None:
    """取包含指定行的符号。

    Args:
        symbols: 文件符号表
        line_no: 行号（1-based）

    Returns:
        命中的符号；没有则返回 None
    """
    for symbol in symbols:
        if symbol.start_line <= line_no <= symbol.end_line:
            return symbol
    return None


def _attribute(
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    base_commit: pygit2.Commit,
    head_commit: pygit2.Commit,
    path: str,
    hunk: Hunk,
) -> list[Symbol]:
    """把单个 hunk 归属到符号。

    有新增行 → 新文件侧；纯删除 → 旧文件侧；两侧都没有改动行 → 空。
    文档注释已经并进实体的行号范围（见 extractor._attach_comment_blocks），
    所以注释行直接就能命中，不需要额外落位。

    Args:
        cache: 文件符号表缓存
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
        return _containing(_file_symbols(cache, repo, head_commit, path, side=False), added)
    if removed:
        return _containing(_file_symbols(cache, repo, base_commit, path, side=True), removed)
    return []


def _file_symbols(
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    commit: pygit2.Commit,
    path: str,
    side: bool,
) -> list[Symbol]:
    """取某个文件在指定侧的符号表，带缓存。

    Args:
        cache: 缓存字典
        repo: pygit2 仓库
        commit: 目标 commit
        path: 文件路径
        side: True 表示旧版本侧

    Returns:
        该文件的符号列表；文件不存在时为空列表
    """
    key = (path, side)
    if key not in cache:
        source = read_blob(repo, commit, path)
        if source is None:
            logger.debug("文件 %s 在该版本中不存在，符号表为空", path)
            cache[key] = []
        else:
            cache[key] = extract_symbols(source)
    return cache[key]


def _containing(symbols: list[Symbol], lines: list[int]) -> list[Symbol]:
    """取出包含任一给定行的符号。

    Args:
        symbols: 候选符号
        lines: 改动行号列表

    Returns:
        命中的符号列表，保持入参顺序
    """
    return [s for s in symbols if any(s.start_line <= line <= s.end_line for line in lines)]
