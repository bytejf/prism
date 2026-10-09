"""hunk → 符号映射。

用 tree-sitter 提取源码里的符号行号范围，与 hunk 的**改动行**求交，
把每个 hunk 归属到包含其改动行的符号上。

diff 用 `-U0` 生成（见 `loader.CONTEXT_LINES`），hunk 里只有改动行，
所以行号直接来自头字段。归属判定用改动行而不是 hunk 范围，是因为
按整段范围求交会把改动其实不在里面的相邻符号也挂上；用改动行后
只有真正包含改动的符号才会命中。
"""

import logging
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import pygit2

from prism.core.ast.extractor import (
    KIND_CONST,
    KIND_FILE,
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
    KIND_FILE: "整文件",
}


class _AttributionSide(NamedTuple):
    """一次归属该看哪一侧的符号表。

    Attributes:
        changed: 该侧的改动行号（1-based）
        path: 该侧的路径。**改名时两侧不同** —— 被审查版本里是新名字，
            基准版本里还是旧名字，查旧版符号表必须用旧路径
        commit: 该侧的 commit
        is_old: 是否为旧版侧
    """

    changed: list[int]
    path: str
    commit: pygit2.Commit
    is_old: bool


def map_hunks_to_symbols(
    repo_path: str | Path,
    base: str,
    head: str,
    diffs: list[Diff],
) -> list[HunkWithSymbol]:
    """把每个 hunk 归属到包含其改动行的符号。

    先按实体边界把 hunk 拆开（`split_by_symbol`），再逐个子 hunk 归属，
    最后丢掉没有符号的那些 —— 所以**返回的每个 hunk 都恰好归属一个符号**，
    不会出现「一个 hunk 挂着一串符号」或「无归属」这两种含糊状态。
    有新增行时用新文件侧（新文件源码 + 新增行号），纯删除时用旧文件侧
    （merge-base 源码 + 删除行号）—— 与文档「删除用旧文件的 AST」一致。
    编号按文件路径排序、拆分后全局连续分配，保证同一输入产生同一编号。

    Args:
        repo_path: 仓库路径
        base: base 版本（分支名 / SHA）
        head: head 版本（分支名 / SHA）
        diffs: 待映射的 diff 列表

    Returns:
        HunkWithSymbol 列表，按文件路径、文件内 hunk 顺序排列；
        每个元素的 symbol 恰好一个

    Raises:
        ValueError: 仓库无法打开，或版本无法解析
    """
    repo, diff_base, head_commit = resolve_range(repo_path, base, head)

    # 同一文件同一侧只读一次：(路径, 是否旧版本) → 符号表
    cache: dict[tuple[str, bool], list[Symbol]] = {}
    results: list[HunkWithSymbol] = []
    counter = 0
    # 丢弃分两类记，**别合成一个数**：一类是正常噪声，另一类是路径对不上
    # 的信号（文件改名后仍拿新路径去查旧版符号表就属于这类，曾导致
    # 「重命名 + 纯删除」的 hunk 被静默丢掉）。合成一个数时，
    # 后者的日志会写成前者的理由，排障直接跑偏。
    outside_entity = 0
    no_symbol_table = 0

    for diff in sorted(diffs, key=lambda item: item.path):
        # 改名：两侧路径不同且既非新增也非删除。提示词里要告诉模型，
        # 好让它去查调用方 / 文档里还写着旧名字的残留
        renamed_from = (
            diff.old_path
            if not diff.is_new and not diff.is_deleted and diff.old_path != diff.new_path
            else None
        )

        for hunk in diff.hunks:
            for part in _split_hunk(hunk, diff, cache, repo, diff_base, head_commit):
                side = _attribution_side(part, diff, diff_base, head_commit)
                if side is None:
                    no_symbol_table += 1
                    logger.warning("hunk 两侧都没有改动行，无法归属：%s", diff.path)
                    continue
                symbols = _file_symbols(cache, repo, side.commit, side.path, side.is_old)
                if not symbols:
                    no_symbol_table += 1
                    logger.warning(
                        "未归类 hunk：%s 在%s版里没有符号表（路径对不上？）",
                        side.path,
                        "旧" if side.is_old else "新",
                    )
                    continue

                symbol = _first_owner(symbols, side.changed)
                # 改动只涉及 license 头这类「与声明隔了空行的独立注释」或只涉及
                # 空行 —— 没有可审的实体，送进 LLM 只是噪音。实测 8 个 OCR
                # 分支 333 个 hunk 里 0 次命中，属于防御性过滤。
                if symbol is None:
                    outside_entity += 1
                    continue

                counter += 1
                results.append(
                    HunkWithSymbol(
                        part=part,
                        file_path=diff.path,
                        symbol=symbol,
                        hunk_id=f"{HUNK_ID_PREFIX}{counter:0{HUNK_ID_WIDTH}d}",
                        head_lines=_head_lines(cache, repo, head_commit, diff.path, symbol),
                        renamed_from=renamed_from,
                        deleted=diff.is_deleted,
                    )
                )

    if outside_entity:
        logger.info("丢弃 %d 个 hunk：改动落在所有实体之外（独立注释或空行）", outside_entity)
    if no_symbol_table:
        logger.warning("丢弃 %d 个 hunk：该侧查不到符号表，请检查路径", no_symbol_table)
    dropped = outside_entity + no_symbol_table
    logger.info("映射完成：%d 个 hunk（原始 %d 个）", len(results), len(results) + dropped)
    return results


def _head_lines(
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    head_commit: pygit2.Commit,
    path: str,
    symbol: Symbol,
) -> tuple[int, int] | None:
    """按名字在被审查版本的符号表里重查归属符号的行号范围。

    不能直接用 `symbol.start_line/end_line`：纯删除的 hunk 归属取自**旧版**
    符号表，行号与被审查版本对不上（实测差值正好等于删除行数），而
    `file_read` 读的是被审查版本 —— 拿旧坐标去读就是读到隔壁代码。

    Args:
        cache: 文件符号表缓存
        repo: pygit2 仓库
        head_commit: 被审查版本的 commit
        path: 文件在被审查版本中的路径
        symbol: 归属的符号（可能来自旧版符号表）

    Returns:
        (起始行, 结束行)；同名符号不在被审查版本里（被整体删除或改名）时返回 None
    """
    for candidate in _file_symbols(cache, repo, head_commit, path, side=False):
        if candidate.name == symbol.name:
            return candidate.start_line, candidate.end_line
    logger.debug("符号 %s 不在被审查版本 %s 中，不给行号范围", symbol.name, path)
    return None


def _attribution_side(
    part: Hunk,
    diff: Diff,
    base_commit: pygit2.Commit,
    head_commit: pygit2.Commit,
) -> _AttributionSide | None:
    """挑出该 hunk 片归属要看哪一侧。

    有新增行 → 新版侧；纯删除 → 旧版侧。**旧版侧必须用 `diff.old_path`**：
    文件改过名时它在旧版树里还叫原来的名字，用新名字去查会得到空符号表，
    整个 hunk 被当成「无归属」丢掉。

    Args:
        part: 待归属的 hunk 片
        diff: 该文件在两侧的路径与状态
        base_commit: 基准版本的 commit
        head_commit: 被审查版本的 commit

    Returns:
        该看哪一侧；两侧都没有改动行时返回 None
    """
    added, removed = changed_lines(part)
    if added:
        return _AttributionSide(added, diff.path, head_commit, False)
    if removed:
        return _AttributionSide(removed, diff.old_path, base_commit, True)
    return None


def _split_hunk(
    hunk: Hunk,
    diff: Diff,
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    base_commit: pygit2.Commit,
    head_commit: pygit2.Commit,
) -> list[Hunk]:
    """按实体边界拆分单个 hunk。

    两侧符号表都给：`+` 行按新文件行号、`-` 行按旧文件行号各自查自己的表。
    两侧的路径从 `diff` 分别取 —— 改名时它们不是同一个路径。

    Args:
        hunk: 待拆分的 hunk
        diff: 该文件的 diff（提供两侧路径）
        cache: 文件符号表缓存
        repo: pygit2 仓库
        base_commit: merge-base 侧的 commit
        head_commit: head 侧的 commit

    Returns:
        子 hunk 列表
    """
    return split_by_symbol(
        hunk,
        new_symbols=_file_symbols(cache, repo, head_commit, diff.path, side=False),
        old_symbols=_file_symbols(cache, repo, base_commit, diff.old_path, side=True),
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
    start, end = display_range(item.part)
    span = str(start) if start == end else f"{start}-{end}"
    return f"{item.hunk_id} {item.file_path}:{span}"


def format_symbol(symbol: Symbol) -> str:
    """渲染 hunk 切片的归属符号文本。

    Args:
        symbol: 归属的符号

    Returns:
        ``(*Agent).Run（方法）`` 形式
    """
    return f"{symbol.name}（{KIND_LABELS.get(symbol.kind, symbol.kind)}）"


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


def _first_owner(symbols: list[Symbol], changed: list[int]) -> Symbol | None:
    """取第一处落在实体里的改动行所属的符号。

    同一片内所有改动行必然落在同一个实体里（见 split_by_symbol），所以取
    **第一个有归属的**改动行即可 —— 不能只看第一个改动行：切片开头可能
    先是空隙行（空行、独立注释），后面才是有归属的改动。
    文档注释已经并进实体的行号范围（见 extractor._attach_comment_blocks），
    所以注释行直接就能命中，不需要额外落位。

    Args:
        symbols: 该侧的符号表
        changed: 该侧的改动行号

    Returns:
        归属的符号；所有改动行都落在实体之外时返回 None
    """
    for line_no in changed:
        if (symbol := _owner_at(symbols, line_no)) is not None:
            return symbol
    return None


def _file_symbols(
    cache: dict[tuple[str, bool], list[Symbol]],
    repo: pygit2.Repository,
    commit: pygit2.Commit,
    path: str,
    side: bool,
) -> list[Symbol]:
    """取某个文件在指定侧的符号表，带缓存。

    **非 .go 的可审文件（目前只有 `go.mod`）合成一个整文件符号**：它没有
    Go 符号可归，不给符号就会在下面被当成「查不到符号表」丢掉（那正是
    go.mod 上的参考此前结构上不可达的原因）。整文件当符号之后，
    `lines` 就是全文件范围，模型照样读得到。

    **只对非 .go 生效**：`.go` 文件解析不出符号是「路径对不上」的信号
    （文件改名后仍拿新路径查旧版符号表），那个 WARNING 必须留着。

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
            symbols = extract_symbols(source)
            if not symbols and PurePosixPath(path).suffix != ".go":
                symbols = [
                    Symbol(
                        name=PurePosixPath(path).name,
                        kind=KIND_FILE,
                        start_line=1,
                        end_line=max(len(source.splitlines()), 1),
                    )
                ]
            cache[key] = symbols
    return cache[key]
