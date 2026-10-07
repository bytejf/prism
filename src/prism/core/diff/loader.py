"""diff 加载。

用 pygit2（不启子进程）计算 merge-base(base, head)，
取出 unified diff 文本后交给 `parser.parse_diff_text` 解析。
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import pygit2

from prism.core.diff.parser import parse_diff_text
from prism.models.diff import Diff

logger = logging.getLogger(__name__)

# diff 的上下文行数。0 表示只输出改动行，等价于 `git diff -U0`。
#
# 这样 hunk 头里的 new_start/new_lines 就是「新增行的起始与数量」、
# old_start/old_lines 就是「删除行的起始与数量」，归属判定直接读头即可，
# 不必再遍历正文把上下文行和改动行分开推。
#
# 审查单元是 Group（同包文件的所有 hunk 在一个会话里审），不是单个 hunk，
# 所以 hunk 变细不影响调用 LLM 的次数，只是让报告粒度更准。
CONTEXT_LINES = 0


def resolve_commit(repo: pygit2.Repository, rev: str) -> pygit2.Commit:
    """把分支名 / 标签 / SHA 解析为 Commit 对象。

    Args:
        repo: 已打开的仓库
        rev: 版本标识，如 "main"、"HEAD"、"abc1234"

    Returns:
        解析出的 Commit

    Raises:
        ValueError: rev 不存在，或解析结果不是 commit
    """
    try:
        obj = repo.revparse_single(rev)
    except (KeyError, pygit2.GitError) as exc:
        raise ValueError(f"无法解析版本 {rev!r}: {exc}") from exc

    peeled = obj.peel(pygit2.Commit)
    if not isinstance(peeled, pygit2.Commit):
        raise ValueError(f"版本 {rev!r} 解析结果不是 commit：{obj.type_str}")
    return peeled


def resolve_range(
    repo_path: str | Path, base: str, head: str
) -> tuple[pygit2.Repository, pygit2.Commit, pygit2.Commit]:
    """打开仓库，解析 base / head 以及它们的 merge-base。

    Args:
        repo_path: 仓库路径
        base: base 版本（分支名 / SHA）
        head: head 版本（分支名 / SHA）

    Returns:
        (仓库, diff 基准 commit, head commit)。基准 commit 为 merge-base；
        无共同祖先时退化为 base 本身

    Raises:
        ValueError: 仓库无法打开，或版本无法解析
    """
    try:
        repo = pygit2.Repository(str(repo_path))
    except pygit2.GitError as exc:
        raise ValueError(f"无法打开仓库 {repo_path}: {exc}") from exc

    base_commit = resolve_commit(repo, base)
    head_commit = resolve_commit(repo, head)

    merge_base_oid = repo.merge_base(base_commit.id, head_commit.id)
    if merge_base_oid is None:
        logger.warning("base 与 head 没有共同祖先，退化为直接比较 %s 与 %s", base, head)
        return repo, base_commit, head_commit

    logger.debug("merge-base(%s, %s) = %s", base, head, merge_base_oid)
    return repo, resolve_commit(repo, str(merge_base_oid)), head_commit


def read_blob(repo: pygit2.Repository, commit: pygit2.Commit, path: str) -> str | None:
    """从指定 commit 的文件树中读取文件内容。

    Args:
        repo: 仓库
        commit: 目标 commit
        path: 仓库内相对路径

    Returns:
        文件文本；文件不存在或不是普通文件（blob）时返回 None
    """
    try:
        entry = commit.tree[path]
    except KeyError:
        return None
    if entry.type_str != "blob":
        return None

    blob = repo[entry.id]
    if not isinstance(blob, pygit2.Blob):
        return None
    return bytes(blob.data).decode("utf-8", errors="replace")


def build_diff_index(diffs: Iterable[Diff]) -> dict[str, Diff]:
    """按路径给 diff 建索引，供 `file_read_diff` 查询。

    两步走，顺序不能反：

    1. 每个文件的**规范路径**（`Diff.path`）先占位 —— 见 models/diff.py
    2. 改过名的文件再补一个**旧路径别名**：提示词里带着
       `renamed-from="old.go"`，模型很可能拿着旧名字来查，查不到就会得到
       「没有改动」这个错答案

    第 2 步用 `setdefault` 而不是直接赋值：同一次 diff 里可能既有「A 改名成 B」
    又有「新建了一个 A」。直接赋值会让新建的 A 被别名盖掉，模型问 A 拿到的是
    B 的改动 —— 一个不报错的错答案。`setdefault` 保证第 1 步建立的规范路径赢。

    Args:
        diffs: 待索引的 diff（通常是过滤后的可审查文件）

    Returns:
        `路径 → Diff` 映射，含改名文件的旧路径别名
    """
    items = list(diffs)
    index: dict[str, Diff] = {item.path: item for item in items}
    for item in items:
        if item.is_new or item.old_path == item.path:
            continue
        index.setdefault(item.old_path, item)
    logger.debug("diff 索引：%d 个文件 → %d 个路径", len(items), len(index))
    return index


def load_diffs(repo_path: str | Path, base: str, head: str) -> list[Diff]:
    """加载 base 与 head 之间、以 merge-base 为基准的 diff。

    用 merge-base 而非 base 本身，避免把 base 分支上的新提交算进 diff。

    Args:
        repo_path: 仓库路径
        base: base 版本（分支名 / SHA）
        head: head 版本（分支名 / SHA）

    Returns:
        Diff 列表

    Raises:
        ValueError: 仓库无法打开，或版本无法解析
    """
    repo, diff_base, head_commit = resolve_range(repo_path, base, head)

    diff = repo.diff(diff_base, head_commit, context_lines=CONTEXT_LINES)
    # 显式开启重命名检测：libgit2 默认走 diff.renames 配置项且不实现 git 的
    # 隐式默认值（true），不开启会把 `R old.go -> new.go` 拆成「删 + 增」两个文件
    diff.find_similar(pygit2.enums.DiffFind.FIND_RENAMES)

    diffs = parse_diff_text(diff.patch or "")

    logger.info(
        "加载 diff 完成：%d 个文件，%d 个 hunk（+%d/-%d）",
        len(diffs),
        sum(len(d.hunks) for d in diffs),
        sum(d.insertions for d in diffs),
        sum(d.deletions for d in diffs),
    )
    return diffs
