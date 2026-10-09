"""文件过滤。

三级过滤，按顺序命中即跳过：
1. 硬性排除 —— 二进制文件、符号链接
2. 白名单 —— 扩展名只保留 .go，外加 `go.mod`（按**文件名**放行）
3. 内置降噪 —— *_test.go、*.pb.go

**全流程只有这一套判据**：前两级之外的路径规则（`path_skip_reason`）同时被
过滤阶段与三个读取工具使用 —— 能审的才读得到，读得到的才可能被审。
工具各写一份判据一定会漂，而漂的表现是「某个文件审得到但读不到」这种
无声的不一致。
"""

import logging
from pathlib import PurePosixPath

from prism.models.diff import Diff
from prism.models.filter import FilterResult, SkippedFile

logger = logging.getLogger(__name__)

# V1 只审 Go 文件，不做用户配置
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".go"})

# 按**文件名**放行的例外，不看扩展名。
#
# 这里只有 `go.mod`，依据是评测：gofr 那一局 30 条参考里有 4 条（13%）落在
# `go.mod` / 子模块 `go.mod` 的 `go` 与 `toolchain` 指令一致性上，而它们
# **结构上不可达**（见 prism-bench 的 missed_references_gofr_classified.md §D）。
# 注意后缀是 `.mod`，所以这条必须在扩展名检查**之前**判，否则会被当成非 .go 挡掉。
ALLOWED_FILENAMES: frozenset[str] = frozenset({"go.mod"})

# 内置降噪：文件名后缀 → 展示用的通配写法（保持插入顺序即匹配优先级）
BUILTIN_NOISE: dict[str, str] = {
    "_test.go": "*_test.go",
    ".pb.go": "*.pb.go",
}

# 跳过原因码
REASON_BINARY = "binary"
REASON_SYMLINK = "symlink"
REASON_EXTENSION = "extension_not_allowed"
REASON_BUILTIN_NOISE = "builtin_noise"


def filter_files(diffs: list[Diff]) -> FilterResult:
    """按规则过滤文件。

    过滤顺序：硬性排除 → 扩展名白名单 → 内置降噪。
    同一文件只记录第一个命中的原因。

    Args:
        diffs: 待过滤的 diff 列表

    Returns:
        FilterResult，含可审查文件与被跳过文件（含原因）
    """
    reviewable: list[Diff] = []
    skipped: list[SkippedFile] = []

    for diff in diffs:
        reason = _skip_reason(diff)
        if reason is None:
            reviewable.append(diff)
        else:
            skipped.append(SkippedFile(path=diff.path, reason=reason))
            logger.debug("跳过 %s（%s）", diff.path, reason)

    logger.info("过滤完成：可审查 %d 个文件，跳过 %d 个", len(reviewable), len(skipped))
    return FilterResult(reviewable=reviewable, skipped=skipped)


def path_skip_reason(path: str) -> str | None:
    """路径层面的跳过原因（扩展名白名单 + 内置降噪）；可审时返回 None。

    **全流程唯一的路径判据**：过滤阶段用它决定跳过哪些文件，三个读取工具
    （`file_read` / `code_search` / `file_read_diff`）用它决定拒掉哪些路径。
    别在工具里另写一份 —— 两份判据漂开之后，「审得到但读不到」不会报错。

    只看路径，**不看二进制 / 符号链接**那两级：那两个要文件内容或 diff 元数据，
    调用方手上不一定有（`code_search` 拿到的是 grep 出来的一行文本）。

    Args:
        path: 仓库内相对路径

    Returns:
        跳过原因；可审时返回 None
    """
    name = PurePosixPath(path).name
    if name not in ALLOWED_FILENAMES:
        suffix = PurePosixPath(name).suffix
        if suffix not in ALLOWED_EXTENSIONS:
            return f"{REASON_EXTENSION}: {suffix or '无扩展名'}"

    for noise_suffix, label in BUILTIN_NOISE.items():
        if name.endswith(noise_suffix):
            return f"{REASON_BUILTIN_NOISE}: {label}"
    return None


def is_reviewable_path(path: str) -> bool:
    """这个路径在不在审查范围内。`path_skip_reason` 的布尔形态。

    Args:
        path: 仓库内相对路径

    Returns:
        可审为 True
    """
    return path_skip_reason(path) is None


def _skip_reason(diff: Diff) -> str | None:
    """判断单个文件应被跳过的原因。

    硬性排除那两级要用 diff 的元数据，所以入口与 `path_skip_reason` 不同，
    但路径判据是**同一个函数** —— 不在这里重抄一遍扩展名与降噪规则。

    Args:
        diff: 单个文件的 diff

    Returns:
        跳过原因；不应跳过时返回 None
    """
    if diff.is_binary:
        return REASON_BINARY
    if diff.is_symlink:
        return REASON_SYMLINK
    return path_skip_reason(diff.path)
