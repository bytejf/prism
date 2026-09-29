"""文件过滤。

三级过滤，按顺序命中即跳过：
1. 硬性排除 —— 二进制文件、符号链接
2. 扩展名白名单 —— 只保留 .go
3. 内置降噪 —— *_test.go、*.pb.go
"""

import logging
from pathlib import PurePosixPath

from prism.models.diff import Diff
from prism.models.filter import FilterResult, SkippedFile

logger = logging.getLogger(__name__)

# V1 只审 Go 文件，不做用户配置
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".go"})

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


def _skip_reason(diff: Diff) -> str | None:
    """判断单个文件应被跳过的原因。

    Args:
        diff: 单个文件的 diff

    Returns:
        跳过原因；不应跳过时返回 None
    """
    if diff.is_binary:
        return REASON_BINARY
    if diff.is_symlink:
        return REASON_SYMLINK

    name = PurePosixPath(diff.path).name
    suffix = PurePosixPath(name).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        return f"{REASON_EXTENSION}: {suffix or '无扩展名'}"

    for noise_suffix, label in BUILTIN_NOISE.items():
        if name.endswith(noise_suffix):
            return f"{REASON_BUILTIN_NOISE}: {label}"
    return None
