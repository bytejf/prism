"""文件过滤相关的数据模型。"""

from dataclasses import dataclass

from prism.models.diff import Diff


@dataclass
class SkippedFile:
    """被跳过的文件。

    Attributes:
        path: 文件路径
        reason: 跳过原因，格式为 ``<原因码>: <说明>``。
            原因码取 binary / symlink / extension_not_allowed / builtin_noise
    """

    path: str
    reason: str


@dataclass
class FilterResult:
    """过滤结果。

    Attributes:
        reviewable: 可审查文件
        skipped: 跳过文件（含原因）
    """

    reviewable: list[Diff]
    skipped: list[SkippedFile]
