"""分组相关的数据模型。"""

from dataclasses import dataclass

from prism.models.ast import HunkWithSymbol


@dataclass
class Group:
    """一个审查单元。

    Attributes:
        group_id: 分组 ID（如 "group-001"）
        package_path: **首片所在的**包路径（如 "internal/agent"）。装箱按包排序，
            所以一组可能跨包（几个小包合在一起，或一个包的尾巴接下一个包的头）——
            给人看的完整形态见 `core/grouping/packages.py` 的 `package_label`
        files: 文件列表（组内 hunk 涉及的路径，升序去重）
        hunks: 带符号的 hunk 列表（按文件路径 + 行号升序）
    """

    group_id: str
    package_path: str
    files: list[str]
    hunks: list[HunkWithSymbol]
