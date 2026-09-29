"""diff 相关的数据模型。"""

from dataclasses import dataclass


@dataclass
class Hunk:
    """单个 hunk，含行号范围和内容。

    Attributes:
        old_start: 旧文件起始行号
        old_lines: 旧文件行数
        new_start: 新文件起始行号
        new_lines: 新文件行数
        content: 原始 hunk 文本（含 @@ 行）
    """

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str


@dataclass
class Diff:
    """单个文件的 diff。

    Attributes:
        old_path: 旧文件路径（新增文件为 /dev/null）
        new_path: 新文件路径（删除文件为 /dev/null）
        hunks: hunk 列表（二进制文件为空）
        insertions: 新增行数
        deletions: 删除行数
        is_new: 是否新增文件
        is_deleted: 是否删除文件
        is_binary: 是否二进制文件
        is_symlink: 变更后是否为符号链接
    """

    old_path: str
    new_path: str
    hunks: list[Hunk]
    insertions: int
    deletions: int
    is_new: bool
    is_deleted: bool
    is_binary: bool
    is_symlink: bool = False

    @property
    def path(self) -> str:
        """本 diff 的规范路径：删除文件取 old_path，其余取 new_path。"""
        return self.old_path if self.is_deleted else self.new_path
