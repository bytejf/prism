"""AST 相关的数据模型。"""

from dataclasses import dataclass

from prism.models.diff import Hunk


@dataclass
class Symbol:
    """函数/方法的符号信息。

    Attributes:
        name: 函数名；方法为 ``(*Receiver).Name`` 形式
        kind: 符号类型（function / method）
        start_line: 起始行（1-based，含）
        end_line: 结束行（1-based，含）
    """

    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class HunkWithSymbol:
    """带归属符号的 hunk。

    Attributes:
        hunk: 原始 hunk
        file_path: 文件路径
        symbols: 归属的符号（可能多个，无法归属时为空）
        hunk_id: hunk 编号（如 "H-01"），全局连续分配
    """

    hunk: Hunk
    file_path: str
    symbols: list[Symbol]
    hunk_id: str
