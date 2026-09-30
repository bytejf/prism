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
    """带归属符号的 hunk 切片 —— 审查的最小单位。

    Attributes:
        part: 原始 hunk 的一片。git 的 hunk 可能横跨多个实体，
            `mapper.split_by_symbol` 按实体边界切开后才是这里的一片；
            字段与 Hunk 完全一致，只是行号范围更窄
        file_path: 文件路径
        symbol: 归属的符号。恒为一个 —— 拆分保证同一片只落在一个实体里，
            无归属的片（改动只涉及独立注释或空行）在映射阶段就被丢弃了
        hunk_id: hunk 编号（如 "H-01"），拆分后全局连续分配
    """

    part: Hunk
    file_path: str
    symbol: Symbol
    hunk_id: str
