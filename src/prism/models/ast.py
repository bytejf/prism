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
        file_path: 文件路径。**用被审查版本的路径** —— 改名后是新名字，
            整体删除时是它原来的名字
        symbol: 归属的符号。恒为一个 —— 拆分保证同一片只落在一个实体里，
            无归属的片（改动只涉及独立注释或空行）在映射阶段就被丢弃了
        hunk_id: hunk 编号（如 "H-01"），拆分后全局连续分配
        head_lines: 归属符号在**被审查版本**里的完整行号范围（闭区间）。
            提示词靠它给模型指路 —— `file_read` 读的就是这一版，两者必须同坐标。
            归属本身可能取自旧版符号表（纯删除的 hunk 就是），所以这里要
            **按名字在被审查版本的符号表里重查**；查不到同名符号
            （被整体删除或改名）时为 None，提示词里就不给这个属性
        renamed_from: 该文件由这个旧路径改名而来；没改名时为 None
        deleted: 该文件在被审查版本里已被整体删除
    """

    part: Hunk
    file_path: str
    symbol: Symbol
    hunk_id: str
    head_lines: tuple[int, int] | None
    renamed_from: str | None
    deleted: bool
