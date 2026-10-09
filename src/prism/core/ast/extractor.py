"""用 tree-sitter 从 Go 源码中提取可归属的符号。

提取范围分两层：
- 函数 / 方法：hunk 落在函数体内时归属到它
- 顶层声明：import / const / var 块与 type 声明。没有这层的话，改动落在
  包级常量、结构体字段、接口方法里时只能笼统报「<顶层>」，信息量太低。

只提取顶层声明：不进入函数体与类型体，因此闭包、结构体字段都不会被误当作符号。
"""

import logging

import tree_sitter_go
from tree_sitter import Language, Node, Parser

from prism.models.ast import Symbol

logger = logging.getLogger(__name__)

# 提取的 AST 节点类型
FUNCTION_NODE = "function_declaration"
METHOD_NODE = "method_declaration"
PACKAGE_NODE = "package_clause"
IMPORT_NODE = "import_declaration"
COMMENT_NODE = "comment"
CONST_NODE = "const_declaration"
VAR_NODE = "var_declaration"
# 类型声明：定义用 type_spec，别名用 type_alias，两者都有 name / type 字段
TYPE_NODES = frozenset({"type_spec", "type_alias"})

# 符号类型，与 models.ast.Symbol.kind 取值一致
KIND_PACKAGE = "package"
KIND_FUNCTION = "function"
KIND_METHOD = "method"
KIND_IMPORT = "import"
KIND_CONST = "const"
KIND_VAR = "var"
KIND_STRUCT = "struct"
KIND_INTERFACE = "interface"
KIND_TYPE = "type"

# **不是提取出来的**，由 mapper 给「没有符号可归的文件」（如 go.mod）合成：
# 整个文件当作一个符号。放在这里是为了让 kind 常量只有一个出处。
KIND_FILE = "file"

# 顶层声明块节点 → 符号类型
BLOCK_KINDS: dict[str, str] = {
    PACKAGE_NODE: KIND_PACKAGE,
    IMPORT_NODE: KIND_IMPORT,
    CONST_NODE: KIND_CONST,
    VAR_NODE: KIND_VAR,
}

# type_spec 的 type 字段类型 → 符号类型，其余归为 KIND_TYPE
TYPE_KINDS: dict[str, str] = {
    "struct_type": KIND_STRUCT,
    "interface_type": KIND_INTERFACE,
}

_parser = Parser(Language(tree_sitter_go.language()))


def extract_symbols(source: str) -> list[Symbol]:
    """从 Go 源码中提取可归属的符号（带文档注释）。

    提取函数 / 方法与顶层声明（package / import / const / var 块、type 声明）。
    不进入函数体，因此闭包（func literal）不会被误当作符号。

    注释在语法树里是独立节点，不挂在它描述的声明下面，所以这里按 Go 的
    文档注释约定把它们并进紧邻实体的行号范围（见 `_attach_comment_blocks`）。

    Args:
        source: Go 源码文本

    Returns:
        Symbol 列表，按源码出现顺序排列；无符号时返回空列表
    """
    tree = _parser.parse(source.encode("utf-8", errors="replace"))
    root = tree.root_node

    symbols: list[Symbol] = []
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.type == FUNCTION_NODE:
            symbols.append(_to_symbol(node, KIND_FUNCTION))
        elif node.type == METHOD_NODE:
            symbols.append(_to_symbol(node, KIND_METHOD))
        elif node.type in BLOCK_KINDS:
            symbols.append(_block_symbol(node, BLOCK_KINDS[node.type]))
        elif node.type in TYPE_NODES:
            symbols.append(_type_symbol(node))
        else:
            stack.extend(node.named_children)

    # 用显式栈遍历，顺序不保证，按起始行排序还原源码顺序
    symbols.sort(key=lambda symbol: symbol.start_line)
    return _attach_comment_blocks(symbols, _top_level_comment_blocks(root))


def _top_level_comment_blocks(root: Node) -> list[tuple[int, int]]:
    """把顶层的连续注释行合成注释块。

    tree-sitter 不会合并相邻注释：`//` 连续 3 行是 3 个独立节点，
    而块注释 `/* ... */` 跨行本身就是 1 个节点。Go 的 ast.CommentGroup
    定义是「相邻且无空行的注释序列」（不限 `//` 还是 `/* */`），
    所以这里按行号相邻来合并：前一节点 end_line + 1 == 后一节点 start_line。

    只取顶层（source_file 的直接子节点）且**独占一行**的注释：
    - 实体内部的注释本来就在实体的行号范围内，不需要并
    - 行尾注释（start_column > 0，如 `const X = 1 // 说明`）与所在声明
      同行号，也已经在范围内；若不排除，它会与上方的文档注释块行号相邻
      而被错误合并，把整个块的末行推到声明之后，rule 1 就永远算不中

    Args:
        root: source_file 节点

    Returns:
        [(起始行, 结束行)]，按出现顺序
    """
    blocks: list[list[int]] = []
    for node in root.named_children:
        if node.type != COMMENT_NODE or node.start_point.column != 0:
            continue
        start, end = node.start_point.row + 1, node.end_point.row + 1
        if blocks and blocks[-1][1] + 1 == start:
            blocks[-1][1] = end
        else:
            blocks.append([start, end])
    return [(start, end) for start, end in blocks]


def _attach_comment_blocks(symbols: list[Symbol], blocks: list[tuple[int, int]]) -> list[Symbol]:
    """把注释块并进紧邻实体的行号范围。

    规则（按优先级）：
    1. 注释块**末行 + 1** == 某实体**首行** → 并进该实体（它是文档注释）
    2. 否则注释块**首行 - 1** == 某实体**末行** → 并进该实体（它是后置说明）
    3. 都不满足 → 保持独立（如 license 头，它与下面的声明之间隔了空行）

    只改行号范围，name / kind 不变。

    Args:
        symbols: 已提取的实体，按起始行排序
        blocks: 注释块列表

    Returns:
        行号范围可能被扩展的符号列表
    """
    starts = {symbol.start_line: index for index, symbol in enumerate(symbols)}
    ends = {symbol.end_line: index for index, symbol in enumerate(symbols)}
    new_starts = [symbol.start_line for symbol in symbols]
    new_ends = [symbol.end_line for symbol in symbols]

    for block_start, block_end in blocks:
        if (index := starts.get(block_end + 1)) is not None:
            new_starts[index] = block_start
        elif (index := ends.get(block_start - 1)) is not None:
            new_ends[index] = block_end

    return [
        Symbol(
            name=symbol.name,
            kind=symbol.kind,
            start_line=new_starts[index],
            end_line=new_ends[index],
        )
        for index, symbol in enumerate(symbols)
    ]


def _to_symbol(node: Node, kind: str) -> Symbol:
    """把函数 / 方法节点转成 Symbol。

    Args:
        node: function_declaration 或 method_declaration 节点
        kind: 符号类型

    Returns:
        Symbol 对象
    """
    name_node = node.child_by_field_name("name")
    name = _text(name_node) if name_node is not None else "<匿名>"

    if kind == KIND_METHOD:
        name = f"({_receiver_type(node.child_by_field_name('receiver'))}).{name}"

    return Symbol(
        name=name,
        kind=kind,
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
    )


def _block_symbol(node: Node, kind: str) -> Symbol:
    """把顶层声明块（package / import / const / var）转成 Symbol。

    Args:
        node: package_clause / import_declaration / const_declaration / var_declaration 节点
        kind: 符号类型

    Returns:
        Symbol 对象，name 与 kind 同名
    """
    return Symbol(
        name=kind,
        kind=kind,
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
    )


def _type_symbol(node: Node) -> Symbol:
    """把类型声明（type_spec / type_alias）转成 Symbol。

    按底层类型区分结构体 / 接口 / 其他；``type ( A int; B string )``
    这样的分组声明会产生多个 spec，各自成为一个符号。

    Args:
        node: type_spec 或 type_alias 节点

    Returns:
        Symbol 对象
    """
    name_node = node.child_by_field_name("name")
    type_node = node.child_by_field_name("type")
    type_name = type_node.type if type_node is not None else ""

    return Symbol(
        name=_text(name_node) if name_node is not None else "<匿名>",
        kind=TYPE_KINDS.get(type_name, KIND_TYPE),
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
    )


def _receiver_type(receiver: Node | None) -> str:
    """取方法接收者的类型文本。

    Args:
        receiver: method_declaration 的 receiver 节点（parameter_list）

    Returns:
        类型文本，如 ``*Service``；取不到时返回 ``?``
    """
    if receiver is None:
        return "?"
    for param in receiver.named_children:
        type_node = param.child_by_field_name("type")
        if type_node is not None:
            return _text(type_node)
    return "?"


def _text(node: Node) -> str:
    """取节点的源码文本。

    Args:
        node: AST 节点

    Returns:
        节点对应文本
    """
    text = node.text
    return "" if text is None else text.decode("utf-8", errors="replace")
