"""用 tree-sitter 从 Go 源码中提取函数 / 方法符号。"""

import logging

import tree_sitter_go
from tree_sitter import Language, Node, Parser

from prism.models.ast import Symbol

logger = logging.getLogger(__name__)

# 提取的 AST 节点类型
FUNCTION_NODE = "function_declaration"
METHOD_NODE = "method_declaration"

# 符号类型，与 models.ast.Symbol.kind 取值一致
KIND_FUNCTION = "function"
KIND_METHOD = "method"

_parser = Parser(Language(tree_sitter_go.language()))


def extract_symbols(source: str) -> list[Symbol]:
    """从 Go 源码中提取函数与方法符号。

    只提取顶层声明：不进入函数体，因此闭包（func literal）不会被误当作符号。

    Args:
        source: Go 源码文本

    Returns:
        Symbol 列表，按源码出现顺序排列；无符号时返回空列表
    """
    tree = _parser.parse(source.encode("utf-8", errors="replace"))

    symbols: list[Symbol] = []
    stack: list[Node] = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == FUNCTION_NODE:
            symbols.append(_to_symbol(node, KIND_FUNCTION))
        elif node.type == METHOD_NODE:
            symbols.append(_to_symbol(node, KIND_METHOD))
        else:
            stack.extend(node.named_children)

    # 用显式栈遍历，顺序不保证，按起始行排序还原源码顺序
    symbols.sort(key=lambda symbol: symbol.start_line)
    return symbols


def _to_symbol(node: Node, kind: str) -> Symbol:
    """把 tree-sitter 节点转成 Symbol。

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
