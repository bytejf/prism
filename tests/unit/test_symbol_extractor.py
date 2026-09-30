"""Go 符号提取的单元测试。

每个用例的源码都以 `package main` 开头，所以期望里都带一个 package 符号 ——
包声明也是可归属的符号（改包名、改包的文档注释都要能落到它上面）。
"""

from prism.core.ast.extractor import (
    KIND_CONST,
    KIND_FUNCTION,
    KIND_IMPORT,
    KIND_INTERFACE,
    KIND_METHOD,
    KIND_PACKAGE,
    KIND_STRUCT,
    KIND_TYPE,
    KIND_VAR,
    extract_symbols,
)

GO_SOURCE = """package main

func Foo(a int) error {
\treturn nil
}

func (s *Service) Run() {
\tgo s.work()
}

func (s Service) Value() {}
"""


def names(source: str) -> list[str]:
    """取源码里所有符号的名字。

    Args:
        source: Go 源码

    Returns:
        符号名列表，按源码顺序
    """
    return [s.name for s in extract_symbols(source)]


def kinds(source: str) -> list[str]:
    """取源码里所有符号的类型。

    Args:
        source: Go 源码

    Returns:
        符号类型列表，按源码顺序
    """
    return [s.kind for s in extract_symbols(source)]


def test_extract_package_clause() -> None:
    """测试提取包声明。"""
    symbols = extract_symbols("package main\n\nfunc A() {}\n")
    assert [(s.name, s.kind, s.start_line, s.end_line) for s in symbols] == [
        ("package", KIND_PACKAGE, 1, 1),
        ("A", KIND_FUNCTION, 3, 3),
    ]


def test_extract_function_and_methods() -> None:
    """测试提取普通函数、指针接收者方法与值接收者方法。"""
    assert names(GO_SOURCE) == ["package", "Foo", "(*Service).Run", "(Service).Value"]
    assert kinds(GO_SOURCE) == [
        KIND_PACKAGE,
        KIND_FUNCTION,
        KIND_METHOD,
        KIND_METHOD,
    ]


def test_extract_line_ranges() -> None:
    """测试行号范围（1-based，含首尾）。"""
    symbols = extract_symbols(GO_SOURCE)
    assert (symbols[0].start_line, symbols[0].end_line) == (1, 1)  # package
    assert (symbols[1].start_line, symbols[1].end_line) == (3, 5)  # Foo
    assert (symbols[2].start_line, symbols[2].end_line) == (7, 9)  # (*Service).Run
    assert (symbols[3].start_line, symbols[3].end_line) == (11, 11)  # (Service).Value


def test_extract_unnamed_receiver() -> None:
    """测试省略接收者变量名的方法。"""
    assert names("package main\n\nfunc (*Ptr) Unnamed() {}\n") == ["package", "(*Ptr).Unnamed"]


def test_function_literal_is_not_a_symbol() -> None:
    """测试函数体内的闭包不算符号。"""
    source = """package main

func Outer() {
\tinner := func() {
\t\tprintln("x")
\t}
\tinner()
}
"""
    assert names(source) == ["package", "Outer"]


def test_multiple_symbols_are_sorted_by_source_order() -> None:
    """测试符号按源码顺序返回。"""
    source = """package main

func A() {}

func B() {}

func C() {}
"""
    assert names(source) == ["package", "A", "B", "C"]


def test_no_symbols() -> None:
    """测试没有任何声明的源码只返回包声明。"""
    assert names("package main\n") == ["package"]
    assert extract_symbols("") == []


def test_broken_source_does_not_raise() -> None:
    """测试语法不完整的源码不抛异常。"""
    assert extract_symbols("package main\n\nfunc Broken( {\n") is not None


def test_extract_from_large_source_with_multibyte_comments() -> None:
    """测试大文件 + 多字节注释。

    回归用例：tree-sitter 运行时与 tree-sitter-go grammar 大版本不匹配时，
    节点文本读取会读到已释放内存，文件到达一定规模后段错误（小文件侥幸不崩）。
    本用例构造一个几十 KB、含中文注释的 Go 文件来触发该路径。
    """
    chunks = []
    for index in range(60):
        chunks.append(
            f"// 方法 {index} 的中文说明，含多字节字符：①②③④⑤\n"
            f"func (r *Runner) Method{index}(ctx context.Context, a string, b []int) bool {{\n"
            f"\t// 中文注释" + "说明" * 200 + "\n"
            "\treturn true\n"
            "}\n"
        )
    source = 'package main\n\nimport "context"\n\n' + "\n".join(chunks)
    assert len(source.encode("utf-8")) > 50_000

    symbols = extract_symbols(source)
    method_symbols = [s for s in symbols if s.kind == KIND_METHOD]
    assert len(method_symbols) == 60
    # 60 个方法 + package + import
    assert len(symbols) == 62
    assert method_symbols[0].name == "(*Runner).Method0"
    assert method_symbols[-1].name == "(*Runner).Method59"


# ---------- 顶层声明：package / import / const / var / type ----------


def test_extract_import_block() -> None:
    """测试提取 import 块。"""
    source = 'package main\n\nimport (\n\t"errors"\n\t"fmt"\n)\n'
    symbols = extract_symbols(source)
    assert [(s.name, s.kind) for s in symbols] == [
        ("package", KIND_PACKAGE),
        ("import", KIND_IMPORT),
    ]
    assert (symbols[1].start_line, symbols[1].end_line) == (3, 6)


def test_extract_const_and_var_blocks() -> None:
    """测试提取 const / var 块。"""
    source = """package main

const (
	A = 1
)

var (
	B = 2
)
"""
    assert [(s.name, s.kind) for s in extract_symbols(source)] == [
        ("package", KIND_PACKAGE),
        ("const", KIND_CONST),
        ("var", KIND_VAR),
    ]


def test_extract_struct_and_interface() -> None:
    """测试提取结构体与接口，且不把字段 / 方法签名当作符号。"""
    source = """package main

type Config struct {
	Name string
}

type Reader interface {
	Read(id string) error
}
"""
    symbols = extract_symbols(source)
    assert [(s.name, s.kind) for s in symbols] == [
        ("package", KIND_PACKAGE),
        ("Config", KIND_STRUCT),
        ("Reader", KIND_INTERFACE),
    ]
    assert (symbols[1].start_line, symbols[1].end_line) == (3, 5)


def test_extract_type_group_produces_one_symbol_per_spec() -> None:
    """测试分组类型声明（type ( ... )）产生多个符号。"""
    source = """package main

type (
	ID   int
	Name string
)
"""
    assert [(s.name, s.kind) for s in extract_symbols(source)] == [
        ("package", KIND_PACKAGE),
        ("ID", KIND_TYPE),
        ("Name", KIND_TYPE),
    ]


def test_extract_type_alias_is_kind_type() -> None:
    """测试类型别名归为 KIND_TYPE。"""
    assert [(s.name, s.kind) for s in extract_symbols("package main\n\ntype Alias = int\n")] == [
        ("package", KIND_PACKAGE),
        ("Alias", KIND_TYPE),
    ]
