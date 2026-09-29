"""Go 符号提取的单元测试。"""

from prism.core.ast.extractor import KIND_FUNCTION, KIND_METHOD, extract_symbols

GO_SOURCE = """package main

func Foo(a int) error {
\treturn nil
}

func (s *Service) Run() {
\tgo s.work()
}

func (s Service) Value() {}
"""


def test_extract_function_and_methods() -> None:
    """测试提取普通函数、指针接收者方法与值接收者方法。"""
    symbols = extract_symbols(GO_SOURCE)
    assert [s.name for s in symbols] == ["Foo", "(*Service).Run", "(Service).Value"]
    assert [s.kind for s in symbols] == [KIND_FUNCTION, KIND_METHOD, KIND_METHOD]


def test_extract_line_ranges() -> None:
    """测试行号范围（1-based，含首尾）。"""
    symbols = extract_symbols(GO_SOURCE)
    assert (symbols[0].start_line, symbols[0].end_line) == (3, 5)
    assert (symbols[1].start_line, symbols[1].end_line) == (7, 9)
    assert (symbols[2].start_line, symbols[2].end_line) == (11, 11)


def test_extract_unnamed_receiver() -> None:
    """测试省略接收者变量名的方法。"""
    symbols = extract_symbols("package main\n\nfunc (*Ptr) Unnamed() {}\n")
    assert [s.name for s in symbols] == ["(*Ptr).Unnamed"]


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
    symbols = extract_symbols(source)
    assert [s.name for s in symbols] == ["Outer"]


def test_multiple_symbols_are_sorted_by_source_order() -> None:
    """测试符号按源码顺序返回。"""
    source = """package main

func A() {}

func B() {}

func C() {}
"""
    assert [s.name for s in extract_symbols(source)] == ["A", "B", "C"]


def test_no_symbols() -> None:
    """测试没有函数声明的源码返回空列表。"""
    assert extract_symbols("package main\n\nvar x = 1\n") == []
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
    methods = []
    for index in range(60):
        methods.append(
            f"// 方法 {index} 的中文说明，含多字节字符：①②③④⑤\n"
            f"func (r *Runner) Method{index}(ctx context.Context, a string, b []int) bool {{\n"
            f"\t// 中文注释" + "说明" * 200 + "\n"
            "\treturn true\n"
            "}\n"
        )
    source = 'package main\n\nimport "context"\n\n' + "\n".join(methods)
    assert len(source.encode("utf-8")) > 50_000

    symbols = extract_symbols(source)
    assert len(symbols) == 60
    assert symbols[0].name == "(*Runner).Method0"
    assert symbols[-1].name == "(*Runner).Method59"
