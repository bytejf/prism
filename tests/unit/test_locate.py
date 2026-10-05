"""评论定位的单元测试。

守的主要是三件事，错了都**没有任何信号**：

- **行号算对** —— 偏几行不影响运行，只是指向隔壁代码
- **侧算对** —— 行号挂着正确的那一版；标错侧等于让读者去错误的文件版本里找
- **不丢评论** —— 定位失败的也照样输出

所以这里逐条钉住行号的推进规则与搜索优先级。
"""

import logging

import pytest

from prism.core.diff.parser import UNKNOWN_PATH
from prism.core.output.locate import locate, locate_group, normalize
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.comment import (
    LOCATE_FAILED,
    LOCATE_GROUP,
    LOCATE_HUNK,
    REASON_EMPTY_EXISTING_CODE,
    REASON_MULTI_MATCH,
    REASON_NO_MATCH,
    SIDE_LEFT,
    SIDE_RIGHT,
)
from prism.models.diff import Hunk
from prism.models.grouping import Group
from prism.models.tool import CommentInput

# 定位不看符号，但 HunkWithSymbol 必须带一个
SYMBOL = Symbol(name="F", kind="function", start_line=1, end_line=99)

# 定位模块的 logger 名，测试日志用
LOCATE_LOGGER = "prism.core.output.locate"


def make_hunk(
    hunk_id: str,
    content: str,
    *,
    path: str = "a.go",
    new_start: int = 10,
    old_start: int = 10,
) -> HunkWithSymbol:
    """构造带正文的 hunk 片。

    正文按 diff 原文写（带 `+` / `-` 前缀）。两侧起始行号分开给，因为它们在
    真实 diff 里经常不同（前面有过删除时就会错开）—— 两侧的行号各自从自己的
    起点数，测试里也照实分开。

    Args:
        hunk_id: hunk 编号
        content: hunk 正文（diff 原文）
        path: 文件路径
        new_start: 新增侧起始行号
        old_start: 删除侧起始行号

    Returns:
        HunkWithSymbol
    """
    body = content.splitlines()
    return HunkWithSymbol(
        part=Hunk(
            old_start=old_start,
            old_lines=sum(1 for line in body if line.startswith("-")),
            new_start=new_start,
            new_lines=sum(1 for line in body if line.startswith("+")),
            content=content,
        ),
        file_path=path,
        symbol=SYMBOL,
        hunk_id=hunk_id,
        head_lines=(SYMBOL.start_line, SYMBOL.end_line),
        renamed_from=None,
        deleted=False,
    )


def make_group(hunks: list[HunkWithSymbol]) -> Group:
    """构造测试用审查单元。

    Args:
        hunks: 组内 hunk

    Returns:
        Group 对象
    """
    return Group(
        group_id="group-001",
        package_path=".",
        files=sorted({item.file_path for item in hunks}),
        hunks=hunks,
    )


def make_input(existing_code: str, content: str = "问题", severity: str = "high") -> CommentInput:
    """构造测试用 CommentInput。

    Args:
        existing_code: 用于定位的代码片段
        content: 评论正文
        severity: 严重程度

    Returns:
        CommentInput 对象
    """
    return CommentInput(
        severity=severity,
        category="category",
        content=content,
        existing_code=existing_code,
    )


# ---------- 归一化 ----------


def test_normalize_strips_prefix_and_whitespace() -> None:
    """测试：行首的 diff 记号与首尾空白都去掉，缩进差异不影响匹配。"""
    assert normalize("+\tfoo()  ") == "foo()"
    assert normalize("-  foo()") == "foo()"
    assert normalize("  foo()  ") == "foo()"


def test_normalize_keeps_case() -> None:
    """测试：大小写不动 —— Go 里 `Err` 与 `err` 是两个东西。"""
    assert normalize("+Err") == "Err"
    assert normalize("+Err") != normalize("+err")


# ---------- 行号：两侧各数各的 ----------


def test_added_lines_number_from_new_start() -> None:
    """测试：新增侧的行号从 `new_start` 起，**删除行不占号**。

    这是行号算错的头号原因：把对面那侧的行也推进计数器，行号会整体偏移，
    而偏移后的值读起来完全正常 —— 不报错、不越界，只是指向隔壁几行。
    """
    hunk = make_hunk(
        "H-01",
        "-\tif err != nil {\n+\tif err == nil {\n+\t\treturn nil\n+\t}",
        new_start=20,
        old_start=7,
    )

    assert locate("H-01", "if err == nil {", [hunk]).start_line == 20
    assert locate("H-01", "return nil", [hunk]).start_line == 21
    assert locate("H-01", "}", [hunk]).start_line == 22


def test_deleted_lines_number_from_old_start() -> None:
    """测试：删除侧的行号从 `old_start` 起，**新增行不占号**。

    与上一条对称。两侧起点在这条用例里特意错开（7 vs 20），用错一侧会得到
    一个明显不对但看着很正常的数。
    """
    hunk = make_hunk(
        "H-01",
        "-\tif err != nil {\n-\t\treturn nil\n+\tif err == nil {",
        new_start=20,
        old_start=7,
    )

    assert locate("H-01", "-if err != nil {", [hunk]).start_line == 7
    assert locate("H-01", "-return nil", [hunk]).start_line == 8


def test_deleted_block_gives_the_whole_range() -> None:
    """测试：整段删除给出旧版本的区间与 LEFT 侧。

    纯删除的 hunk（`new_lines == 0`）在新版本里没有任何位置，这类评论以前
    只能进聚合；现在挂到被删掉的那几行上，`side` 标明行号是旧版本的坐标。
    """
    hunk = make_hunk("H-01", '-\tif key == "" {\n-\t\treturn ""\n-\t}', new_start=6, old_start=7)

    found = locate("H-01", 'if key == "" {\n\treturn ""\n}', [hunk])

    assert found.side == SIDE_LEFT
    assert (found.start_line, found.end_line) == (7, 9)
    assert found.level == LOCATE_HUNK


# ---------- 侧：命中哪一版 ----------


def test_added_code_anchors_right() -> None:
    """测试：新增的代码挂 RIGHT，行号是被审查版本的坐标。"""
    hunk = make_hunk("H-01", "-\tfoo()\n+\tbar()", new_start=8, old_start=11)

    found = locate("H-01", "bar()", [hunk])

    assert (found.side, found.start_line) == (SIDE_RIGHT, 8)


def test_deleted_code_anchors_left() -> None:
    """测试：被删的代码挂 LEFT，行号是基准版本的坐标。"""
    hunk = make_hunk("H-01", "-\tfoo()\n+\tbar()", new_start=8, old_start=11)

    found = locate("H-01", "foo()", [hunk])

    assert (found.side, found.start_line) == (SIDE_LEFT, 11)


def test_plain_snippet_prefers_the_new_side() -> None:
    """测试：没带记号的片段先试新增侧。

    两侧都能匹配时（这份 diff 是特意造的：同一行既是删的也是加的），新增侧
    胜出 —— 审查关心的是改动**之后**的状态，评论挂在那儿才对。
    """
    hunk = make_hunk("H-01", "-\tfoo()\n+\tfoo()", new_start=8, old_start=11)

    found = locate("H-01", "foo()", [hunk])

    assert (found.side, found.start_line) == (SIDE_RIGHT, 8)


def test_plain_snippet_falls_back_to_the_old_side() -> None:
    """测试：没带记号、新增侧又找不到时，落到删除侧。

    模型从 diff 抄的时候可能顺手把 `-` 抹掉。抹掉了也照样该定位到被删的那行，
    而不是进聚合。
    """
    hunk = make_hunk("H-01", "-\toldCall()\n+\tnewCall()", new_start=8, old_start=11)

    found = locate("H-01", "oldCall()", [hunk])

    assert (found.side, found.start_line) == (SIDE_LEFT, 11)


def test_mixed_snippet_anchors_the_added_part() -> None:
    """测试：一半 `-` 一半 `+` 的片段挂在新增侧。

    模型想表达「把 A 换成了 B」时会连着抄两行。评论讲的是替换**之后**的代码，
    所以按 `+` 那部分定位；`-` 那部分只当上下文，不参与定案。
    """
    hunk = make_hunk("H-01", "-\toldCall()\n+\tnewCall()", new_start=8, old_start=11)

    found = locate("H-01", "-oldCall()\n+newCall()", [hunk])

    assert (found.side, found.start_line) == (SIDE_RIGHT, 8)


def test_deleted_needle_is_tried_before_plain_on_the_old_side() -> None:
    """测试：删除侧内部，带 `-` 记号的片段比不带记号的先试。

    两个 needle 在这份 diff 里都能命中删除侧（`first()` 在 old 11、`second()`
    在 old 12），模型明确标了 `-` 的那个更可信，所以结果该是 12 而不是 11。
    """
    hunk = make_hunk("H-01", "-\tfirst()\n+\tadded()\n-\tsecond()", new_start=8, old_start=11)

    found = locate("H-01", "-second()\nfirst()", [hunk])

    assert (found.side, found.start_line) == (SIDE_LEFT, 12)


# ---------- 匹配的其它细节 ----------


def test_locate_multiline_range() -> None:
    """测试：多行片段给出首尾行号。"""
    hunk = make_hunk("H-01", "+\tfunc f() {\n+\t\treturn nil\n+\t}")

    found = locate("H-01", "func f() {\n\treturn nil\n}", [hunk])

    assert found.level == LOCATE_HUNK
    assert (found.start_line, found.end_line) == (10, 12)


def test_locate_ignores_prefix_and_indent() -> None:
    """测试：模型从源码抄（不带 `+`、缩进不同）也能命中。"""
    hunk = make_hunk("H-01", "+\t\tif err != nil {")

    found = locate("H-01", "if err != nil {", [hunk])

    assert found.level == LOCATE_HUNK
    assert found.start_line == 10


def test_locate_blank_line_does_not_break_match() -> None:
    """测试：片段中间的空行不打断连续匹配。

    新增一个空行在 diff 里是孤零零一个 `+`，归一化后是空串。让它参与比对，
    「抄了空行的那条评论」就永远匹配不上。
    """
    hunk = make_hunk("H-01", "+\ta := 1\n+\n+\tb := 2")

    found = locate("H-01", "a := 1\n\nb := 2", [hunk])

    assert (found.start_line, found.end_line) == (10, 12)


def test_multi_hit_in_hunk_takes_first() -> None:
    """测试：**同一个 hunk 内**多命中取第一个，并记下候选数。

    两个候选都在同一段改动里，取首个比把评论丢进未定位聚合更划算 —— 代价
    只是行号可能偏几行。candidates 带出去让日志能统计这条规则的触发频率。
    """
    hunk = make_hunk("H-01", "+\treturn nil\n+\treturn nil")

    found = locate("H-01", "return nil", [hunk])

    assert found.level == LOCATE_HUNK
    assert (found.start_line, found.end_line) == (10, 10)
    assert found.candidates == 2


# ---------- 第二层：组内 ----------


def test_group_hit_rewrites_hunk_id() -> None:
    """测试：组内命中要**改写 hunk_id**。

    第二层可能在另一个文件里命中，而 path 是从 hunk_id 查出来的。保留原编号
    的话，评论的 path 与 start_line 会指向不同文件 —— 两个字段各说各话，
    还都不报错。
    """
    hunks = [
        make_hunk("H-01", "+\tfoo()", path="pkg/a.go"),
        make_hunk("H-02", "+\tbar()", path="pkg/b.go", new_start=33),
    ]

    found = locate("H-01", "bar()", hunks)

    assert found.level == LOCATE_GROUP
    assert found.hunk_id == "H-02"
    assert found.start_line == 33


def test_group_hit_keeps_its_side() -> None:
    """测试：组内命中也要带上侧 —— 别组命中的可能是删除行。"""
    hunks = [
        make_hunk("H-01", "+\tfoo()", path="pkg/a.go"),
        make_hunk("H-02", "-\tbar()", path="pkg/b.go", old_start=44),
    ]

    found = locate("H-01", "-bar()", hunks)

    assert (found.level, found.side, found.start_line) == (LOCATE_GROUP, SIDE_LEFT, 44)


def test_deleted_code_in_the_own_hunk_beats_a_group_hit() -> None:
    """测试：**层优先于侧** —— 本 hunk 的删除行命中，胜过别组的删除行命中。"""
    hunks = [
        make_hunk("H-01", "-\tdup()", path="pkg/a.go", old_start=5),
        make_hunk("H-02", "-\tdup()", path="pkg/b.go", old_start=44),
    ]

    found = locate("H-01", "-dup()", hunks)

    assert (found.level, found.hunk_id, found.start_line) == (LOCATE_HUNK, "H-01", 5)


def test_group_multi_hit_gives_up() -> None:
    """测试：组内多命中放弃 —— 跨 hunk 的两个候选可能是无关代码。"""
    hunks = [
        make_hunk("H-01", "+\tfoo()", path="pkg/a.go"),
        make_hunk("H-02", "+\tbar()", path="pkg/b.go"),
        make_hunk("H-03", "+\tbar()", path="pkg/c.go"),
    ]

    found = locate("H-01", "bar()", hunks)

    assert found.level == LOCATE_FAILED
    assert found.reason == REASON_MULTI_MATCH
    assert found.hunk_id == "H-01", "未定位时保留模型原报的编号"


def test_no_match_fails() -> None:
    """测试：两层都没中 → 未定位。"""
    hunks = [make_hunk("H-01", "+\tfoo()")]

    found = locate("H-01", "这不是 diff 里的任何一行", hunks)

    assert found.level == LOCATE_FAILED
    assert found.reason == REASON_NO_MATCH
    assert (found.start_line, found.end_line, found.side) == (0, 0, None)


def test_empty_existing_code_fails_with_its_own_reason() -> None:
    """测试：空片段（含纯空白、纯空行）单独记原因。

    schema 只保证字段存在，保证不了内容有意义。它与「匹配不上」是两回事：
    一个是模型没给锚点，一个是给了但找不到 —— 排障方向完全不同。
    """
    hunks = [make_hunk("H-01", "+\tfoo()")]

    for raw in ("", "   ", "\n\n", "+\n-"):
        found = locate("H-01", raw, hunks)
        assert found.reason == REASON_EMPTY_EXISTING_CODE, raw


# ---------- 组级：不丢评论 ----------


def test_locate_group_keeps_every_comment() -> None:
    """测试：无论定位成不成功，评论一条都不丢。"""
    group = make_group([make_hunk("H-01", "+\tfoo()\n-\tgone()", old_start=3)])
    hunk_comments = {
        "H-01": [
            make_input("foo()", "能定位的"),
            make_input("找不到的", "定位失败的"),
        ]
    }

    comments = locate_group(group, hunk_comments)

    assert [item.content for item in comments] == ["能定位的", "定位失败的"]
    assert [item.locate_level for item in comments] == [LOCATE_HUNK, LOCATE_FAILED]


def test_locate_group_carries_side_and_coordinates() -> None:
    """测试：侧与行号一起落到 Comment 上，且不变量成立。

    不变量：**定位成功 ⇔ side 非空**。它没被写进类型，所以要有用例守着 ——
    漏了 side 的评论会以「新版行号」的样子渲染出来，而它其实是旧版的。
    """
    group = make_group([make_hunk("H-01", "+\tbar()\n-\tgone()", new_start=8, old_start=3)])
    hunk_comments = {
        "H-01": [
            make_input("bar()", "新增侧"),
            make_input("-gone()", "删除侧"),
            make_input("找不到的", "未定位"),
        ]
    }

    comments = locate_group(group, hunk_comments)

    assert [(item.side, item.start_line) for item in comments] == [
        (SIDE_RIGHT, 8),
        (SIDE_LEFT, 3),
        (None, 0),
    ]
    for item in comments:
        assert (item.locate_level == LOCATE_FAILED) == (item.side is None)


def test_locate_group_path_follows_the_hit_hunk() -> None:
    """测试：组内命中后 path 与 hunk_id 仍同源。"""
    group = make_group(
        [
            make_hunk("H-01", "+\tfoo()", path="pkg/a.go"),
            make_hunk("H-02", "+\tbar()", path="pkg/b.go"),
        ]
    )

    comments = locate_group(group, {"H-01": [make_input("bar()")]})

    assert (comments[0].hunk_id, comments[0].path) == ("H-02", "pkg/b.go")


def test_locate_group_failed_keeps_reported_path() -> None:
    """测试：定位失败时 path 仍取自模型报的 hunk —— 评论得有个去处。"""
    group = make_group([make_hunk("H-01", "+\tfoo()", path="pkg/a.go")])

    comments = locate_group(group, {"H-01": [make_input("找不到的")]})

    assert comments[0].path == "pkg/a.go"
    assert comments[0].hunk_id == "H-01"


def test_locate_group_unknown_hunk_uses_placeholder() -> None:
    """测试：hunk_id 与分组对不上、又定位不到时用占位路径，不抛异常。

    report_hunks 已经挡掉了未知编号，这条是防御性的 —— 编号与分组真对不上
    时不能让整次审查崩在渲染前一步。
    """
    group = make_group([make_hunk("H-01", "+\tfoo()")])

    comments = locate_group(group, {"H-99": [make_input("哪都不在的片段")]})

    assert comments[0].path == UNKNOWN_PATH
    assert comments[0].hunk_id == "H-99"


def test_locate_group_rescues_a_bad_hunk_id() -> None:
    """测试：编号报错但代码片段在别的 hunk 里 → 照样定位成功。

    模型偶尔会把编号张冠李戴。第二层按代码找，比按编号找更可靠，顺手救回来。
    """
    group = make_group([make_hunk("H-01", "+\tfoo()", path="pkg/a.go")])

    comments = locate_group(group, {"H-99": [make_input("foo()")]})

    assert (comments[0].hunk_id, comments[0].path) == ("H-01", "pkg/a.go")
    assert comments[0].locate_level == LOCATE_GROUP


def test_locate_group_preserves_collection_order() -> None:
    """测试：不排序，保持收集顺序 —— 排序交给 sort_comments。"""
    group = make_group([make_hunk("H-01", "+\tfoo()", new_start=30), make_hunk("H-02", "+\tfoo()")])
    hunk_comments = {
        "H-01": [make_input("foo()", "先"), make_input("foo()", "后")],
        "H-02": [make_input("foo()", "末")],
    }

    comments = locate_group(group, hunk_comments)

    assert [item.content for item in comments] == ["先", "后", "末"]


def test_locate_group_empty() -> None:
    """测试：没有评论时不报错，也不打无意义的日志。"""
    assert locate_group(make_group([]), {}) == []


# ---------- 日志（§4.1：失败要可见） ----------


def test_unlocated_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    """测试：未定位打到 WARNING —— 用户得知道有评论没挂上位置。"""
    group = make_group([make_hunk("H-01", "+\tfoo()")])

    with caplog.at_level(logging.WARNING, logger=LOCATE_LOGGER):
        locate_group(group, {"H-01": [make_input("找不到的")]})

    assert "未定位 1 条" in caplog.text
    assert REASON_NO_MATCH in caplog.text


def test_ambiguous_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """测试：同 hunk 多命中取首个的条数要能看见。

    这是「取首个」这条规则的**代价指标** —— 不统计它就无从判断这个取舍值不值。
    """
    group = make_group([make_hunk("H-01", "+\treturn nil\n+\treturn nil")])

    with caplog.at_level(logging.INFO, logger=LOCATE_LOGGER):
        locate_group(group, {"H-01": [make_input("return nil")]})

    assert "同 hunk 多命中取首个 1 条" in caplog.text


def test_deleted_side_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """测试：删除侧的条数要能看见。

    这是「删除行也能挂」这条新能力的**使用率指标** —— 它同时说明聚合里少了
    多少条，是判断这套侧模型值不值的依据。
    """
    group = make_group([make_hunk("H-01", "+\tfoo()\n-\tgone()", old_start=3)])

    with caplog.at_level(logging.INFO, logger=LOCATE_LOGGER):
        locate_group(group, {"H-01": [make_input("foo()"), make_input("-gone()")]})

    assert "删除侧 1 条" in caplog.text
