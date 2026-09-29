"""diff 解析的单元测试。"""

from prism.core.diff.parser import UNKNOWN_PATH, parse_diff_text

# 两个 hunk 的普通修改：hunk1 新增 2 行删除 1 行，hunk2 删除 1 行
TWO_HUNK_DIFF = "\n".join(
    [
        "diff --git a/internal/agent/agent.go b/internal/agent/agent.go",
        "index 1111111..2222222 100644",
        "--- a/internal/agent/agent.go",
        "+++ b/internal/agent/agent.go",
        "@@ -10,4 +10,5 @@ func (a *Agent) Run() {",
        " line1",
        "-line2",
        "+line2new",
        "+line3new",
        " line4",
        "@@ -30,2 +31,1 @@ func (a *Agent) Stop() {",
        " lineA",
        "-lineB",
    ]
)


def test_parse_two_hunks() -> None:
    """测试解析单个文件的两个 hunk 及增删行数。"""
    diffs = parse_diff_text(TWO_HUNK_DIFF)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.old_path == "internal/agent/agent.go"
    assert diff.new_path == "internal/agent/agent.go"
    assert diff.path == "internal/agent/agent.go"
    assert len(diff.hunks) == 2
    assert diff.insertions == 2
    assert diff.deletions == 2
    assert not diff.is_new
    assert not diff.is_deleted
    assert not diff.is_binary


def test_parse_hunk_line_ranges_and_content() -> None:
    """测试 hunk 的行号范围与原始内容（含 @@ 行）。"""
    hunks = parse_diff_text(TWO_HUNK_DIFF)[0].hunks

    assert (hunks[0].old_start, hunks[0].old_lines) == (10, 4)
    assert (hunks[0].new_start, hunks[0].new_lines) == (10, 5)
    assert hunks[0].content.startswith("@@ -10,4 +10,5 @@ func (a *Agent) Run() {")
    assert "-line2" in hunks[0].content
    # hunk 内容在下一个 @@ 行处截断，不含下一个 hunk
    assert hunks[0].content.endswith(" line4")

    assert (hunks[1].old_start, hunks[1].old_lines) == (30, 2)
    assert (hunks[1].new_start, hunks[1].new_lines) == (31, 1)
    assert "-lineB" in hunks[1].content


def test_parse_new_file() -> None:
    """测试新增文件：old_path 为 /dev/null，path 取 new_path。"""
    text = "\n".join(
        [
            "diff --git a/new.go b/new.go",
            "new file mode 100644",
            "index 0000000..3333333",
            "--- /dev/null",
            "+++ b/new.go",
            "@@ -0,0 +1,2 @@",
            "+package main",
            "+",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.is_new
    assert not diff.is_deleted
    assert diff.old_path == "/dev/null"
    assert diff.new_path == "new.go"
    assert diff.path == "new.go"
    assert diff.insertions == 2
    assert diff.deletions == 0


def test_parse_deleted_file() -> None:
    """测试删除文件：new_path 为 /dev/null，path 取 old_path。"""
    text = "\n".join(
        [
            "diff --git a/old.go b/old.go",
            "deleted file mode 100644",
            "index 3333333..0000000",
            "--- a/old.go",
            "+++ /dev/null",
            "@@ -1,2 +0,0 @@",
            "-package main",
            "-",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.is_deleted
    assert not diff.is_new
    assert diff.old_path == "old.go"
    assert diff.new_path == "/dev/null"
    assert diff.path == "old.go"
    assert diff.deletions == 2


def test_parse_binary_file_keeps_no_hunks() -> None:
    """测试二进制文件：标记 is_binary 且不解析内容。"""
    text = "\n".join(
        [
            "diff --git a/logo.png b/logo.png",
            "new file mode 100644",
            "index 0000000..4444444",
            "Binary files /dev/null and b/logo.png differ",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1
    assert diffs[0].is_binary
    assert diffs[0].hunks == []


def test_parse_multiple_files() -> None:
    """测试一次解析多个文件。"""
    text = "\n".join(
        [
            TWO_HUNK_DIFF,
            "diff --git a/b.go b/b.go",
            "index 5555555..6666666 100644",
            "--- a/b.go",
            "+++ b/b.go",
            "@@ -1,1 +1,2 @@",
            " package main",
            "+// 新增注释",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 2
    assert [d.path for d in diffs] == ["internal/agent/agent.go", "b.go"]
    assert diffs[1].insertions == 1


def test_parse_headerless_diff() -> None:
    """测试手写 diff（无 diff --git 头）。"""
    text = "\n".join(
        [
            "--- a/test.go",
            "+++ b/test.go",
            "@@ -1,1 +1,2 @@",
            " package main",
            "+// extra",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1
    assert diffs[0].new_path == "test.go"
    assert diffs[0].insertions == 1


def test_deleted_line_starting_with_dashes_is_not_a_file_header() -> None:
    """测试删除行内容以 `--` 开头时（diff 行为 `--- xxx`）不被误判为文件头。"""
    text = "\n".join(
        [
            "diff --git a/x.go b/x.go",
            "--- a/x.go",
            "+++ b/x.go",
            "@@ -1,3 +1,2 @@",
            " keep",
            "--- removed comment line",
            "+added",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.old_path == "x.go"
    assert diff.new_path == "x.go"
    assert len(diff.hunks) == 1
    assert diff.deletions == 1
    assert diff.insertions == 1
    assert "--- removed comment line" in diff.hunks[0].content


def test_parse_pure_rename_without_content_change() -> None:
    """测试纯重命名块（无 ---/+++ 行）：路径退回 diff --git 头，且不算新增。"""
    text = "\n".join(
        [
            "diff --git a/old.go b/renamed.go",
            "similarity index 100%",
            "rename from old.go",
            "rename to renamed.go",
        ]
    )
    diffs = parse_diff_text(text)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.old_path == "old.go"
    assert diff.new_path == "renamed.go"
    assert diff.path == "renamed.go"
    assert diff.hunks == []
    assert not diff.is_new
    assert not diff.is_deleted


def test_parse_symlink_creation() -> None:
    """测试新增符号链接：模式 120000 标记 is_symlink。"""
    text = "\n".join(
        [
            "diff --git a/link.go b/link.go",
            "new file mode 120000",
            "index 0000000..abc1234",
            "--- /dev/null",
            "+++ b/link.go",
            "@@ -0,0 +1 @@",
            "+target/path",
        ]
    )
    diffs = parse_diff_text(text)
    assert diffs[0].is_symlink
    assert diffs[0].is_new


def test_parse_symlink_target_change() -> None:
    """测试符号链接指向变更：index 行尾部模式为 120000。"""
    text = "\n".join(
        [
            "diff --git a/link.go b/link.go",
            "index abc1234..def5678 120000",
            "--- a/link.go",
            "+++ b/link.go",
            "@@ -1 +1 @@",
            "-old-target",
            "+new-target",
        ]
    )
    diffs = parse_diff_text(text)
    assert diffs[0].is_symlink
    assert not diffs[0].is_new
    assert not diffs[0].is_deleted


def test_parse_mode_change_to_symlink() -> None:
    """测试普通文件改成符号链接。"""
    text = "\n".join(
        [
            "diff --git a/f b/f",
            "old mode 100644",
            "new mode 120000",
        ]
    )
    diffs = parse_diff_text(text)
    assert diffs[0].is_symlink


def test_parse_mode_change_from_symlink_is_not_symlink() -> None:
    """测试符号链接改成普通文件：只看变更之后的状态，不算符号链接。"""
    text = "\n".join(
        [
            "diff --git a/f b/f",
            "old mode 120000",
            "new mode 100644",
        ]
    )
    diffs = parse_diff_text(text)
    assert not diffs[0].is_symlink


def test_parse_regular_file_is_not_symlink() -> None:
    """测试普通文件不会被标记为符号链接。"""
    assert not parse_diff_text(TWO_HUNK_DIFF)[0].is_symlink


def test_parse_unparsable_text_falls_back_to_raw() -> None:
    """测试无法解析的文本降级为原始文本，不抛异常。"""
    raw = "\n".join(["这不是 diff", "随便什么内容"])
    diffs = parse_diff_text(raw)
    assert len(diffs) == 1

    diff = diffs[0]
    assert diff.path == UNKNOWN_PATH
    assert diff.hunks[0].content == raw
    assert diff.insertions == 0


def test_parse_empty_text() -> None:
    """测试空文本返回空列表。"""
    assert parse_diff_text("") == []
    assert parse_diff_text("\n\n") == []
