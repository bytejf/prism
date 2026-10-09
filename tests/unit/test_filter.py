"""文件过滤的单元测试。"""

from prism.core.filter.rules import (
    REASON_BINARY,
    REASON_BUILTIN_NOISE,
    REASON_EXTENSION,
    REASON_SYMLINK,
    filter_files,
)
from prism.models.diff import Diff


def make_diff(
    path: str,
    *,
    is_binary: bool = False,
    is_symlink: bool = False,
    is_deleted: bool = False,
) -> Diff:
    """构造测试用 Diff。

    Args:
        path: 文件路径（新增文件用；删除文件的 old_path 取该值）
        is_binary: 是否二进制
        is_symlink: 是否符号链接
        is_deleted: 是否删除文件

    Returns:
        Diff 对象
    """
    return Diff(
        old_path=path,
        new_path="/dev/null" if is_deleted else path,
        hunks=[],
        insertions=0,
        deletions=0,
        is_new=False,
        is_deleted=is_deleted,
        is_binary=is_binary,
        is_symlink=is_symlink,
    )


def test_filter_go_only() -> None:
    """测试只保留 .go 文件。"""
    diffs = [make_diff("a.go"), make_diff("b.py"), make_diff("c_test.go")]
    result = filter_files(diffs)
    assert len(result.reviewable) == 1
    assert result.reviewable[0].new_path == "a.go"
    assert len(result.skipped) == 2


def test_filter_excludes_builtin_noise() -> None:
    """测试内置降噪排除 *_test.go 与 *.pb.go，并记录原因。"""
    diffs = [
        make_diff("internal/agent/agent_test.go"),
        make_diff("internal/api/service.pb.go"),
        make_diff("internal/agent/agent.go"),
    ]
    result = filter_files(diffs)

    assert [d.path for d in result.reviewable] == ["internal/agent/agent.go"]
    reasons = {s.path: s.reason for s in result.skipped}
    assert reasons["internal/agent/agent_test.go"] == f"{REASON_BUILTIN_NOISE}: *_test.go"
    assert reasons["internal/api/service.pb.go"] == f"{REASON_BUILTIN_NOISE}: *.pb.go"


def test_filter_excludes_binary() -> None:
    """测试排除二进制文件。"""
    result = filter_files([make_diff("logo.png", is_binary=True), make_diff("a.go")])
    assert [d.path for d in result.reviewable] == ["a.go"]
    assert result.skipped[0].reason == REASON_BINARY


def test_filter_excludes_symlink() -> None:
    """测试排除符号链接。"""
    result = filter_files([make_diff("link.go", is_symlink=True), make_diff("a.go")])
    assert [d.path for d in result.reviewable] == ["a.go"]
    assert result.skipped[0].reason == REASON_SYMLINK


def test_filter_hard_exclusion_takes_priority() -> None:
    """测试硬性排除优先于扩展名与降噪判断。"""
    result = filter_files(
        [
            make_diff("legacy.py", is_binary=True),  # 扩展名也不对，但应报 binary
            make_diff("agent_test.go", is_symlink=True),  # 降噪也命中，但应报 symlink
        ]
    )
    assert result.reviewable == []
    assert [s.reason for s in result.skipped] == [REASON_BINARY, REASON_SYMLINK]


def test_filter_extension_reason_includes_suffix() -> None:
    """测试扩展名不匹配时原因里带上实际后缀。"""
    result = filter_files([make_diff("libs.mod"), make_diff("Makefile")])
    assert [s.reason for s in result.skipped] == [
        f"{REASON_EXTENSION}: .mod",
        f"{REASON_EXTENSION}: 无扩展名",
    ]


def test_filter_allows_go_mod_by_filename() -> None:
    """`go.mod` 按文件名放行（依据：评测里 13% 的参考落在它上面）。

    同时守住边界：**只有 go.mod 这一个名字**放行，别的 `.mod` 仍然被挡
    —— 按文件名判而不是按扩展名判，就是不想把 `.mod` 一类都放进来。
    """
    result = filter_files(
        [
            make_diff("go.mod"),
            make_diff("pkg/gofr/datasource/file/s3/go.mod"),  # 子模块也要放行
            make_diff("libs.mod"),
        ]
    )
    assert [d.path for d in result.reviewable] == [
        "go.mod",
        "pkg/gofr/datasource/file/s3/go.mod",
    ]
    assert [s.path for s in result.skipped] == ["libs.mod"]


def test_filter_deleted_go_file_uses_old_path() -> None:
    """测试删除的 .go 文件按 old_path 判定并可审查。"""
    result = filter_files([make_diff("internal/old.go", is_deleted=True)])
    assert len(result.reviewable) == 1
    assert result.reviewable[0].path == "internal/old.go"


def test_filter_empty_input() -> None:
    """测试空输入。"""
    result = filter_files([])
    assert result.reviewable == []
    assert result.skipped == []
