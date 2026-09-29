"""unified diff 文本解析。

逐行状态机解析 git unified diff 文本，产出 `prism.models.diff` 中的数据结构。
解析失败时降级为「原始文本 hunk」，不抛异常、不中断流程。
"""

import logging
import re

from prism.models.diff import Diff, Hunk

logger = logging.getLogger(__name__)

# hunk 头：@@ -old_start[,old_lines] +new_start[,new_lines] @@ 后接可选上下文
# 行数省略时表示 1 行
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# 文件块起始行：diff --git a/x b/x
# ponytail: 不处理 git 对含特殊字符路径的 C 风格转义（\t、\nnn）
_DIFF_GIT_RE = re.compile(r"^diff --git (\S+) (\S+)")

# 无法从 diff 文本中还原出路径时的占位符
UNKNOWN_PATH = "<unknown>"

# git 中符号链接的文件模式（普通文件为 100644 / 100755）
SYMLINK_MODE = "120000"

# 未变更侧路径的占位符
DEV_NULL = "/dev/null"


def parse_diff_text(text: str) -> list[Diff]:
    """解析 unified diff 文本为 Diff 列表。

    Args:
        text: git unified diff 文本，可含多个文件

    Returns:
        Diff 列表；无法解析路径的块降级为「原始文本 hunk」，不抛异常
    """
    diffs = [
        diff
        for block in _split_blocks(text.splitlines())
        if (diff := _parse_block(block)) is not None
    ]
    logger.debug(
        "解析 diff 文本：%d 个文件，%d 个 hunk",
        len(diffs),
        sum(len(diff.hunks) for diff in diffs),
    )
    return diffs


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """按 `diff --git` 把 diff 文本切分成文件块。

    没有 `diff --git` 前缀时（手写 diff）整体视为一个块。

    Args:
        lines: diff 文本按行切分的结果

    Returns:
        文件块列表，每块是若干行
    """
    blocks: list[list[str]] = []
    headerless = False  # 当前块是否为手写 diff（无 diff --git 头）

    for line in lines:
        if line.startswith("diff --git "):
            blocks.append([line])
            headerless = False
        elif headerless or (blocks and line.strip()):
            blocks[-1].append(line)
        elif line.strip():
            # 手写 diff：没有 diff --git 头，从第一个非空行开始整块收下
            blocks.append([line])
            headerless = True
    return blocks


def _parse_block(lines: list[str]) -> Diff | None:
    """解析单个文件的 diff 块。

    Args:
        lines: 该文件块的所有行

    Returns:
        Diff 对象；输入为空时返回 None
    """
    if not lines:
        return None

    old_path: str | None = None
    new_path: str | None = None
    is_new = False
    is_deleted = False
    is_binary = False
    is_symlink = False
    hunks: list[Hunk] = []
    insertions = 0
    deletions = 0

    # 当前正在累积的 hunk：头部匹配、头部原文、内容行
    header: re.Match[str] | None = None
    header_line = ""
    body: list[str] = []

    def flush() -> None:
        """把累积中的 hunk 落盘，并统计增删行数。"""
        nonlocal header, header_line, body, insertions, deletions
        if header is None:
            return
        hunks.append(_make_hunk(header_line, header, body))
        insertions += sum(1 for line in body if line.startswith("+"))
        deletions += sum(1 for line in body if line.startswith("-"))
        header, header_line, body = None, "", []

    for line in lines:
        # 在 hunk 内部：`--- ` / `+++ ` 也可能是增删行的内容，必须先判内容行
        if header is not None and _is_hunk_body_line(line):
            body.append(line)
            continue

        flush()

        if (matched := _HUNK_HEADER_RE.match(line)) is not None:
            header, header_line = matched, line
        elif line.startswith("new file mode"):
            is_new = True
            is_symlink = _has_symlink_mode(line)
        elif line.startswith("deleted file mode"):
            is_deleted = True
            is_symlink = _has_symlink_mode(line)
        elif line.startswith(("index ", "new mode ")):
            # `old mode` 不参与判断：只关心变更「之后」是不是符号链接
            is_symlink = is_symlink or _has_symlink_mode(line)
        elif line.startswith(("Binary files ", "GIT binary patch")):
            is_binary = True
        elif line.startswith("--- "):
            old_path = _normalize_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _normalize_path(line[4:])
    flush()

    # 二进制文件不解析内容
    if is_binary:
        hunks = []

    # 没有 ---/+++ 行时，退回 diff --git 头里的路径
    if old_path is None or new_path is None:
        git_header = _DIFF_GIT_RE.match(lines[0])
        if git_header is not None:
            old_path = old_path or _normalize_path(git_header.group(1))
            new_path = new_path or _normalize_path(git_header.group(2))

    # 完全无法解析：降级为原始文本，保留内容供后续阶段使用
    if old_path is None and new_path is None:
        logger.warning("diff 块无法解析出路径，降级为原始文本：%r", lines[0])
        return Diff(
            old_path=UNKNOWN_PATH,
            new_path=UNKNOWN_PATH,
            hunks=[Hunk(0, 0, 0, 0, "\n".join(lines))],
            insertions=0,
            deletions=0,
            is_new=False,
            is_deleted=False,
            is_binary=False,
            is_symlink=False,
        )

    return Diff(
        old_path=old_path or DEV_NULL,
        new_path=new_path or DEV_NULL,
        hunks=hunks,
        insertions=insertions,
        deletions=deletions,
        is_new=is_new,
        is_deleted=is_deleted,
        is_binary=is_binary,
        is_symlink=is_symlink,
    )


def _has_symlink_mode(line: str) -> bool:
    """判断文件模式行是否声明了符号链接模式（120000）。

    适用于 `new file mode 120000`、`index a..b 120000`、`new mode 120000`。

    Args:
        line: diff 中的模式相关行

    Returns:
        是否为符号链接模式
    """
    parts = line.split()
    return bool(parts) and parts[-1] == SYMLINK_MODE


def _is_hunk_body_line(line: str) -> bool:
    """判断是否为 hunk 内容行。

    内容行以空格（上下文）、`+`（新增）、`-`（删除）、`\\`（无换行标记）开头，
    空行也视作上下文行。

    Args:
        line: diff 文本中的一行

    Returns:
        是否属于 hunk 内容
    """
    return line[:1] in (" ", "+", "-", "\\", "")


def _make_hunk(header_line: str, matched: re.Match[str], body: list[str]) -> Hunk:
    """由 hunk 头与内容行构造 Hunk。

    Args:
        header_line: hunk 头原文，如 `@@ -1,3 +1,5 @@ func foo() {`
        matched: hunk 头的正则匹配结果
        body: hunk 内容行

    Returns:
        Hunk 对象
    """
    return Hunk(
        old_start=int(matched.group(1)),
        old_lines=int(matched.group(2)) if matched.group(2) is not None else 1,
        new_start=int(matched.group(3)),
        new_lines=int(matched.group(4)) if matched.group(4) is not None else 1,
        content="\n".join([header_line, *body]),
    )


def _normalize_path(raw: str) -> str:
    """去掉 git 的 a/ b/ 前缀与包裹引号。

    Args:
        raw: `--- ` / `+++ ` 后的原始路径文本

    Returns:
        规范化后的路径；`/dev/null` 原样返回
    """
    path = raw.strip()
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        # ponytail: 只剥引号，不解码挂引号路径里的转义序列
        path = path[1:-1]
    if path == DEV_NULL:
        return path
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path
