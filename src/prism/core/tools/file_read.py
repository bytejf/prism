"""文件读取工具。

读取文件指定行范围，输出带行号，便于 LLM 引用具体位置。
限制单次行数与路径范围：防超长输出、防路径穿越。
"""

import logging
from pathlib import Path, PurePosixPath

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState

logger = logging.getLogger(__name__)

# 单次最多读取的行数
MAX_LINES = 500

# 超出上限时的提示（文档要求提示里含「截断」）
TRUNCATED_HINT = "（已截断：单次最多读取 {max_lines} 行）"


# parse_docstring=True：把 docstring 里 `Args:` 的逐条说明填进参数 schema。
# 默认是 False —— 不开的话**模型看不到任何参数说明**，我们写在 docstring 里的
# 「文件相对路径」「1-based」一个字都传不出去（实测：3 个参数有描述的 0 个）。
# 顺带把 `Args:` / `Returns:` 段从工具描述里剥掉，模型只看到摘要那一段。
@tool(parse_docstring=True)
def file_read(
    path: str,
    start_line: int,
    end_line: int,
    runtime: ToolRuntime[ReviewContext, ReviewState],
) -> str:
    """读取文件指定行范围的内容。

    Args:
        path: 文件相对路径
        start_line: 起始行号（1-based）
        end_line: 结束行号（1-based，最多 500 行）
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        文件内容（每行前缀行号），超出 500 行时截断并提示
    """
    ctx = runtime.context
    logger.debug("file_read 调用：path=%s start_line=%d end_line=%d", path, start_line, end_line)
    if not ctx.project_root:
        logger.warning("file_read 缺少 project_root")
        return "错误：ReviewContext.project_root 未设置"

    # 路径校验放在读取之前：git 读取路径不经过文件系统的 resolve()，
    # 若先读再校验，``../`` 会绕过检查
    try:
        _validate_relative_path(path)
    except ValueError as exc:
        logger.warning("file_read 路径校验失败：%s", exc)
        return f"错误：{exc}"

    text = _read_reviewed(ctx, path)
    if text is None:
        return f"错误：文件不存在：{path}"

    lines = text.splitlines()
    total = len(lines)
    if start_line > total:
        logger.warning("file_read 起始行超界：%d > %d", start_line, total)
        return f"错误：起始行 {start_line} 超出文件总行数 {total}"

    start = max(start_line, 1)
    end = min(max(end_line, start), total)
    truncated = end - start + 1 > MAX_LINES
    if truncated:
        end = start + MAX_LINES - 1

    body = "\n".join(f"{number:>6}: {lines[number - 1]}" for number in range(start, end + 1))
    logger.debug("file_read 完成：%d 行（总 %d 行，截断=%s）", end - start + 1, total, truncated)
    if truncated:
        return f"{body}\n{TRUNCATED_HINT.format(max_lines=MAX_LINES)}"
    return body


def _read_reviewed(ctx: ReviewContext, path: str) -> str | None:
    """读取被审查版本的文件内容，版本不可读时退化为读工作区。

    必须先读被审查版本：审查目标是 head 分支，而工作区可能停在别的分支，
    直接读工作区会拿到改之前的代码，导致系统性漏报。

    退化条件只看「**能不能读到版本**」，不看「版本里有没有这个文件」：

    - 版本可读 → 版本树是唯一事实来源。文件不在树里就是「不存在」，返回 None。
      此时若退化为读工作区，会把 PR 里根本没有的本地文件喂给 LLM，而且与
      code_search（搜版本树）对「这个版本有哪些文件」的答案不一致。
    - 版本不可读（没传 head_rev，或版本解析失败）→ 只能用工作区。

    Args:
        ctx: 工具执行上下文
        path: 文件相对路径

    Returns:
        文件文本；文件不存在或读不到时返回 None
    """
    if ctx.head_source() is not None:
        text = ctx.read_source(path)
        if text is None:
            logger.warning("file_read 文件不在被审查版本 %s 中：%s", ctx.head_rev, path)
        return text

    try:
        target = _resolve_inside_project(ctx.project_root, path)
    except ValueError as exc:
        logger.warning("file_read 路径校验失败：%s", exc)
        return None

    if not target.is_file():
        logger.warning("file_read 文件不存在：%s", path)
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def _validate_relative_path(path: str) -> None:
    """校验 path 是项目内的相对路径。

    纯字符串判断，与文件系统无关，因此对 git 读取路径同样有效。

    Args:
        path: 待校验的路径

    Raises:
        ValueError: 空路径、绝对路径，或含 ``..`` 段
    """
    if not path.strip():
        msg = "path 不能为空"
        raise ValueError(msg)

    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        msg = f"必须是相对路径：{path}"
        raise ValueError(msg)
    if ".." in candidate.parts:
        msg = f"路径不得包含 ..：{path}"
        raise ValueError(msg)


def _resolve_inside_project(project_root: str, path: str) -> Path:
    """把相对路径解析为项目内的绝对路径。

    用 resolve() 展开相对路径与符号链接后再比对，防止 ``../`` 或软链接越权。

    Args:
        project_root: 项目根目录
        path: 文件相对路径

    Returns:
        解析后的绝对路径

    Raises:
        ValueError: project_root 未设置，或路径越出项目根目录
    """
    if not project_root:
        msg = "ReviewContext.project_root 未设置"
        raise ValueError(msg)

    root = Path(project_root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        msg = f"路径越出项目根目录：{path}"
        raise ValueError(msg)
    return target
