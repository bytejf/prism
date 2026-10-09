"""文件读取工具。

读取文件指定行范围，输出带行号，便于 LLM 引用具体位置。
限制单次行数与路径范围：防超长输出、防路径穿越。
"""

import logging
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.config.settings import settings
from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState
from prism.core.filter.rules import path_skip_reason
from prism.core.tools.paths import check_relative_path

logger = logging.getLogger(__name__)

# 单次最多读取的行数见 default.toml 的 [tools.file_read].max_lines

# 超出上限时的提示（文档要求提示里含「截断」）
TRUNCATED_HINT = "（已截断：单次最多读取 {max_lines} 行）"


# 工具装饰器四个统一成这个形态：**描述走 `description=`、参数说明走 docstring 的
# `Args:`**（`parse_docstring=True` 把它填进参数 schema）。
#
# **别改用 `args_schema=`**：显式给了 args_schema，框架就不再从函数签名推导，
# `runtime: ToolRuntime[...]` 会从 `get_input_schema()` 里消失、注入断掉，
# 调用时报 `TypeError: missing 1 required positional argument: 'runtime'`。
# 详见 CLAUDE.md 的「框架缺口」。
@tool(parse_docstring=True, description="读取文件指定行范围的内容。")
def file_read(
    path: str,
    start_line: int,
    end_line: int,
    runtime: ToolRuntime[ReviewContext, ReviewState],
) -> str:
    """读取文件指定行范围的内容。

    下面 `Args:` 里的说明**就是模型看到的参数说明**；`runtime` 由框架注入、
    模型看不到，会被自动排除。

    Args:
        path: 文件相对路径
        start_line: 起始行号（1-based）
        end_line: 结束行号（1-based，最多 500 行）
        runtime: 运行时（框架注入，模型看不到）

    Returns:
        文件内容（每行前缀行号），超出 500 行时截断并提示
    """
    max_lines = settings.tools.file_read.max_lines
    ctx = runtime.context
    logger.debug("file_read 调用：path=%s start_line=%d end_line=%d", path, start_line, end_line)
    if not ctx.project_root:
        logger.warning("file_read 缺少 project_root")
        return "错误：ReviewContext.project_root 未设置"

    # 路径校验放在读取之前：git 读取路径不经过文件系统的 resolve()，
    # 若先读再校验，``../`` 会绕过检查
    try:
        check_relative_path(path)
    except ValueError as exc:
        logger.warning("file_read 路径校验失败：%s", exc)
        return f"错误：{exc}"

    # 审查范围与过滤阶段**同一套规则**：审不到的文件也不必读。这一步挡的不只是
    # 越权，更是 token —— 一次 `git grep` 或一个打包产物就能把一组预算烧穿
    reason = path_skip_reason(path)
    if reason is not None:
        logger.debug("file_read 拒绝范围外文件：%s（%s）", path, reason)
        return f"错误：{path} 不在审查范围内（{reason}）"

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
    truncated = end - start + 1 > max_lines
    if truncated:
        end = start + max_lines - 1

    body = "\n".join(f"{number:>6}: {lines[number - 1]}" for number in range(start, end + 1))
    logger.debug("file_read 完成：%d 行（总 %d 行，截断=%s）", end - start + 1, total, truncated)
    if truncated:
        return f"{body}\n{TRUNCATED_HINT.format(max_lines=max_lines)}"
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
