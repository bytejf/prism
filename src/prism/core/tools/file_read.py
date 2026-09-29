"""文件读取工具。

读取文件指定行范围，输出带行号，便于 LLM 引用具体位置。
限制单次行数与路径范围：防超长输出、防路径穿越。
"""

import logging
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from prism.core.tools.context import ToolContext

logger = logging.getLogger(__name__)

# 单次最多读取的行数
MAX_LINES = 500

# 超出上限时的提示（文档要求提示里含「截断」）
TRUNCATED_HINT = "（已截断：单次最多读取 {max_lines} 行）"


def make_file_read_tool(ctx: ToolContext) -> BaseTool:
    """构造 file_read 工具，绑定运行时上下文。

    Args:
        ctx: 工具执行上下文

    Returns:
        绑定上下文的 file_read 工具
    """

    @tool
    def file_read(path: str, start_line: int, end_line: int) -> str:
        """读取文件指定行范围的内容。

        Args:
            path: 文件相对路径
            start_line: 起始行号（1-based）
            end_line: 结束行号（1-based，最多 500 行）

        Returns:
            文件内容（每行前缀行号），超出 500 行时截断并提示
        """
        logger.debug(
            "file_read 调用：path=%s start_line=%d end_line=%d", path, start_line, end_line
        )
        try:
            target = _resolve_inside_project(ctx.project_root, path)
        except ValueError as exc:
            logger.warning("file_read 路径校验失败：%s", exc)
            return f"错误：{exc}"

        if not target.is_file():
            logger.warning("file_read 文件不存在：%s", path)
            return f"错误：文件不存在：{path}"

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
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
        logger.debug(
            "file_read 完成：%d 行（总 %d 行，截断=%s）", end - start + 1, total, truncated
        )
        if truncated:
            return f"{body}\n{TRUNCATED_HINT.format(max_lines=MAX_LINES)}"
        return body

    return file_read


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
        msg = "ToolContext.project_root 未设置"
        raise ValueError(msg)

    root = Path(project_root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        msg = f"路径越出项目根目录：{path}"
        raise ValueError(msg)
    return target
