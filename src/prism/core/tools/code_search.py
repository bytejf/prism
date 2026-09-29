"""代码搜索工具。

基于 ``git grep`` 实现：走 git 索引，天然尊重 .gitignore，
且不引入额外依赖。搜索范围限定在 ``ctx.project_root``。
"""

import logging
import subprocess

from langchain_core.tools import BaseTool, tool

from prism.core.tools.context import ToolContext

logger = logging.getLogger(__name__)

# 单次最多返回的结果数，LLM 传入更大的值也会被压到这个上限
MAX_RESULTS = 50

# git grep 子进程超时（秒）
TIMEOUT_SECONDS = 30

# git grep 的退出码：0 有匹配，1 无匹配，其余为错误
RC_NO_MATCH = 1


def make_code_search_tool(ctx: ToolContext) -> BaseTool:
    """构造 code_search 工具，绑定运行时上下文。

    Args:
        ctx: 工具执行上下文

    Returns:
        绑定上下文的 code_search 工具
    """

    @tool
    def code_search(
        pattern: str,
        file_pattern: str = "*.go",
        max_results: int = 50,
    ) -> str:
        """按内容搜索代码。

        Args:
            pattern: 搜索模式（支持正则）
            file_pattern: 文件过滤，默认 "*.go"
            max_results: 最大结果数，默认 50

        Returns:
            匹配行列表（含文件路径、行号、内容）；无匹配或出错时返回说明文本
        """
        limit = max(1, min(max_results, MAX_RESULTS))
        logger.debug(
            "code_search 调用：pattern=%s file_pattern=%s max_results=%d",
            pattern,
            file_pattern,
            limit,
        )

        if not ctx.project_root:
            logger.warning("code_search 缺少 project_root")
            return "错误：ToolContext.project_root 未设置"

        # -c core.quotepath=false：含中文等非 ASCII 的路径按原样输出，
        #   否则 git 会转义成 "a-\346\234\200.md"，LLM 拿到后无法回传给 file_read
        # -n 带行号，-E 扩展正则，-I 跳过二进制文件
        command = [
            "git",
            "-c",
            "core.quotepath=false",
            "grep",
            "--line-number",
            "--extended-regexp",
            "-I",
            pattern,
            "--",
            file_pattern,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ctx.project_root,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("code_search 超时：%s", pattern)
            return f"错误：搜索超时（超过 {TIMEOUT_SECONDS} 秒）"
        except OSError as exc:
            logger.warning("code_search 无法执行 git：%s", exc)
            return f"错误：无法执行 git grep：{exc}"

        if completed.returncode not in (0, RC_NO_MATCH):
            logger.warning(
                "code_search 失败：rc=%d %s", completed.returncode, completed.stderr.strip()
            )
            return f"错误：git grep 失败：{completed.stderr.strip()}"

        matched = completed.stdout.splitlines()
        if not matched:
            logger.debug("code_search 无匹配：%s", pattern)
            return f"未找到匹配：{pattern}"

        body = "\n".join(matched[:limit])
        if len(matched) > limit:
            logger.debug("code_search 截断：共 %d 条，返回 %d 条", len(matched), limit)
            body += f"\n（已截断：共 {len(matched)} 条匹配，只显示前 {limit} 条）"
        return body

    return code_search
