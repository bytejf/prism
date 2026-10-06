"""代码搜索工具。

基于 ``git grep`` 实现：走 git 索引，天然尊重 .gitignore，
且不引入额外依赖。搜索范围限定在 ``ctx.project_root``。
"""

import logging
import subprocess

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.config.settings import settings
from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState

logger = logging.getLogger(__name__)

# 结果数上限与 git grep 超时见 default.toml 的 [tools.code_search]

# git grep 的退出码：0 有匹配，1 无匹配，其余为错误
RC_NO_MATCH = 1


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py
@tool(parse_docstring=True, description="按内容搜索代码。")
def code_search(
    pattern: str,
    runtime: ToolRuntime[ReviewContext, ReviewState],
    file_pattern: str = "*.go",
    # 默认值取自配置，让工具 schema 里挂的 default 与配置一致 ——
    # 写死一个 50，改了配置之后模型看到的默认值就与实际行为对不上了
    max_results: int = settings.tools.code_search.max_results,
) -> str:
    """按内容搜索代码。

    下面 `Args:` 里的说明**就是模型看到的参数说明**；`runtime` 由框架注入、
    模型看不到，会被自动排除。

    Args:
        pattern: 搜索模式（支持正则）
        runtime: 运行时（框架注入，模型看不到）
        file_pattern: 文件过滤，默认 "*.go"
        max_results: 最大结果数，传超过配置上限的值会被压到上限

    Returns:
        匹配行列表（含文件路径、行号、内容）；无匹配或出错时返回说明文本

    Notes:
        ReviewContext.head_rev 已设置时搜索该版本，否则搜索工作区。
    """
    # 参数是「模型要多少」，配置是「最多给多少」—— 取两者的小值，
    # 别把配置值赋回参数名，那样模型传的值就失效了
    cap = settings.tools.code_search.max_results
    timeout_seconds = settings.tools.code_search.timeout_seconds
    ctx = runtime.context
    limit = max(1, min(max_results, cap))
    logger.debug(
        "code_search 调用：pattern=%s file_pattern=%s max_results=%d",
        pattern,
        file_pattern,
        limit,
    )

    if not ctx.project_root:
        logger.warning("code_search 缺少 project_root")
        return "错误：ReviewContext.project_root 未设置"

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
    ]
    # 指定 rev 时搜的是该版本的树，而不是工作区 —— 工作区可能停在别的分支，
    # 直接搜工作区会拿到改之前的代码
    if ctx.head_rev:
        command.append(ctx.head_rev)
    command.extend(["--", file_pattern])
    try:
        completed = subprocess.run(
            command,
            cwd=ctx.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("code_search 超时：%s", pattern)
        return f"错误：搜索超时（超过 {timeout_seconds} 秒）"
    except OSError as exc:
        logger.warning("code_search 无法执行 git：%s", exc)
        return f"错误：无法执行 git grep：{exc}"

    if completed.returncode not in (0, RC_NO_MATCH):
        logger.warning("code_search 失败：rc=%d %s", completed.returncode, completed.stderr.strip())
        return f"错误：git grep 失败：{completed.stderr.strip()}"

    matched = completed.stdout.splitlines()
    # git grep 带 rev 搜索时给每条结果加 `<rev>:` 前缀（`HEAD:db.go:12: ...`，
    # rev 原样回显，分支名 / SHA 都是）。剥掉它：结果的路径要与 file_read /
    # file_read_diff 是同一套坐标，模型把这一行里的路径直接拿去调工具就能用 ——
    # 留着前缀，它得自己认出那不是路径的一部分，认错就是白查一次。
    if ctx.head_rev:
        prefix = f"{ctx.head_rev}:"
        matched = [line.removeprefix(prefix) for line in matched]
    if not matched:
        logger.debug("code_search 无匹配：%s", pattern)
        return f"未找到匹配：{pattern}"

    body = "\n".join(matched[:limit])
    if len(matched) > limit:
        logger.debug("code_search 截断：共 %d 条，返回 %d 条", len(matched), limit)
        body += f"\n（已截断：共 {len(matched)} 条匹配，只显示前 {limit} 条）"
    return body
