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
from prism.core.filter.rules import is_reviewable_path
from prism.core.tools.paths import check_no_nul, check_relative_path

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
        max_results: 最大结果数；**默认值就是上限**，传更大的会被压到上限

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

    # 两个参数都会进 argv，先过我们自己的校验：越界让 git 拒的话，模型拿到的是
    # 带本机绝对路径的英文 stderr；含 NUL 的话 subprocess 直接抛异常（没人接）。
    # 见 core/tools/paths.py
    try:
        check_no_nul(pattern, field="pattern")
        check_relative_path(file_pattern, field="file_pattern")
    except ValueError as exc:
        logger.warning("code_search 参数校验失败：%s", exc)
        return f"错误：{exc}"

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

    # 按 `\n` 切，**不能用 `splitlines()`**：后者把 U+2028 / U+2029 / `\x0b`
    # 这些 Unicode 行边界也当换行，而 git 只认 `\n`。压缩过的 JS 里就有 U+2028
    # （实测 swagger-ui-bundle.js），于是一条匹配被切成几段、后几段丢了路径前缀
    # —— 既少了一条结果，还白占 `max_results` 的额度。
    matched = [line for line in completed.stdout.split("\n") if line]
    # git grep 带 rev 搜索时给每条结果加 `<rev>:` 前缀（`HEAD:db.go:12: ...`，
    # rev 原样回显，分支名 / SHA 都是）。剥掉它：结果的路径要与 file_read /
    # file_read_diff 是同一套坐标，模型把这一行里的路径直接拿去调工具就能用 ——
    # 留着前缀，它得自己认出那不是路径的一部分，认错就是白查一次。
    if ctx.head_rev:
        prefix = f"{ctx.head_rev}:"
        matched = [line.removeprefix(prefix) for line in matched]

    # **在限条数之前按审查范围过滤**，与 diff 过滤用同一套规则。
    #
    # 这一步挡的不是越权（读得到也评不了，评论的 path 由 hunk_id 查出来），
    # 是 **token**：搜索范围是整个仓库，模型可以把 file_pattern 放开成 `*`，
    # 而 git grep 的一条结果 = 一整行源码 —— 打包产物的一行就是 1.4 MB
    # （实测：swagger-ui-bundle.js 单行 1,461,344 字符，一次搜索把上下文从
    # ~10k 炸到 ~500k token，撞穿 2 元上限、整组 0 条评论交卷）。
    # 过滤放在截断**之前**：`max_results` 限的是条数，不是字节，
    # 先截断会把额度浪费在永远不会返回的行上。
    before = len(matched)
    matched = [line for line in matched if is_reviewable_path(line.partition(":")[0])]
    skipped = before - len(matched)
    if skipped:
        logger.debug("code_search 过滤掉 %d 条范围外的匹配", skipped)

    if not matched:
        logger.debug("code_search 无匹配：%s", pattern)
        if skipped:
            # **「被过滤」与「真没有」必须分成两句话。** 合成一句时模型会把前者
            # 读成后者 —— 实测它搜不到 *_test.go，就写下「这个包里没有测试」，
            # 而那个包有 10 个测试文件、还是这次 PR 自己改的（见文档 25 号 P1）。
            # 后半句要说「无需再找」，否则它接着花轮次去找那批文件。
            return (
                f"未找到匹配：{pattern}\n"
                f"（另有 {skipped} 条命中落在审查范围之外 —— 非 Go 源码、测试文件或"
                f"生成文件，与本任务无关，**不必再找**）"
            )
        return f"未找到匹配：{pattern}"

    body = "\n".join(matched[:limit])
    # 部分被过滤时也要说：不说的话，模型会把返回的这几条当成「全部调用点」
    if skipped:
        body += f"\n（另有 {skipped} 条命中落在审查范围之外，与本任务无关）"
    if len(matched) > limit:
        logger.debug("code_search 截断：共 %d 条，返回 %d 条", len(matched), limit)
        body += f"\n（已截断：共 {len(matched)} 条匹配，只显示前 {limit} 条）"
    return body
