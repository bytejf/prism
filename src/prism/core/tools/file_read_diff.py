"""跨文件改动查询工具。

分组的粒度是「包」，一个跨包改动（改了接口定义、改了实现、改了调用方）会被切成
三个互不可见的审查单元。`file_read` 能读别处的代码，但读不出「那里**改了什么**」
—— 读整个文件既贵又看不出改动本身。这个工具补的就是这一格：按路径取该文件的 diff。

数据来自 `ReviewContext.diffs`（`review` 命令启动时建的 `path → Diff` 索引），
**不重新读 git、不读工作区**：一次审查内 diff 不变，查的是已经在内存里的那份。

## 输出形态：git 原味，**不带 hunk 编号**

`<hunk id="H-01">` 是「本组待审清单」的编号。别的组的 hunk 没有编号 —— 硬编一个
等于邀请模型去评论越界代码，而 `report_hunks` 只认本组的编号，它只会白费一轮。
所以这里给的是 git 自己的形态（`diff --git` / `---` `+++` / `@@`），坐标与
`file_read` 同为「被审查版本」。

hunk 头里的行数**一律显式写出**（`@@ -5,1 +5,1 @@`），不学 git 在行数为 1 时
省略（`@@ -5 +5 @@`）—— 省掉的那个数正是模型要拿来数行号的起点，让它自己判断
「这是行号还是行数」是白送一次误读。
"""

import logging

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState
from prism.models.diff import DEV_NULL, Diff

logger = logging.getLogger(__name__)

# 查不到时的回答。
#
# **被过滤的文件（测试文件、敏感文件……）与「确实没改」在这里是同一句话**：
# 索引里只放可审查文件，所以说不出是哪一种。刻意不区分 —— `file_read` 与
# `code_search` 也不按过滤结果设卡（前者能读到测试文件、后者能搜到它），
# 只有这个工具多一句「你没权限看」，三个工具对「过滤」的态度就不一致了。
NOT_FOUND = "未找到该文件的改动：{path}（可能没有改动，也可能不在本次审查范围内）"

MISSING_INDEX = "错误：本次审查没有可用的 diff 索引，file_read_diff 不可用"


# 工具装饰器的形态约定与「为什么不能用 args_schema」，见 file_read.py
@tool(
    parse_docstring=True,
    description=(
        "读取指定文件在本次改动中「改了什么」。\n"
        "怀疑改动波及其他文件时用它，比用 file_read 读整个文件省得多。"
    ),
)
def file_read_diff(path: str, runtime: ToolRuntime[ReviewContext, ReviewState]) -> str:
    """读取指定文件的 diff。

    下面 `Args:` 里的说明**就是模型看到的参数说明**；`runtime` 由框架注入、
    模型看不到，会被自动排除。

    Args:
        path: 文件相对路径（改过名的文件，新旧路径都能查）

    Returns:
        该文件的完整 diff（含 hunk 头）；没有改动、不在审查范围内，
        或本次审查没建索引时返回说明文本
    """
    ctx = runtime.context
    # 索引为空是**装配错误**，不是「这个文件没改」：真那样的话每个查询都会得到
    # 「没找到改动」，模型据此判断「调用方没跟着改」并报出假问题 —— 一条不报错
    # 的错链路。所以这里显式报错并告警，让它能被看见。
    if not ctx.diffs:
        logger.warning("file_read_diff 不可用：ReviewContext.diffs 为空")
        return MISSING_INDEX

    if not path.strip():
        return "错误：path 不能为空"

    # 查不到就查不到了：这里只是字典查询，不碰文件系统，``..`` 与绝对路径
    # 最多是查不到，不构成路径穿越 —— 不做 file_read 那套路径校验
    diff = ctx.diffs.get(path)
    if diff is None:
        logger.debug("file_read_diff 未命中：%s", path)
        return NOT_FOUND.format(path=path)

    text = render_diff(diff)
    logger.debug("file_read_diff 命中：%s（%d 字符）", path, len(text))
    return text


def render_diff(diff: Diff) -> str:
    """把 Diff 渲染成 git 风格的文本。

    形态照抄 `git diff`：文件头 + hunk 头 + 正文。文件名带 ``a/`` ``b/`` 前缀、
    新增/删除/改名各有一行标记 —— 模型见过真实 diff 无数次，形态越像它越不会
    误读。新增与删除文件的那一侧用 ``/dev/null``。

    Args:
        diff: 单个文件的 diff

    Returns:
        git 风格的 diff 文本
    """
    # diff --git 行里 git 从不写 /dev/null：新增文件也写成 a/x b/x，用
    # 下面那行 new file 表示「这一侧不存在」
    git_old = diff.new_path if diff.is_new else diff.old_path
    git_new = diff.old_path if diff.is_deleted else diff.new_path

    lines = [f"diff --git a/{git_old} b/{git_new}"]
    if diff.is_new:
        lines.append("new file")
    elif diff.is_deleted:
        lines.append("deleted file")
    elif diff.old_path != diff.new_path:
        # 改名：把旧名字写在明处。模型可能正是拿着提示词里的
        # `renamed-from` 来查的，让它在这里对得上号
        lines.append(f"rename from {diff.old_path}")
        lines.append(f"rename to {diff.new_path}")

    old_ref = DEV_NULL if diff.is_new else f"a/{diff.old_path}"
    new_ref = DEV_NULL if diff.is_deleted else f"b/{diff.new_path}"
    lines.append(f"--- {old_ref}")
    lines.append(f"+++ {new_ref}")
    for hunk in diff.hunks:
        lines.append(f"@@ -{hunk.old_start},{hunk.old_lines} +{hunk.new_start},{hunk.new_lines} @@")
        if hunk.content:
            lines.append(hunk.content)
    return "\n".join(lines)
