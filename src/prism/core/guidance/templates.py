"""提示词模板文本。

静态文本集中在这里、用三引号整块书写 —— 一眼看出 LLM 收到的整体结构，
而不是 `"\\n".join((...))` 一行一行拼。变量由 `builder` 填进来。

三条约定：

1. 每个模板用 `_t(...)` 包住，去掉三引号带来的首尾换行，使渲染结果与
   手写拼接**逐字节一致**。文件里因此可以自然地让闭合引号独占一行。
2. **只有这里的字面文本受 f-string 模板约束**（花括号要写成 `{{}}`）。
   传给模板的**变量值**不受影响 —— diff 正文里有大量 Go 的 `{}`，
   它们是变量值，不需要转义。
3. `MAIN_PROMPT` 是组装好的 `ChatPromptTemplate`，变量契约由它暴露
   （`input_variables`），少传变量会直接报错而不是拼出半截提示词。
"""

from langchain_core.prompts import ChatPromptTemplate

# 强制审查维度（V1 固定，不做用户配置）
REVIEW_DIMENSIONS: tuple[str, ...] = (
    "**业务逻辑正确性** — 改动是否符合业务语义？是否存在逻辑错误？",
    "**跨文件一致性** — 调用方 / 实现方是否同步适配？",
    "**意图与实现一致性** — 改动是否实现了 commit message 声称的意图？",
    "**边界条件** — 空值、并发、错误路径、极端输入是否处理？",
    "**副作用** — 改动是否影响其他功能的隐式契约？",
)

# 报告节奏引导
REPORT_PACE: tuple[str, ...] = (
    "- 每完成 1~2 个 hunk 的审查，立即调用 report_hunks 报告结论",
    "- 不要等到所有 hunk 审完后一次性报告",
    "- 如果某个 hunk 需要深入探索，可以在探索完成后报告该 hunk",
    "- 如果发现多个问题属于同一 hunk，可以合并到一次 report_hunks 中",
)

# 本组 diff 段落的标题
DIFF_TITLE = "### 本组 diff"


def _t(text: str) -> str:
    """规范化模板：去掉三引号带来的首尾换行。

    让模板文件里可以自然地让闭合引号独占一行，而渲染结果与
    `"\\n".join(...)` 手写拼接逐字节一致。

    Args:
        text: 三引号模板原文

    Returns:
        去掉首尾换行的文本
    """
    return text.strip("\n")


SYSTEM = _t("""\
你是一名资深 Go 代码审查员。你正在审查一个变更集（PR）中的若干 hunk。

## 工作方式

- 先用 file_read 读取 hunk 周边上下文，必要时用 code_search 查调用方与定义
- 审查完成后用 report_hunks 报告该 hunk 的结论（有问题就给出评论，没问题也要 complete=true）
- 所有 hunk 报告完毕后调用 end_done 结束

## 硬约束

- 只审查给定的 hunk，不要审查未列出的代码
- 每条评论必须给出 severity、category、content、existing_code
- 报告节奏与审查维度的具体要求见随后的 user 消息
""")

USER = _t("""\
{guidance}

### 本组 diff

{diff_text}
""")

SCOPE = _t("""\
### 审查范围
本组 hunk：{span}（共 {count} 个）

{listing}
""")

SCOPE_EMPTY = _t("""\
### 审查范围
本组没有 hunk。
""")

DIMENSIONS = _t("""\
### 审查要求（对所有 hunk 强制适用）

无论初始注意力等级如何，每个 hunk 都必须完整审查以下维度：

{items}

⚠️ 注意：所有 hunk 都可能存在严重缺陷。
静态分析无法发现业务语义级的 Bug，你的语义审查是发现这类问题的唯一防线。
""")

PACE = _t("""\
### 报告节奏

随着审查进度逐步报告，不要攒批。

{items}
""")

FINISH = _t("""\
### 结束审查

所有 hunk 审查完毕后，调用 end_done 结束审查。
如果有 hunk 未标记 complete，end_done 会被拒绝，并提示剩余 hunk。
""")

DIFF_FILE = _t("""\
--- {path} ---
{hunks}
""")

# 主任务提示词：system 静态文本 + user 的指导与 diff。
# 变量契约由 input_variables 暴露，少传变量会在 format_messages 时报错。
MAIN_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", USER)])
