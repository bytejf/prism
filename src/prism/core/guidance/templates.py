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

# 工作方式与报告节奏。
#
# 这两件事以前分头写在 system（「工作方式」）与 user（「报告节奏」）里，同一句
# 「报告完调 end_done」说了两遍。现在合并成一段放 user：system 只留**不随组变化**
# 的东西（角色、输入格式、硬约束），随本次审查走的都放 user。
WORKFLOW: tuple[str, ...] = (
    "先用 file_read 读取 hunk 周边上下文，必要时用 code_search 查调用方与定义",
    "每完成 1~2 个 hunk 的审查，立即调用 report_hunks 报告结论，不要攒批",
    "某个 hunk 需要深入探索时，可以在探索完成后报告该 hunk",
    "同一 hunk 的多个问题可以合并到一次 report_hunks 中",
)


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

## 输入格式

待审代码用下列标签描述。**标签与属性全部由我方生成，含义如下**：

| 标签 / 属性 | 含义 |
|---|---|
| `<diff>` | 本组待审 hunk 的容器 |
| `<file path="...">` | 一个文件。`path` 是仓库内相对路径 |
| `<file renamed-from="...">` | 该文件由旧路径改名而来，**只在改名时出现** |
| `<file deleted="true">` | 该文件已被整体删除，**只在删除时出现** |
| `<hunk>` | 一个 hunk（一段连续改动） |
| `id="H-01"` | hunk 编号。report_hunks / end_done 用它指代 |
| `old="7,3"` `new="6,0"` | 改动在旧 / 新文件中的位置，格式同 git 的 `@@` 头 |
| `symbol="(*Store).Save"` | 改动归属的符号名 |
| `kind="method"` | 符号种类，取值见下方说明 |
| `lines="23-29"` | 归属符号在新文件中的行号范围（闭区间） |
| 标签正文 | hunk 的 diff 正文：`-` 删除行，`+` 新增行 |

说明：

- `old` / `new` 的格式是 `起始行,行数`，**行数为 0 表示该侧没有内容** ——
  新增文件是 `old="0,0"`，纯删除是 `new="6,0"`。旧文件只用于理解改动过程，
  **没有工具能读到它**。
- `kind` 的取值：`function` / `method` / `struct` / `interface` / `type` /
  `const` / `var` / `package` / `import`。
- `lines` 是**要读代码时唯一该用的属性**，它和 file_read 是同一套坐标。
  符号已不在新文件中（被整体删除或改名）时没有这个属性。
- `deleted="true"` 的文件 file_read 读不到，改用 code_search 查它的调用方。
- **所有行号都是「被审查版本」的坐标**，与 file_read 读到的一致。
- 正文按**原文**给出，未做 XML 转义（Go 的 `<-`、`&&`、`&Store{}` 原样出现）。

## 硬约束

- 只审查给定的 hunk，不要审查未列出的代码
- 每条评论必须给出 severity、category、content、existing_code
""")

USER = _t("""\
{diff_block}

{requirements}
""")

SCOPE = _t("""\
### 审查范围

本组 hunk：{span}（共 {count} 个），分布在 {files} 个文件。

{diff}
""")

SCOPE_EMPTY = _t("""\
### 审查范围

本组没有 hunk。
""")

DIMENSIONS = _t("""\
### 审查要求（对所有 hunk 强制适用）

每个 hunk 都必须完整审查以下维度：

{items}

⚠️ 注意：所有 hunk 都可能存在严重缺陷。
静态分析无法发现业务语义级的 Bug，你的语义审查是发现这类问题的唯一防线。
""")

WORKFLOW_SECTION = _t("""\
### 工作方式

{items}
""")

FINISH = _t("""\
### 结束审查

所有 hunk 审查完毕后，调用 end_done 结束审查。
如果有 hunk 未标记 complete，end_done 会被拒绝，并提示剩余 hunk。
""")

# hunk 标签：位置用 git 的 `起始行,行数` 写法，两侧都摆明。
# `{extra}` 放可选属性（如 `lines="5-9"`），没有时渲染成空串
HUNK = _t("""\
<hunk id="{hunk_id}" old="{old}" new="{new}" symbol="{symbol}" kind="{kind}"{extra}>
{content}
</hunk>""")

DIFF_FILE = _t("""\
<file path="{path}"{extra}>
{hunks}
</file>""")

# 主任务提示词：system 静态文本 + user 的 diff 块与要求。
# 变量契约由 input_variables 暴露，少传变量会在 format_messages 时报错。
MAIN_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", USER)])
