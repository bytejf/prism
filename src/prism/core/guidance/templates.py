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

# 摘要的 7 个段标题。**输出格式要求与格式校验共用这一份** —— 校验时数
# 「出现了几个」，少于一半就当压缩失败（模型跑题了）。
# 与下面 COMPRESSION_SYSTEM 里那段标题清单必须一致，由单测看着
SUMMARY_SECTIONS: tuple[str, ...] = (
    "## Primary Request",
    "## Completed Hunks",
    "## Key Findings",
    "## Files Examined",
    "## Current Work",
    "## Next Step",
    "## Errors and Fixes",
)

# 压缩用的 system 段。
#
# 它自己的 system，不复用 SYSTEM：复用的话压缩请求的前缀就与审查请求逐字相同、
# 能吃到前缀缓存，但清理已经让缓存断在第一个工具结果上，省不出多少；而独立一份
# 换来的是**压缩提示词自包含** —— 以后改 SYSTEM 只改内容，不会连带改变压缩路径
# 的行为。这个取舍记在实现反馈文档里。
COMPRESSION_SYSTEM = _t("""\
你是一名资深 Go 代码审查员，现在负责**压缩**一段审查历史。

下面给你的是一段**已经进行了一部分**的代码审查会话的完整记录，它不是你正在
进行的对话。你的任务是把它压缩成结构化摘要，供审查继续下去。
**不要继续审查，不要评论这段历史本身。**

## 压缩原则

1. 保留所有已产出的评论：hunk 编号 + severity + 正文
2. 保留所有已确认的审查结论：读了什么、查到什么
3. 保留当前正在审查的 hunk 的进度
4. **不要编造历史里没发生的事**。历史里没有的信息写「（无）」，不要猜
5. 历史里凡是被替换成「内容已清理」的工具结果，表示该内容**当前不在上下文里、
   但可以重新读取** —— 在 `Files Examined` 段记录它的位置即可，不要试图复现内容

## 输出格式

严格按以下 7 个二级标题输出。某段没有内容就写「（无）」，**不要省略标题**：

## Primary Request
本组审查的目标：审查哪个包、多少个 hunk。

## Completed Hunks
（本段为历史快照，权威进度以随后的 <review_status> 为准）
已审查完毕的 hunk 及其结论摘要。

## Key Findings
已产出评论的浓缩版，逐条保留 hunk 编号与 severity。

## Files Examined
读过 / 搜过的位置，按「路径:行范围（读了什么）」列出。
信息取自历史里「内容已清理」的占位符。

## Current Work
**本段必须详细** —— 它是「最近几轮原始消息」的替代品：
正在审查哪个 hunk、它在哪个文件的哪些行、已经读了什么、已经查到什么、
还差什么才能下结论。

## Next Step
接下来要做的事。

## Errors and Fixes
历史里出现过的误判、被工具拒绝的调用及其原因。
""")

# 压缩用的 user 段：背景（审查 system + 要求 + diff）在前，待压缩的历史在中，
# 权威进度在后。**不绑工具**，压缩是一次纯信息搬运。
COMPRESSION_USER = _t("""\
## 本组审查的背景（原文，供你理解上下文）

### 审查用的 system 提示词

{review_system}

### 审查要求与本组 diff

{background}

## 待压缩的审查历史

{history}

## 当前 hunk 进度（权威来源）

{status}

## 你的任务

把以上审查历史压缩为结构化摘要。只输出摘要本身，不要有任何前后缀。
""")
