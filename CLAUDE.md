# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
uv run pytest tests/unit        # 单测，不调 LLM，快
uv run pytest tests/e2e         # e2e，**真实调用 DeepSeek**，慢且花钱
uv run pytest tests/unit/test_main_loop.py::test_plain_text_after_tool_call_does_not_end_loop   # 单个用例

ruff check . && ruff format src tests
mypy src tests                  # 两个一起传：只传 tests 会解析不到 prism 包
uvx pyright --pythonpath .venv/bin/python src/   # 第二个类型检查器，不进依赖
```

**e2e 必须用 `uv run`**：smoke test 起子进程调 `prism`，直接跑 `.venv/bin/pytest`
时它不在 PATH 上，会以 `FileNotFoundError` 失败。

本机 `prism` 是通过软链接进 PATH 的（`~/.local/bin/prism` → `.venv/bin/prism`），
任意目录可直接调。e2e 之外还有 `~/Projects/python/prism-min-test` 作为走查目标（见末尾）。

## 架构：三段流水线

```
diff → 过滤 → hunk 归属符号 → 按包分组        ← 确定性，零 token
                                  ↓
                   每个「审查单元」跑一次 ReAct 循环   ← core/agent/
                                  ↓
                   按 hunk 定位评论 → 排序 → 终端渲染  ← core/output/
```

审查前四段各有一个 CLI 子命令（`diff` / `filter` / `hunk-map` / `group`），
**它们是给人看的干跑入口**，用在花钱之前逐段核对范围，不是 `review` 的内部实现。

所有子命令都要 `--path` 指定被审查仓库（**必填，不回落当前目录**：审查会花 token、
会把代码发给外部 API，审错仓库的代价比多敲一个参数大）。`--base` / `--head`
默认 `main` / `HEAD`，以 merge-base 为基准，只统计 PR 实际引入的改动。

## 循环的语义（改 `core/agent/` 前必读）

`create_agent` 的默认退出条件是「模型不再调用工具」，**prism 的退出条件不同**：
纯文本不停机，而是注入提醒继续。这条靠 `after_model` 返回 `jump_to="model"` 推翻
（`jump_to` 的优先级高于经典退出判定）。

- `middleware.py` 的 `review_progress`（`before_model`）—— 追加进度快照、判定语义退出
- `middleware.py` 的 `review_pacing`（`after_model`）—— 记轮次、纯文本时注入提醒
- 轮次上限交给框架的 `ModelCallLimitMiddleware`，别自己数

**时序**：`after_model` 在**工具执行之前**跑，`before_model` 在**工具执行之后**跑。
要看「工具跑完的状态」必须用 `before_model`。

## 状态与并发（最容易踩的一块）

模型一轮可以并行发多个工具调用，它们在**同一个 super-step** 里执行，返回的
`Command.update` 会被同时写入。LangGraph 对**没有 reducer 的键**只允许一步一写 ——
两个工具撞在一起就是 `InvalidUpdateError: At key 'xxx': Can receive only one value
per step`，整组审查崩掉（实测 21 轮里有 8 轮是并行调用）。

因此：

- `hunk_status` 有 reducer，**工具必须返回增量**（只含本次改动的键）。返回全量快照
  会让并行写互相覆盖 —— **reducer 和返回形状是配套的，只改一边不成立**
- 工具**不要**去写别的工具也会写的键（`end_done_count` 的重置为此挪到了 middleware）
- `_last_tool_calls` 只认**遇到的第一条 AI 消息**：纯文本响应的 `tool_calls` 是空列表，
  继续往前翻会误判成「有工具调用」，循环提前收尾、`exit_reason` 读不出来

## 提示词与坐标（改 `guidance/` 前必读）

提示词里**所有行号必须是被审查版本（head）的坐标** —— 和 `file_read` 读到的是
同一套。给模型一个不是 head 的坐标，它照着读就是读到隔壁代码。

（这条只管**送进模型**的坐标。评论定位出来的是**输出**坐标，那里两版都有，
由 `Comment.side` 标明 —— 见下面「评论定位」一节。）

这条踩过两次，两次都不报错：

- **纯删除的 hunk 归属取自旧版符号表**，行号和 head 对不上（差值正好等于删除
  行数）。所以 `HunkWithSymbol.head_lines` 是**按名字在 head 侧重查**的，不是
  照抄 `symbol.start_line`；查不到（符号被整体删除/改名）时给 `None`，
  提示词里**不给** `lines`，而不是给个旧坐标
- **hunk 位置**用 `old="起始行,行数" new="..."`（git 的 `@@` 头写法）。
  区间写法（`起始-结束`）表达不了纯删除的**空区间**；整文件删除时
  `new_start == 0`，会退化成「第 0 行」这种废坐标。`0` 在这个写法里天然就是
  「该侧没有内容」

`display_range` 仍保留旧的「回退旧行号」语义 —— 那是给 `prism hunk-map` 这个
**人看的干跑入口**用的，对着 `git diff` 核范围时 git 自己的口径就是旧坐标。
别为了「一致」把它也改掉，提示词已经不走它了。

诊断经验：`mapper.py` 的丢弃计数**分两类**（`outside_entity` / `no_symbol_table`）。
合成一个数时，「重命名 + 纯删除被丢弃」的信号被写成了「改动只涉及注释或空行」，
bug 藏了很久。**丢弃日志说错原因，比丢东西本身更难发现。**

## 评论定位（改 `core/output/locate.py` 前必读）

审查后第一步：把 LLM 报的评论挂到具体行号上。**零 LLM**，同一输入同一输出。
两个维度，正交：

- **层**（`locate_level`）：本 hunk 内 → 本组其它 hunk → 未定位聚合
- **侧**（`side`）：`RIGHT` = 新增行、行号是**被审查版本**的坐标；`LEFT` = 删除行、
  行号是**基准版本**的坐标。取值照抄 GitHub 的 `side` 参数，将来直接透传

- **两侧都要能挂**。GitHub 的限制从来不是「只能评论新行」，而是「行必须在 diff
  里」—— 删除行是一等公民。按「只搜新增行」跑实测 **71%**（7 条丢 2 条，全是
  删除类），改成两侧后 **100%**（见 `02-评论定位和反思/13-评论定位实现反馈.md`）
- **侧必须标出来**：`LEFT` 的行号拿到当前文件里找是找不到的。渲染标
  `（已删除）`，排序也按侧分组 —— 不让两套坐标直接比大小
- **只有一个 `side`，没有 `start_side`**。GitHub 那个字段是为「跨左右的多行范围」
  准备的，而我们的匹配永远单侧（needle 先按行首记号拆过），加了就是只有一个
  实现的接口
- **行号推进**：某一侧只数**本侧**的记号（`+` 从 `new_start` 数、`-` 从
  `old_start` 数）。多推一格不报错、不越界，只是指向隔壁几行 —— 这类错误没有
  任何信号，靠变异测试 + 回 git 里逐字读来守
- **层优先于侧**：本 hunk 的 LEFT 命中比别组的 RIGHT 命中精确。每层里按
  `+`行 → 无记号 → `-`行 → 无记号（另一侧）的顺序试完，才算这层没中
- **组内命中要改写 `hunk_id`**：`path` 是从它查出来的，保留模型原报的编号会让
  `path` 与 `start_line` 指向不同文件，两个字段各说各话还都不报错
- **空片段先判**（`empty_existing_code`），不能等两层搜索都失败再报 `no_match`
  —— 那是个指向错误方向的提示（同配置层的「先查未知键，再读值」）
- 定位在 `_run_groups` 里**每组跑完一次**，不在全局收尾处：反思阶段是每组一次
  LLM 调用，它要的素材（本组 hunk + 评论 + 定位元数据）都在同一个循环体里

**已知缺口（接 GitHub 时要确认）**：文件改过名时旧侧的行住在**旧路径**下，而
`Comment.path` 只有新路径 —— 终端会显示 `新名字:48（已删除）`。GitLab 的
position 是显式的 `old_path`，GitHub 只有一个 `path`，它对 LEFT 评论要哪一侧
待核实。

## 框架缺口（LangChain v1，都有实测依据）

| 缺口 | 处置 |
|---|---|
| `create_agent` 的 `middleware` 参数只填了 `AgentMiddleware` 三个类型参数中的两个（漏 `ResponseT`），连它自带的 `ModelCallLimitMiddleware` 都过不了 mypy | `cast`，注释写明是框架签名问题 |
| 四个 overload 都不把 `state_schema` / `context_schema` 串进返回类型，推出来是 `AgentState[Any]` / `Any` —— `invoke` 的 `context=` 传错也不报错 | `agent.py` 手写 `Agent` 别名（**比框架推出来的更精确**）+ 出参 `cast` |
| `@wrap_model_call` / `@wrap_tool_call` 装饰**模块级函数**时，返回的中间件实例是**全局共享**的 | 用**类式中间件**，每组审查由 `build_review_agent` 造一个实例，跨轮次状态才不会串到下一组 |
| `@tool` 的 **`parse_docstring` 默认 False** —— docstring 里 `Args:` 的参数说明**一个字都到不了模型** | 四个工具统一 `@tool(parse_docstring=True, description="…")`：描述走 `description=`、参数说明走 docstring 的 `Args:` |
| **`@tool(args_schema=...)` 会让 `runtime: ToolRuntime[...]` 注入失效** —— 显式给了 args_schema，框架就不再从函数签名推导，`runtime` 从 `get_input_schema()` 里消失，调用时报 `TypeError: missing 1 required positional argument: 'runtime'`。**四个工具全吃 runtime，所以全都不能用 `args_schema`** | 一律不给 `args_schema`。嵌套结构（`report_hunks` 的 `reports`）靠**类型注解**表达，一样能出嵌套 schema |
| `parse_docstring=True` + `args_schema=` 一起给时，工具描述会**静默变成参数模型的 docstring** | 同上：不给 `args_schema`。另注意**参数模型的 docstring 本身也会进 schema**，所以要写成给模型看的话，设计理由用 `#` 注释 |
| 工具定义**入站**还要过一道转换：`convert_to_openai_tool` 会剥掉所有 `title`、剥掉参数顶层的 `description`、把 `$defs` + `$ref` **内联展开** | 日志与对账一律用 `convert_to_openai_tool(tool)["function"]`，别拿 `tool_call_schema.model_json_schema()`（那是转换前的，`report_hunks` 两份差 2000 vs 1279 字节） |
| `recursion_limit` 按 **super-step** 算，`create_agent` 实测每轮约 **6.1** 步 | `max_rounds * 7 + 20` |

**不使用已废弃的 API**（尤其 `langchain-classic` 包）；其余尽量用框架，别自己重造。

**工具定义的形态约定**：四个工具一律 `@tool(parse_docstring=True, description="…")`。
`description=` 是**给模型看的**那一段；docstring 的 `Args:` 是**参数说明的来源**；
`Returns:` / `Notes:` 之类给开发者看的内容不会发出去（`Notes:` 要发就把内容并进 `description=`）。
不要为了「都用 Pydantic」改用 `args_schema=` —— 见上表，它会让 `runtime` 注入失效。

v1 的「框架味」在 **`langchain` 包**（agents / middleware），不在 `langchain-core`
——后者只是底座。反方向也要警惕：`langchain` 已把 chains 移到 `langchain-classic`，
所以把流水线包成 LCEL 链是往 v1 弱化的方向走。

## 代码约定

注释、docstring、提交信息、设计文档**一律用中文**；`models/` 下**一个模型一个文件**
（同一个域的多张表可以同文件，见 `models/ast.py`）。
`ruff check .` + `mypy src tests` 是硬门槛，改完必须全绿（配置见 `pyproject.toml`）。

## 配置

**默认值只有一份：`src/prism/config/default.toml`。** 代码里不写第二份。

```python
from prism.config.settings import settings

max_rounds = max(settings.review.min_rounds, n * settings.review.rounds_per_hunk)
```

优先级 `环境变量 > default.toml`；唯一走环境变量的是 `DEEPSEEK_API_KEY`（密钥不落盘）。
读取见 `config/settings.py`，它只做三件事：读 toml、按键记进 dataclass、拦下配错的键。

**什么进文件、什么不进**，判据是「改它会不会改变我对 LLM / git / 框架的契约」：

| | 例子 |
|---|---|
| **进**（旋钮，配错了看得见） | 窗口、思考档位、压缩阈值、轮次公式、工具上限、日志级别 |
| **不进**（契约，配错了*看不见*） | 标签名 `<diff>` / `<review_status>`、`H-01` 编号格式、退出原因码、severity 排序、定位层级与失败原因码、提示词全文、过滤白名单、`REFETCHABLE_TOOLS` |

把契约写进文件等于让它变成「可以配错的东西」，而配错的后果是静默失效。
完整清单与理由写在 `default.toml` 开头的注释里。

三个**刻意不做成旋钮**的值：`api_key`（环境变量）、`temperature`（思考模式下被 API
忽略）、`recursion_limit` 的倍数（由框架实测「每轮约 6.1 个 super-step」推出来）。

读取是**硬报错**的：未知键、缺键、类型不符都在**读的那一刻**拦下并报出文件与键路径。
`max_lines = "500"` 这种 tomllib 照收不误，混到工具里才炸就离原因太远了。
**进一层先查未知键、再读值** —— 顺序反了的话，把 `reserve_tokens` 拼成
`reserve_token` 会报成「缺少 reserve_tokens」，那是个指向错误方向的提示。

`SCHEMA`（段 → dataclass + 排除字段）是读取与校验**共用**的那份映射，期望的键由
dataclass 字段推出来，不另写键名清单 —— 否则又多一处会漂移的东西。

## 模型与成本（DeepSeek）

- **思考模式默认开启、默认档位 `high`** —— 不显式配就是一直在按最贵档付费。
  `default.toml` 里显式写了 `reasoning_effort = "low"`
- **思考模式下 `temperature` 被 API 忽略**
- `ChatOpenAI` **不提取** `reasoning_content`（思考正文）；`ChatDeepSeek` 入站捕获、
  出站序列化却丢弃 —— `core/llm/client.py` 的子类补回。**实测 `deepseek-flash`
  不强制回传**，所以那是防御性的，不是修一个正在发生的 bug
- model / base_url 在 `config/default.toml` 的 `[llm]`，只有 api_key 从环境变量读

## 日志

两个模块分工明确：`observability/logging.py` 只管**怎么渲染**（`PrismHandler`），
`observability/llm_trace.py` 只管**打什么内容**（`ReviewTrace` 中间件）。
业务代码照常 `logger.info(...)`，头部（时间/级别/短路径）全自动。

三个开关必须显式设，都是实测踩出来的：

- `Console(stderr=True)` —— 日志走 stderr，stdout 留给审查结果
- `Console(soft_wrap=True)` —— **关掉折行**，否则正文被插入真换行、不再是原文
- `markup=False` —— 否则日志里的 `[方括号]` 被当成 rich 标记**整段消失**

**在 pytest 里看日志**要同时给 `PRISM_LOG_LEVEL`（环境变量）与 `-s`（关捕获；
pytest 的捕获是 **fd 级**的，绕不过）。**别用 `--log-cli-level`** —— 它走 pytest
自己的格式，不是项目约定的格式。conftest 的 `pytest_collection_finish` 会自诊断。
PyCharm：Environment variables 填 `PRISM_LOG_LEVEL=INFO`，Additional Arguments 加 `-s`。

`PRISM_LOG_LEVEL` **优先于**配置文件的 `[log].level` —— 临时看一次轨迹不该要求改文件。
传了非法值（如 `VERBOSE`）时退回 `INFO`，不回落到配置值：那是用户明确的一次覆盖意图，
静默换成另一个数反而难查。

## 验证怎么做

这个项目的规矩是**结论必附证据**：

1. **变异测试** —— 改完逻辑先把实现改回旧写法，确认对应用例**真的会失败**
2. **真实数据对账** —— 拿真实仓库跑一遍，把数字与 `git diff` 自己的口径核。
   有个现成的例子：`tests/test_locate_against_git.py` 把定位出来的行号**回 git
   里逐字读回来比对** —— 期望值是 git 给的，不是我们自己写进用例里的，所以它
   能证伪「我以为的行号规则」。两个走查仓库都在时才跑（改定位必跑）
3. e2e 打真实接口，**偶发红灯**（实测遇到过 1 次）—— 先重跑一次，别直接当回归

对账用的两个项目：

- `~/Projects/python/prism-min-test` —— 2 个 Go 文件、6 个 hunk、**单审查单元**，
  约 5 轮收尾。`NOTES.md` 列了埋的 6 处问题，专供走查与断点调试
- `~/Projects/python/prism-tests` —— 多组、覆盖边缘情况。它维护三个 diff 产物
  （`diff.txt` / `diff-U0.txt` / `diff-stat.txt`，在 `.gitignore` 里）供**人工研读**，
  所以**改过它的 Go 文件或切过分支之后必须跑 `./regen-diff.sh`**，否则读到的是过期 diff
- `~/Projects/golang/open-code-review`（OCR）—— 大样本对账，用于规模类问题

```bash
PRISM_LOG_LEVEL=INFO prism review --path ~/Projects/python/prism-min-test
```

## 设计文档

在 `/Users/caojunfeng/Projects/golang/open-code-review/acr-min/`，**一个阶段一个目录**：

| 目录 | 设计（冻结） | 反馈 / 取舍（活） |
|---|---|---|
| `00-最小闭环/` | 1~5 | 6（实现偏离，五类分）、7（设计复盘） |
| `01-上下文压缩/` | 9 | 10（实现反馈）、11（设计取舍） |
| `02-评论定位和反思/` | 12 | 13（实现反馈） |

- **设计文档冻结、永不回头改**；实现与它不符时**记进反馈文档**
  （唯一例外：反馈文档里写明「这条必须回改设计文档」的，如 13 号 §六 E1）
- 反馈文档按「文档缺口 / 文档写错 / 前提失效 / 工程取舍 / 文档不一致」五类分，
  并附**证据**（门槛数字 + 变异测试表 + 真实数据对账）
- **`acr-min` 不在 git 里**，别试图提交

- **实现发生变更后要同步反馈文档**：改完代码就顺手记一笔，别攒着
