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
                        排序 → 终端渲染             ← core/output/
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

## 框架缺口（LangChain v1，都有实测依据）

| 缺口 | 处置 |
|---|---|
| `create_agent` 的 `middleware` 参数只填了 `AgentMiddleware` 三个类型参数中的两个（漏 `ResponseT`），连它自带的 `ModelCallLimitMiddleware` 都过不了 mypy | `cast`，注释写明是框架签名问题 |
| 四个 overload 都不把 `state_schema` / `context_schema` 串进返回类型，推出来是 `AgentState[Any]` / `Any` —— `invoke` 的 `context=` 传错也不报错 | `agent.py` 手写 `Agent` 别名（**比框架推出来的更精确**）+ 出参 `cast` |
| `@wrap_model_call` / `@wrap_tool_call` 装饰**模块级函数**时，返回的中间件实例是**全局共享**的 | 用**类式中间件**，每组审查由 `build_review_agent` 造一个实例，跨轮次状态才不会串到下一组 |
| `@tool` 的 **`parse_docstring` 默认 False** —— docstring 里 `Args:` 的参数说明**一个字都到不了模型** | 要参数描述就显式开；但开了会剥掉 `Notes:` 段，先确认那段的规则能挪走（`report_hunks` 就是为此没开） |
| `recursion_limit` 按 **super-step** 算，`create_agent` 实测每轮约 **6.1** 步 | `max_rounds * 7 + 20` |

**不使用已废弃的 API**（尤其 `langchain-classic` 包）；其余尽量用框架，别自己重造。

## 模型与成本（DeepSeek）

- **思考模式默认开启、默认档位 `high`** —— 不显式配就是一直在按最贵档付费。
  `LLMConfig.reasoning_effort` 默认 `low`
- **思考模式下 `temperature` 被 API 忽略**
- `ChatOpenAI` **不提取** `reasoning_content`（思考正文）；`ChatDeepSeek` 入站捕获、
  出站序列化却丢弃 —— `core/llm/client.py` 的子类补回。**实测 `deepseek-flash`
  不强制回传**，所以那是防御性的，不是修一个正在发生的 bug
- model / base_url 固定在 `models/llm.py`，只有 api_key 从环境变量读

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

## 验证怎么做

这个项目的规矩是**结论必附证据**：

1. **变异测试** —— 改完逻辑先把实现改回旧写法，确认对应用例**真的会失败**
2. **真实数据对账** —— 拿真实仓库跑一遍，把数字与 `git diff` 自己的口径核
3. e2e 打真实接口，**偶发红灯**（实测遇到过 1 次）—— 先重跑一次，别直接当回归

对账用的两个项目：

- `~/Projects/python/prism-min-test` —— 2 个 Go 文件、6 个 hunk、**单审查单元**，
  约 5 轮收尾。`NOTES.md` 列了埋的 6 处问题，专供走查与断点调试
- `~/Projects/python/prism-tests` —— 多组、覆盖边缘情况；改过 Go 文件后跑 `./regen-diff.sh`

```bash
PRISM_LOG_LEVEL=INFO prism review --path ~/Projects/python/prism-min-test
```

## 设计文档

在 `/Users/caojunfeng/Projects/golang/open-code-review/acr-min/00-最小闭环/`：

- **1~5 号冻结**（原始设计），**6 / 7 号是活的**
  - 6 = 实现与 1~5 不符之处，按「文档缺口 / 文档写错 / 前提失效 / 工程取舍 / 文档不一致」五类分
  - 7 = 设计复盘，含 20 条踩坑清单与「共性」归纳
- **那个目录不在 git 里**，别试图提交

实现与文档不符时，**记进 6 / 7，不要回头改 1~5**。
