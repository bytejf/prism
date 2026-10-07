# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
uv run pytest tests/unit        # 单测，不调 LLM，快 ← **回归跑这个**
uv run pytest tests/e2e         # e2e，**真实调用 DeepSeek**，慢且花钱
uv run pytest tests/manual      # 手动调试 / 校准脚本，多数真调 DeepSeek，别进回归
uv run pytest tests/unit/test_main_loop.py::test_plain_text_after_tool_call_does_not_end_loop   # 单个用例

ruff check . && ruff format src tests
mypy src tests                  # 两个一起传：只传 tests 会解析不到 prism 包
uvx pyright --pythonpath .venv/bin/python src/   # 第二个类型检查器，不进依赖
```

**目录的判据是「要不要人驱动」，不是「花不花钱」**：

| 目录 | 性质 | 进回归？ |
|---|---|---|
| `tests/unit/` | 断言套件，不联网 | ✅ 就它 |
| `tests/e2e/` | 断言套件，**真调接口** | ❌ 慢且花钱 |
| `tests/manual/` | **不是断言套件** —— 给断点、给校准、要人盯着跑 | ❌ 手动 |
| `tests/test_locate_against_git.py` | 断言套件，免费，但要两个走查仓库 | ❌ 仓库不在就 skip |

`tests/manual/` 里每个文件的开头都写了「跑法 + 看什么变量 + 花不花钱」。
它们**故意保留 `test_` 前缀**：IDE 的测试运行器靠它才认得出，能单独 Debug 一个
用例。（所以**别**在 conftest 里 `collect_ignore` 掉这个目录 —— 那会让 IDE 也
找不到用例。）

**e2e 必须用 `uv run`**：smoke test 起子进程调 `prism`，直接跑 `.venv/bin/pytest`
时它不在 PATH 上，会以 `FileNotFoundError` 失败。

本机 `prism` 是通过软链接进 PATH 的（`~/.local/bin/prism` → `.venv/bin/prism`），
任意目录可直接调。e2e 之外还有 `~/Projects/python/prism-min-test` 作为走查目标（见末尾）。

## 架构：三段流水线

```
diff → 过滤 → hunk 归属符号 → 静态分析 → 分组+装箱   ← 确定性，零 token
                                  ↓
                   每个「审查单元」跑一次 ReAct 循环   ← core/agent/
                                  ↓
              定位 → 反思（每组一次 LLM 调用）        ← core/output/ · core/reflection/
                                  ↓
                        排序 → 终端渲染             ← core/output/
```

审查前各段都有一个 CLI 子命令（`diff` / `filter` / `hunk-map` / `static` /
`group`），**它们是给人看的干跑入口**，用在花钱之前逐段核对范围，不是 `review`
的内部实现。静态分析夹在「hunk 归属符号」与「按包分组」之间：它的发现要按文件
分桶进各组，而「关联到哪个 hunk」需要映射结果 —— 这一段**依赖外部二进制**
`golangci-lint`（没装就跳过，见下）。

所有子命令都要 `--path` 指定被审查仓库（**必填，不回落当前目录**：审查会花 token、
会把代码发给外部 API，审错仓库的代价比多敲一个参数大）。`--base` / `--head`
默认 `main` / `HEAD`，以 merge-base 为基准，只统计 PR 实际引入的改动。

## 循环的语义（改 `core/agent/` 前必读）

`create_agent` 的默认退出条件是「模型不再调用工具」，**prism 的退出条件不同**：
纯文本不停机，而是注入提醒继续。这条靠 `after_model` 返回 `jump_to="model"` 推翻
（`jump_to` 的优先级高于经典退出判定）。

- `middleware.py` 的 `review_progress`（`before_model`）—— 追加进度快照、判定语义退出
  与预算
- `middleware.py` 的 `review_pacing`（`after_model`）—— 记轮次、累加用量、
  纯文本时注入提醒
- **轮数不是约束**（摘掉了，理由见下面「预算与止损」）

**时序**：`after_model` 在**工具执行之前**跑，`before_model` 在**工具执行之后**跑。
要看「工具跑完的状态」必须用 `before_model`。

## 预算与止损（改 `budget.py` / `middleware.py` 前必读）

**约束用资源本身表达**：钱与时间，不数轮。实测一次审查的成本构成是
**输出 54% / 未命中输入 35% / 命中输入 11%**，而同一轮的成本能差几十倍
（一轮 `out 23` 调 end_done，另一轮 `out 1555` 报 7 条评论 + 366 思考 token）——
轮数是代理指标，数不出来这个差别。

五道闸，**各管一类，日志里必须能分清是哪一道响**：

| 闸 | 管什么 | 落点 | 退出原因 |
|---|---|---|---|
| `no_progress_exit_rounds = 8` | 转圈不干活（连续 8 轮没有 report_hunks） | `review_pacing` | `no_progress` |
| 单组钱 / 时间（2 元 / 300 秒） | 一组跑飞 | `ReviewProgress.before_model` | `budget_cost` / `budget_time` |
| 整条钱 / 时间（10 元 / 20 分钟） | 组数太多或前几组太贵 | `_run_groups` 的循环 | （停剩下的组） |
| 工具失败（连续 3 提醒 / 累计 5 退出） | 把预算烧在一直失败的调用上 | `ToolFailureBudget` | `tool_failures` |
| `max_super_steps = 1400` | 图失控（**不该发生**） | `recursion_limit` | `recursion` |

四条不能动的规矩：

- **语义退出优先于预算**：审完了就是审完了。让预算把一个已完成的组标成「因预算
  停止」，是给用户一个假的缺口（`exit_reason_of` 的顺序就是这个意思）
- **拿不到 `usage_metadata` 不算零**：记进 `UsageTotals.missing` 并警告一次 ——
  成本闸完全依赖它，缺了它就是「不报错的错链路」，那时只剩时间闸与 recursion
- **上限填 0 表示不限制**，不是「立刻停」：配错的最坏结果是少一道闸，而不是所有
  review 一开局就停
- **任何闸触发都带缺口退出，且把缺口报出来**：`MainLoopResult.pending_hunk_ids`
  一路传到 `ReviewStats.pending_hunks`，在报告头部写「有 N 个 hunk 没能审完」。
  不报的话「评论数：0」看起来就像「这段代码很干净」

三处容易写错的细节：

- **用量要按轮增量累加**（`review_pacing` 里加进 `state["usage"]`），不能最后扫
  消息列表 —— 压缩会把历史消息删掉，账会漏。反思那边不压缩，扫终态即可
- **反思的用量也算进全局账**：组级闸看不到它（那时循环已经结束），漏了会系统性
  低估「每组一次反思调用」
- **`stop_reason` 是单键、先写者胜**：同一轮有两个停手时，原因由中间件列表的顺序
  定（预算排在前）

**为什么可以不要轮数上限**：实测小组 4/30 轮、大组 12/250 轮，余量 7~20 倍；
换成预算后单组 2 元 ≈ 70~150 轮、300 秒 ≈ 50 轮，都比它紧。它原来的三个职责
（转圈、慢烧、图失控）分别由 no_progress、预算、`recursion_limit` 接管。

`recursion_limit` 取**绝对数**而不是「轮数 × 倍数」：实测 6.1 步/轮，乘出来的余量
只有 15%，框架或我们多加一个钩子就会让它反过来先响 —— 而它响是我们的 bug，
所以它只记 ERROR + 带缺口退出，绝不抛异常穿出去（用 `stream` 保住中途状态）。

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

## 跨文件改动查询（改 `core/tools/file_read_diff.py` 前必读）

分组**按包排序、按预算装箱**：一个跨包改动（改接口 + 改实现 + 改调用方）**可能**被
切进不同的审查单元（也可能因为都没装满而同组 —— 那是运气，不是保证）。补足手段
只有一个 —— `file_read_diff(path)`：按路径取该文件的 diff，
让模型自己判断「那个文件跟着改了吗」。**刻意不做调用图分组**，理由与触发条件写在
设计文档 15 号（精度实测见 `4-建图技术方案`：`(*Agent).Run` 的 57 个 callee 里只有
约 15 个是项目内函数，噪音比 73%；**错误的分组比不分组更糟**）。

- 数据是 `ReviewContext.diffs`（`build_diff_index` 建的索引），**不重读 git、不读工作区**
  —— 一次审查内 diff 不变，查的是已经在内存里的那份
- 索引**只含可审查文件**，在 `review` 里**过滤之后**建：被过滤的文件不进索引，
  也就不会通过这个工具暴露出去（与压缩的清理判据同源）
- **改名文件的旧路径也建别名**（`setdefault`，规范路径优先）：提示词里带着
  `renamed-from`，模型可能拿旧名字来查，不建别名它拿到的是「没有改动」这个错答案。
  但**别在工具描述里承诺「新旧都能查」**：别名只在那个名字没被本次改动里的另一个
  文件占用时才存在（「A 改名成 B 又新建了 A」时，`A` 指向的是新建的那个 —— 那也
  是 `A` 这个路径在当前变更集里的正确含义）。模型端只写「文件相对路径」
- **输出是 git 原味、不带 `hunk_id`**：`<hunk id>` 是本组待审清单的编号，别的组的
  hunk 编不出号，硬编一个等于邀请它越界评论，而 `report_hunks` 只认本组编号 ——
  白费一轮。hunk 头的行数一律显式写出（`@@ -5,1 +5,1 @@`），不学 git 在行数为 1 时省略
- **索引为空要报错，不能回「这个文件没改」**：装配错误会退化成「每个查询都说没改」
  → 模型判断「调用方没同步改」→ 报出假问题。一条不报错的错链路
- 结果进压缩的**可重新获取**白名单（`REFETCHABLE_TOOLS`），压缩时清成占位符
- **三个工具的输出必须能互相喂**：路径一律是仓库内相对路径，拿到就能直接查。
  `code_search` 带 rev 搜时 git 会加 `<rev>:` 前缀（`HEAD:db.go:12: ...`），
  代码里剥掉 —— 留着它，模型得自己认出那不是路径的一部分，认错就白查一次
- 提示词里**不出现「组」这个概念**：只谈「文件」，判断权交给模型；硬约束里写明
  「别处的改动只作为判断依据，不要在它上面报评论」
- 「查不到」= 没改动**或**不在审查范围内，两者在工具这里**是同一句话**：
  `file_read` 能读到测试文件、`code_search` 能搜到它，只有这个工具多设一道卡，
  三个工具对「过滤」的态度就不一致了
- 改了工作方式段 / 硬约束段 / 这个工具，跑一次
  `tests/manual/check_cross_file.py`（现场搭一个「改了接口、别的包的实现与调用方
  没跟着改」的单组仓库，真调 LLM，约 10 秒）。实测模型会 `code_search` 找到实现方
  → `file_read_diff` 确认它没改 → 把结论挂在接口那一行上；两个走查仓库都触发不了
  这个场景（一个单审查单元、一个组内同包）

## 工具的参数是信任边界（改 `core/tools/` 前必读）

**入参是模型给的** —— 这是本项目唯一一处「外部输入直接送到系统调用」的地方。
三个吃路径的工具里，有两个把参数交给外面，而外面的反应各不相同：git 自己拒越界
pathspec（`fatal: outside repository`，英文 stderr 还带**本机绝对路径**，那段文字
要发给外部 API）、`subprocess` 对 NUL 直接抛 `ValueError`、pygit2 给 `KeyError`。
**校验收在我们的这一侧**（`core/tools/paths.py`），失败就长一个样。

- `check_relative_path`：空 / NUL / 绝对路径 / `..` 段。`check_no_nul` 单独给
  「不是路径但要进 argv」的参数（`code_search` 的 `pattern`）
- **别放宽它去迁就 git**：合法 pathspec（`*.go`、`:(top)*.go`）一样要通过，
  校验不能比 git 自己更窄 —— 用例里有正面清单
- `file_read_diff` **不走校验**：纯字典查询，越界路径的结局就是「查不到」，
  没有要防的东西。**别照着 `file_read` 抄一份进来**
- 字符串校验唯一挡不住的是**符号链接**（`link.go` 字面合法）：由读取侧再挡一道
  （`_resolve_inside_project` 的 `resolve()` + `is_relative_to`）。走 git 那条路
  读的是 blob 内容，软链接本身就是一段文本，天然不跟随
- **评论的 `path` 模型改不了**：`report_hunks` 只收 `existing_code`，path 由
  `hunk_id` 查出来。这是一处**结构上**的防护，别为了「方便」让它自己填路径
- 工具体内的异常**由 `tool_error_guard` 兜**（见框架缺口表）：整组不因为它崩，
  但也不是静默 —— ERROR + 完整堆栈

验证方式：`tests/unit/test_tools.py` 的「路径安全」一节（含 symlink 逃逸、
NUL、`--` 挡住选项注入），变异 9/9（`/tmp/mutate_paths.py`）。

## 分组与预算（改 `core/grouping/` 前必读）

**算法只有一条：所有 hunk 排成一列（同一个包的挨在一起），从上往下一路合并进当前
组，装不下就封口、开新组接着合并。** 包只决定**顺序**，不决定边界 —— 所以

- 一个大包会被**切**成几组（装不下时），几个小包也会被**并**进同一组（都没装满时）
- 于是组数由预算决定，不由包数决定。`Group.package_path` 只是「首片所在的包」，
  跨包的组要用 `package_label` 显示成 `pkg/a → pkg/b`（CLI 与日志都走它）
- 实测：prism-tests（50 hunk / 13 文件）是 **1 组**；一个 2 万行的真实 PR 是
  **5 组**，每组 80~116 个 hunk、约 18~20k token

四条不能动的规矩：

- **切分粒度是文件，不是 hunk**：一个文件的 hunk 默认整块进同一组。只有**这个文件
  自己**就超预算（改了几百行的大文件）才退化成按 hunk 切，并记 WARNING ——
  到那一层再没有更细的边界
- **成本用真实渲染量出来**：`group_tokens` 直接调 `build_diff_block` /
  `build_static_block` 渲染候选内容再数字符，**不另写一套估算模型**。
  代价是装箱过程 O(k²) 次渲染（实测最大的组 116 个 hunk），换来的是不会出现
  「成本模型算得下、真渲染超了」这种漂移
- **每一处判断都用 `group_cost`（含固定开销）**，不要拿 `group_tokens` 单独跟预算
  比 —— 两个口径不一致时，会出现「这个文件明明能拆、却被塞成一个超预算的组」。
  这个 bug 是在真实仓库上被 `prism group` 的 ⚠️ 标记抓出来的（用例
  `test_no_group_exceeds_the_budget_when_splitting_is_possible` 钉住它）
- **预算是唯一的分组依据**，别再加「按包封口」这类边界：合并本来就是省钱的
  （固定开销 1111 token/组 + 一次反思调用都摊薄了），加的边界只会让它更碎

**预算管不到的地方**：一组的 hunk 越多，那一次会话也越长（`max_rounds` 是按 hunk
数给的），而会话长度没有别的上限（压缩触发点在 1M 窗口的 98%）。20k 的预算约束的是
**起手提示词**，不是会话。要收紧得另加一个「单组 hunk 数上限」，加之前先看实测。

`prism group` 会**顺带跑一次静态分析**：静态发现参与预算，不跑它，干跑出来的分组
与真审时的不一样，那就失去核对的意义了。输出里每组带 `约 N/20000 token`，
超预算的那组带 ⚠️。

## 静态分析（改 `core/static/` 前必读）

golangci-lint 跑一次，发现**只当线索**：不进评论列表、不进定位、不参与排序，
最终输出依旧只来自 LLM（设计 18 号 §2.1）。四步：**跑工具 → 解析 → 关联 hunk →
按预算注入**，后置校验在每组审完后核对引用。

- **跑的是被审查版本**。golangci-lint 只能分析工作区、读不了任意 rev，而
  `file_read` / `code_search` 读的都是 `--head` —— 所以先
  `git worktree add --detach <tmp> <head>` 检出、跑完删掉（实测 0.08s/次）。
  别改成直接跑工作区：工作区停在别的分支时，报出来的行号与待审代码错位，
  而错的行号比没有发现更坏。**GitHub CI 里这一步是免费的**：workflow 检出的
  本来就是 PR 的版本（`golangci-lint-action` 也一样）
- **失败一律降级，但「跳过」≠「没发现」**：没装 / 超时 / 退出码非 0 / stdout 不是
  合法 JSON / 目标不是 Go 模块 → WARNING + `skip_reason`，审查照常继续。
  CLI 与日志里两者**必须分开说** —— 合成一句「没有静态发现」会让「没装工具」
  看起来像「代码很干净」（同 `file_read_diff` 索引为空要报错那条）
- **只留可审查文件上的发现**：工具报的是整个仓库，我们按 reviewable 集合过滤。
  没改动的文件的发现进来只会挤占 token 并诱导模型评论没改过的代码
- **优先级按 linter 名映射**（`PRIORITY_BY_LINTER`），**不看工具自己的
  `Severity`**：实测那个字段大面积为空（errcheck 全是 `""`，gosec 全是
  `"medium"`，与规则严重性无关）。`typecheck` 关不掉（`--default=none` 也启用）、
  它报的是编译错误，按 high 收
- **不做 `--new-from-*`**：那是 CI 那套「只看改动的行」的口径。我们的口径是
  「改过的**文件**里的发现全要」—— hunk 关联与 token 截断已经各兜了一道
- **注入两处呈现**：独立成段列出（保住跨 hunk 的全局视角）+ hunk 上标
  `static_hint="S-01"`（只标编号不标严重性）。**hunk 的展示顺序**按关联发现的
  最高优先级排，**编号一律不动** —— 编号是报评论与查日志的坐标
- **`_render_diff` 按有序字典分组，不是 `groupby`**：展示顺序重排之后「同文件的
  hunk 相邻」不再天然成立，用 `groupby` 会让同一个文件渲染成两个 `<file>` 块
- **静态发现占分组预算**：分组装箱时把它们算进去（见下面「分组与预算」一节），
  所以提示词里不会有「装了但装不下」的发现 —— 注入那一侧**不再做二次裁剪**：
  再裁一次只会让「注入的」与「算过成本的」变成两批东西
- **注入的挑选只算一次**（`_run_groups` 里），注入与后置校验共用同一批编号
- **后置校验按最终保留的评论算**（反思删掉的那条不算数），只对**高优先级**未处理
  告警。设计 §2.11 的四条规则落地成三条：**不做「内容与某条发现相似 → 可能处理」
  这个分桶**（golangci-lint 是英文、评论是中文，文本相似度实测退化成「全部算未处理」，
  记在反馈文档里）
- **提示词的三层弱化不能改回去**：「提示」而不是「发现」、「工具标记」而不是
  「严重性」、中文「高优先」而不是 high/medium/low（避免与我们的 severity 混淆）。
  主任务段是 6 条**「问自己」式**维度，不是例子 —— 例子会让模型只查那一种问题
- 用了 v2 的 CLI（`--default=none` / `--output.json.path=stdout` /
  `--path-mode=abs`），v1→v2 的差异（`gosimple` 并入 `staticcheck`、
  stdout 里混着文本报告与统计摘要）都在 `analyzer.py` 的注释里

验证：`tests/unit/test_static.py`（32 条，**不依赖真实二进制**，喂真实的 JSON
载荷 + monkeypatch 造失败形态）；变异 15/15（`/tmp/mutate_static.py`，含
「去掉 `--show-stats=false`」「不按可审查文件过滤」「编号乱排」这类）；
真跑一次工具的场景在 `tests/manual/check_static.py`（现场搭一个 PR 新增的文件，
里面的问题 gosec / errcheck 抓得住，约 13 秒）。

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

## 反思（改 `core/reflection/` 前必读）

定位完成后，**每组一次 LLM 调用**做证伪：找出被 diff 直接推翻的评论，删掉。
**只删不增**，失败一律 fail-open（原样保留全部评论 + WARNING）。

- **走 `create_agent` 图**（`reflect.py`），工具是**实体工具**（`tools.py`，有执行
  体）。换成图是为了两件事：① 校验结果能作为**工具结果回给模型**，它可以在同一轮
  对话里改一版重报；② 挂 `ReviewTrace` 之后，「工具定义（含完整参数 schema）/
  每次工具调用的参数与耗时」与审查流程是同一套日志 —— 裸 `bind_tools(...).invoke(...)`
  那条路径上 `bind_tools` 只是把 schema 发给 API，**本地一个字都不打**
- **受理完就退出，不多调一次模型**：`ReflectionCompletion` 在 `before_model` 里
  看到 `reported > 0` 就 `jump_to="end"`。不管它的话 `create_agent` 会再问一次
  「还有吗」—— 那是一次纯浪费（审查那边用同一手法为 `end_done` 省最后一次调用）
- **校验分两档**（有意区分的）：`removals` 有一处不合法 → **整份退回**，不写状态，
  让模型重报一个完整结论（半份结论会让「失败即全留」说不清）；`misses` 不合法 →
  **丢掉那一条、其余照收**（漏报是提示不是评论，不能因为它按住删除结论）
- **保护性否决**（内存安全 / 并发 / 链接与声明一致性 / 行为兼容性变更）是
  **提示词层**的，与 OCR 一致 —— 这几类主题即使被 diff 推翻也不许删。实测有效：
  定标里那条「`Label` 加了 mutex 会死锁」明明被 diff 推翻，因为主题是并发被留下
- **信息边界**：只给 diff + 评论 + 定位元数据，**不给**审查者的工具调用结果
  —— 给了它会顺着那套逻辑「同意」，复核退化成自我背书
- **`REFLECTION_MAX_CALLS = 3`**（首次 + 一次纠错 + 一次余量）刻意不做成配置项：
  它是重试的安全阀，调小了会让纠错半途而废**且不报错**
- 提示词改动要看 `guidance/templates.py` 的 `REFLECTION_SYSTEM`：它开头就是
  「**默认答案是全部保留**」，「不算删除依据」那段是防过度删除的正面清单 ——
  两段都别删

验证这条链路用 `tests/manual/calibrate_reflection.py`（手动跑，不是断言套件）：
真实 diff + 混进去的假评论，量「该删的删没删、不该删的动没动」。**改提示词必跑。**

## 框架缺口（LangChain v1，都有实测依据）

**这里是摘要，只留影响日常写代码的那些**；完整台账（含报错原文、后果、证据位置）
在 `acr-min/03-添加工具和调用图/17-框架缺口清单.md`。新撞一个坑，两边都补。

| 缺口 | 处置 |
|---|---|
| `create_agent` 的 `middleware` 参数只填了 `AgentMiddleware` 三个类型参数中的两个（漏 `ResponseT`），连它自带的 `ModelCallLimitMiddleware` 都过不了 mypy（那张图还在用） | `cast`，注释写明是框架签名问题 |
| 四个 overload 都不把 `state_schema` / `context_schema` 串进返回类型，推出来是 `AgentState[Any]` / `Any` —— `invoke` 的 `context=` 传错也不报错 | `agent.py` 手写 `Agent` 别名（**比框架推出来的更精确**）+ 出参 `cast` |
| `@wrap_model_call` / `@wrap_tool_call` 装饰**模块级函数**时，返回的中间件实例是**全局共享**的 | 用**类式中间件**，每组审查由 `build_review_agent` 造一个实例，跨轮次状态才不会串到下一组 |
| `@tool` 的 **`parse_docstring` 默认 False** —— docstring 里 `Args:` 的参数说明**一个字都到不了模型** | 五个工具统一 `@tool(parse_docstring=True, description="…")`：描述走 `description=`、参数说明走 docstring 的 `Args:` |
| **`@tool(args_schema=...)` 会让 `runtime: ToolRuntime[...]` 注入失效** —— 显式给了 args_schema，框架就不再从函数签名推导，`runtime` 从 `get_input_schema()` 里消失，调用时报 `TypeError: missing 1 required positional argument: 'runtime'`。**五个工具全吃 runtime，所以全都不能用 `args_schema`** | 一律不给 `args_schema`。嵌套结构（`report_hunks` 的 `reports`）靠**类型注解**表达，一样能出嵌套 schema |
| `parse_docstring=True` + `args_schema=` 一起给时，工具描述会**静默变成参数模型的 docstring** | 同上：不给 `args_schema`。另注意**参数模型的 docstring 本身也会进 schema**，所以要写成给模型看的话，设计理由用 `#` 注释 |
| 工具定义**入站**还要过一道转换：`convert_to_openai_tool` 会剥掉所有 `title`、剥掉参数顶层的 `description`、把 `$defs` + `$ref` **内联展开** | 日志与对账一律用 `convert_to_openai_tool(tool)["function"]`，别拿 `tool_call_schema.model_json_schema()`（那是转换前的，`report_hunks` 两份差 2000 vs 1279 字节） |
| `ToolNode` 的默认兜底 **`_default_handle_tool_errors` 只把 `ToolInvocationError` 转成工具结果，最后一行是 `raise e`** —— 工具里任何一个意外异常都会穿过图、穿过 `run_review_agent`，把**整条 review 带走**（后面所有组不跑，前面审出来的评论也一起没，CLI 最后才渲染） | 注册 `tool_error_guard`（`middleware.py` 的 `wrap_tool_call`），放中间件链**最内侧**：ERROR + 堆栈记全，回给模型一句「换一种方式」，日志的 tool 行照样在（`ReviewTrace` 在它外侧） |
| `recursion_limit` 按 **super-step** 算，`create_agent` 实测每轮约 **6.1** 步 | `[review].max_super_steps`，取宽松绝对数（按轮数乘的余量只有 15%） |
| `stream()` 的 `config` 必须是 `RunnableConfig` 类型：裸 dict 字面量被推成 `dict[str, int]`，四个 overload 全不匹配（`invoke` 那边碰巧不挑） | 标注类型后调用 |
| **思考模式下不能强制 `tool_choice`** —— `with_structured_output` 的 `function_calling` / `json_schema` 两条路都靠强制，实测直接 `400 Thinking mode does not support this tool_choice`；`json_mode` 另要提示词里出现 "json" 且只保证合法 JSON、不保证 schema | 单次结构化调用也走 `bind_tools(tool_choice="auto")`（审查循环本来就没强制过），回参自己用 Pydantic 校验 —— 见 `core/reflection/reflect.py` |

**不使用已废弃的 API**（尤其 `langchain-classic` 包）；其余尽量用框架，别自己重造。

**工具定义的形态约定**：五个工具一律 `@tool(parse_docstring=True, description="…")`。
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

max_cost = settings.review.max_group_cost
```

优先级 `环境变量 > default.toml`；唯一走环境变量的是 `DEEPSEEK_API_KEY`（密钥不落盘）。
读取见 `config/settings.py`，它只做三件事：读 toml、按键记进 dataclass、拦下配错的键。

**什么进文件、什么不进**，判据是「改它会不会改变我对 LLM / git / 框架的契约」：

| | 例子 |
|---|---|
| **进**（旋钮，配错了看得见） | 窗口、思考档位、压缩阈值、**钱/时间的预算与价格表**、工具失败阈值、工具上限、分组的 token 预算、静态分析超时、日志级别 |
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

**外部二进制只有 golangci-lint 一个**（`go install
github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest`，实测装在
`~/go/bin`）。它是**可选依赖**：没装时静态分析跳过、审查照常跑，所以单测里
一条都不依赖它 —— 喂真实 JSON 载荷给解析层，用 monkeypatch 造失败形态。
真跑一次工具的场景留给 `tests/manual/` 下的脚本。

## 设计文档

在 `/Users/caojunfeng/Projects/golang/open-code-review/acr-min/`，**一个阶段一个目录**：

| 目录 | 设计（冻结） | 反馈 / 取舍（活） |
|---|---|---|
| `00-最小闭环/` | 1~5 | 6（实现偏离，五类分）、7（设计复盘） |
| `01-上下文压缩/` | 9 | 10（实现反馈）、11（设计取舍） |
| `02-评论定位和反思/` | 12 | 13（定位）、14（反思） |
| `03-添加工具和调用图/` | 15 | 16（跨组工具实现反馈）、17（**框架缺口清单**，台账） |
| `04-添加静态审查/` | 18 | 19（静态分析实现反馈，含 golangci-lint v1→v2 的坑） |
| `05-预算与止损/` | —— | 20（**审查约束清单**：五类闸的作用与 trade-off，含摘掉 max_rounds 的理由） |

- **设计文档冻结、永不回头改**；实现与它不符时**记进反馈文档**
  （唯一例外：反馈文档里写明「这条必须回改设计文档」的，如 13 号 §六 E1）
- 反馈文档按「文档缺口 / 文档写错 / 前提失效 / 工程取舍 / 文档不一致」五类分，
  并附**证据**（门槛数字 + 变异测试表 + 真实数据对账）
- **`acr-min` 不在 git 里**，别试图提交

- **实现发生变更后要同步反馈文档**：改完代码就顺手记一笔，别攒着
