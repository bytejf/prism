# Prism

一个**确定性优先**的 Go 代码审查 Agent。

传统 linter 只能发现模式化问题，纯 LLM 审查又贵、幻觉多、不可复现。
Prism 用「确定性优先 + LLM 补充」的混合架构：能用规则判定的绝不调 LLM。

## 安装

```bash
uv sync
```

## 快速开始

```bash
export DEEPSEEK_API_KEY=sk-...

# 先干跑一遍看清范围（不花钱、不调用 LLM）
prism group --path ~/Projects/some-repo --base main --head origin/some-branch

# 确认后再真审
prism review --path ~/Projects/some-repo --base main --head origin/some-branch
```

`--path` 是**必填**的，指向被审查的 git 仓库；不给会直接报错。
刻意不做「默认当前目录」：审查会花 token、会把代码发给外部 API，
少敲一个参数就审错仓库的代价比多敲一个大。

`review` 的输出：

```
=== Prism Code Review ===
审查范围：merge-base(main, feature)..feature
文件数：3 | 评论数：5

─── db.go ───
[安全-SQL注入 · high]
将参数化查询改为字符串拼接，id 直接拼入 SQL，存在严重 SQL 注入漏洞。……

=== 统计 ===
high: 4 | medium: 1 | low: 0
```

## 命令

全部命令都用 `--path` 指定被审查的 git 仓库，以它的 `merge-base(base, head)`
为基准比较，只统计 PR 实际引入的改动。

### 审查前流程（确定性，不调用 LLM）

四个命令依次是流水线的下一个环节，可以逐个跑、逐个核对：

| 命令 | 功能 |
|------|------|
| `prism diff` | 加载并解析 diff，打印文件数 / hunk 数 / 增删行数 |
| `prism filter` | 按规则过滤文件，打印可审查 / 跳过清单（含跳过原因） |
| `prism hunk-map` | 把每个 hunk 归属到它重叠的函数 / 方法 |
| `prism group` | 按包路径分组，输出即为送进 LLM 的「审查单元」 |

### 完整审查（调用 LLM）

| 命令 | 功能 |
|------|------|
| `prism review` | diff → 过滤 → hunk 映射 → 分组 → LLM 审查 → 定位 → 反思 → 排序 → 终端渲染 |

所有命令都要 `--path` 指定被审查仓库，另有 `--base` / `--head`（默认
`main` / `HEAD`）。完整命令清单与参数见 `prism --help`。

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | **必填**（仅 `review` 需要）。缺失时 `review` 以退出码 1 明确报错 |
| `PRISM_LOG_LEVEL` | 日志级别，覆盖配置文件里的 `[log].level`。只作用于 `prism` 命名空间，不影响第三方库。想看完整轨迹设 `DEBUG`，想只看结果设 `WARNING` |

## 配置文件

**默认值只有一份**，在 `src/prism/config/default.toml`：模型与窗口、压缩阈值、
审查轮次、工具上限、日志级别。改值改那个文件即可，代码里没有第二份。

```toml
[llm]
model = "deepseek-flash"
context_window = 1000000     # 压缩的触发阈值由它算出来
reasoning_effort = "low"     # API 默认是 high（最贵档）

[compression]
reserve_tokens = 20000       # 已用达到 (窗口 - 预留) 就压缩

[review]
min_rounds = 30
rounds_per_hunk = 5

[tools.file_read]
max_lines = 500
```

文件里每个键都带注释说明它控制什么，开头还列了**哪些值故意不在这里**（标签名、
`H-01` 编号格式、退出原因码、提示词全文……）——那些是对 LLM / git / 框架的契约，
写进配置等于把它们变成「可以配错的东西」，而配错的后果是静默失效。

配置读错时**直接报错**并指出是哪个文件、哪一段、哪个键，不静默回落到默认值。

## 设计要点

- **工具读「被审查版本」而不是工作区** —— `--head` 指哪就读哪，
  工作区停在别的分支也能正确审查
- **hunk → 函数映射走 AST** —— tree-sitter 提取函数行号范围后求交，
  不靠 LLM 猜测
- **进度由代码统计** —— LLM 只报告原始结论，完成状态、剩余清单都由代码维护
- **行号靠静态匹配，不靠 LLM 猜** —— 评论挂到具体行上走三层静态回退
  （本 hunk 内 → 本组其它 hunk → 聚合），零 LLM、同一输入同一输出。
  新增行与**被删除的行**都能挂（锚点模型照抄 GitHub 的 `side`：`RIGHT` 数被
  审查版本、`LEFT` 数基准版本，输出里标 `（已删除）`）。
  定位不到**不丢弃**，单独列为「未定位评论」并注明原因（没给锚点 / 匹配不到 /
  歧义），让人知道哪几条需要人工确认
- **LLM 只做证伪，不做验证** —— 每组审查完做一次复核：只找「被 diff 直接推翻」
  的评论并删掉，**不新增、不改写**。默认答案是全部保留（留下一条错评论只多花
  几秒注意力，删掉一条对的评论会让真实发现静默消失）。内存安全 / 并发 /
  声明一致性 / 行为兼容性这几类主题**一律不删**，哪怕复核者确信它错了。
  复核者通过**工具调用**交结论，编号对不上或说不出理由会被当场退回、改完重报；
  复核失败就原样保留全部评论并告警，绝不因为复核出问题而丢评论
- **跨文件改动可查** —— 按包分组会把「改接口 + 改实现 + 改调用方」切成互不可见的
  审查单元，`file_read_diff` 让模型按需取其他文件的 diff，自己判断有没有同步适配。
  刻意**不做调用图分组**：实测调用图的噪音比 73%，而且错误的分组比不分组更糟
- **输出顺序可复现** —— 按 severity + 文件路径排序，同一输入两次运行结果一致

## 开发

```bash
uv run pytest tests/unit/     # 单元测试（不调用 LLM）
uv run pytest tests/e2e/      # 端到端 smoke test（真实调用 LLM）
uv run pytest tests/          # 全部
uv run ruff check . && uv run ruff format --check .
uv run mypy src/ tests/
```

测试用的 Go 项目在 `tests/fixtures/go-sample/`（`base/` 无 bug、`buggy/` 埋了 5 个），
由测试用 `tmp_path` 程序化构造成临时 git 仓库。
`tests/fixtures/go-sample/BUGS.md` 记录了埋入的 bug 清单。
