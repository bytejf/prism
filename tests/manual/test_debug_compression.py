"""压缩流程的单步调试 —— 在**真实 API** 上跑一次「历史被压缩」的全过程。

**这个文件是给 IDE 断点用的，不是断言型测试。** 每一步的产物都留在具名变量里：

    diffs     → list[Diff]           加载 + 解析
    filtered  → FilterResult         可审查 / 跳过
    mapped    → list[HunkWithSymbol] 每个 hunk 的归属符号
    groups    → list[Group]          审查单元（本文件里只有一个）
    group     → Group                本次要审的那一个
    state     → ReviewState          首条 user 消息（指导 + diff）
    final     → ReviewState          循环结束时的状态，**含压缩后的消息列表**
    summaries → list[BaseMessage]    摘要消息（压缩真的发生了才有）

## 为什么要真调 API

压缩的触发依据是 API 返回的 `usage_metadata`、摘要是真模型写出来的 ——
这两样假模型都给不了。`tests/unit/test_compression.py` 覆盖的是**逻辑**
（清理、截断、熔断、重组），这里覆盖的是**真跑一遍会不会出事**。

## 怎么把费用压到最低

- 仓库里**只有一个 Go 文件**（`service.go`，6 个 hunk），一个审查单元
- 窗口与预留取得很小：可用上限只有 3000 token，而真实第 1 轮请求就有约 2500
  （system + 5 个工具的 schema + 指导 + diff），第 2 轮必然超过 → **压缩在第 2
  轮触发**，不用跑几十轮去凑

一次完整跑约 6~10 次模型调用。想再省就把 `RESERVE_TOKENS` 调大。

## 实测数字（2026-10-04，`service.go` 6 个 hunk，跑了五次）

**触发点不是固定的** —— 五次的「上一轮已用」落在 **2466 ~ 4876** 之间。原因是
**模型每轮读不读文件是随机的**：先 file_read 再 report，历史就厚；直接 report，
历史就薄。所以下面这些数字看**量级**，别当常数。

```
压缩 · 触发 · 第 1/2 次
<token>
上一轮已用 3284 · 可用上限 3000 · 窗口 6000 · 固定预留 3000

压缩 · 请求 · 历史 5 条
<system> …（压缩提示词全文）
<user>   …（背景 + 待压缩的历史全文）

压缩 · 响应 · 第 1 次 · 耗时 4195ms
<content> …（7 段摘要全文）

压缩 · 完成 · 第 1/2 次
<size>    历史 261 token → 摘要 453 token
<ratio>   1.74
<context> 压缩后上下文 915 token

（循环继续）
user · 轮次=3                       ← 摘要，编号是新的，所以打出来了
user · 轮次=3 · 本轮新增            ← 权威进度
ai · 轮次=3 · 耗时 1347ms
```

四件事一眼可验：

- **压缩确实在 `before_model` 里发生**（不是工具里）—— 触发日志紧跟在上一轮的工具结果之后
- **压缩后只打新消息、不重打 A 段** —— 摘要与进度是新的消息编号，A 段编号没变
- **第二次压缩把第一份摘要也压进去了** —— 末尾「摘要 1 条」而不是 2 条，
  这就是递归压缩（上一轮的 `<compression_summary>` 也是历史的一部分）
- **收尾那一轮不压缩** —— 退出判定在压缩之前，`end_done` 已接受的那一轮不会再压

**压缩率可能大于 1，这不是 bug**：7 段摘要有个**固定的地板**（约 400~600 token）。
历史只攒了两三百 token 时，压缩反而让上下文**变大**（实测见过 2.48）。
这是「拿一个比自身还小的东西去摘要」，真实场景（几十万压到几千）不会出现。
想看压缩率稳定小于 1，把 `RESERVE_TOKENS` 调大、让历史先攒厚再压。

## 怎么跑

必须**两个都给**，缺一个就一行日志都看不到：

    PRISM_LOG_LEVEL=INFO uv run pytest tests/manual/test_debug_compression.py -s

`-s` 不能省：pytest 的捕获是 **fd 级**的（`dup2` 重定向 fd 2），绕不过去。
想看压缩请求与摘要原文就把级别改成 `DEBUG`（正文很长，几十万字符量级）。
"""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from langchain_core.messages import BaseMessage

from prism.config.settings import settings
from prism.core.agent.agent import build_review_agent, initial_state
from prism.core.agent.context import ReviewContext
from prism.core.agent.state import ReviewState, pending_ids
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.models.llm import API_KEY_ENV
from prism.observability.logging import PRISM_LOGGER, PrismHandler

# 挂在 `prism` 命名空间**下面**，不挂 `__name__`：
# 项目的日志级别与 handler 都只配在 `prism` 这个 logger 上，root 被固定在
# WARNING（否则第三方库的 DEBUG 会淹进来）。挂 `__name__` 的话本文件的
# INFO 日志一条都出不来，而且是**静默**的
logger = logging.getLogger(f"{PRISM_LOGGER}.tests.debug_compression")

# 只取这一个文件：多一个文件就多几轮真实请求，而这里要看的只是压缩流程
# `parents[1]` 而不是 `parent`：夹具在 `tests/fixtures/`，本文件在 `tests/manual/`
# —— 脚本从 `tests/` 搬进 `tests/manual/` 那一次，这条相对路径没跟着改，于是
# 这个手动脚本**静默坏了**（跑起来是 FileNotFoundError，不是断言失败）
FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "go-sample"
SOURCE_FILE = "service.go"

BASE_BRANCH = "main"
HEAD_BRANCH = "feature"

# 可用上限 = 6000 - 3000 = 3000 token。实测第 1 轮 usage 约 2466、第 2 轮约 3321，
# 而 usage 只增不减（每轮把上一轮的全量历史重发一遍），所以第 2 轮必然超限。
# 想验证「不触发时不压缩」就把窗口调到默认的 1_000_000。
CONTEXT_WINDOW = 6_000
RESERVE_TOKENS = 3_000

pytestmark = pytest.mark.skipif(
    not os.getenv(API_KEY_ENV),
    reason=f"未设置 {API_KEY_ENV}，本文件要真实调用 DeepSeek",
)


def enable_prism_logging(level: int = logging.INFO) -> None:
    """把 prism 的日志挂到控制台。

    conftest 里那个同名夹具要**显式设了 `PRISM_LOG_LEVEL`** 才生效 —— 这个文件是
    给人调试用的，「忘了设环境变量所以什么都没有」比「日志多了点」更糟。所以这里
    不看环境变量直接挂。

    仍然走项目的 `PrismHandler`（时间 / 级别 / 短路径那套格式），不用
    `basicConfig` —— 后者会被 pytest 挂在 root 上的 handler 顶掉，变成空操作。

    级别只作用于 `prism` 命名空间，httpx / openai 的 DEBUG 不会被放出来。

    Args:
        level: 日志级别，默认 INFO；想看压缩请求与摘要原文就传 DEBUG
    """
    prism = logging.getLogger(PRISM_LOGGER)
    prism.setLevel(level)
    if not any(isinstance(handler, PrismHandler) for handler in prism.handlers):
        prism.addHandler(PrismHandler())


@pytest.fixture
def one_file_repo(tmp_path: Path, git: Callable[..., None]) -> Path:
    """构造一个**只有一个 Go 文件**的临时 git 仓库。

    用 e2e 那对 fixture（`base/service.go` → `buggy/service.go`）里的一个文件：
    两版之间的改动是真的有问题（`Process` 丢了错误处理、`Run` 的 goroutine
    没有退出路径），所以摘要里会有真内容可写。

    Args:
        tmp_path: pytest 临时目录
        git: conftest 提供的 git 辅助函数

    Returns:
        临时仓库路径
    """
    repo = tmp_path / "go-sample"
    repo.mkdir()
    git(repo, "init", "-q", "-b", BASE_BRANCH)
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@example.com")

    source = FIXTURE_DIR / "base" / SOURCE_FILE
    (repo / SOURCE_FILE).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init: base code")

    git(repo, "checkout", "-q", "-b", HEAD_BRANCH)
    changed = FIXTURE_DIR / "buggy" / SOURCE_FILE
    (repo / SOURCE_FILE).write_text(changed.read_text(encoding="utf-8"), encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feat: introduce bugs")
    return repo


def dump_messages(messages: Sequence[BaseMessage], *, limit: int = 600) -> None:
    """把消息列表按条打出来，供核对压缩后的组装结果。

    压缩后应当是三件事：**A 段（指导 + diff）+ 摘要 + 权威进度**。
    顺序不对、A 段没了、摘要没打标签，都是一眼能看出来的。

    Args:
        messages: 消息列表
        limit: 单条正文最多打多少字符
    """
    logger.info("消息列表共 %d 条", len(messages))
    for index, message in enumerate(messages, start=1):
        body = str(message.content)
        suffix = f"…（共 {len(body)} 字符）" if len(body) > limit else ""
        logger.info(
            "[%d] %s id=%s\n%s%s",
            index,
            type(message).__name__,
            message.id,
            body[:limit],
            suffix,
        )


def test_debug_compression(one_file_repo: Path) -> None:
    """在真实 API 上跑一次，让压缩在第 2 轮触发。

    Args:
        one_file_repo: 只有一个 Go 文件的临时仓库
    """
    enable_prism_logging()

    # ── 审查前四段：diff → 过滤 → 归属 → 分组 ────────────────────────
    diffs = load_diffs(one_file_repo, BASE_BRANCH, HEAD_BRANCH)
    filtered = filter_files(diffs)
    mapped = map_hunks_to_symbols(one_file_repo, BASE_BRANCH, HEAD_BRANCH, filtered.reviewable)
    groups = group_by_package(mapped)

    logger.info(
        "审查前：%d 个文件 → 可审 %d → %d 个 hunk → %d 个审查单元",
        len(diffs),
        len(filtered.reviewable),
        len(mapped),
        len(groups),
    )
    assert len(groups) == 1, "只放了一个文件，应当只有一个审查单元"

    group = groups[0]
    state = initial_state(group)
    logger.info(
        "审查单元 %s：%s，%d 个文件，%d 个 hunk",
        group.group_id,
        group.package_path,
        len(group.files),
        len(group.hunks),
    )
    logger.info("首条消息（指导 + diff）共 %d 字符", len(str(state["messages"][0].content)))

    # ── 真实模型 + 极端压缩配置 ──────────────────────────────────────
    config = replace(
        settings.compression, context_window=CONTEXT_WINDOW, reserve_tokens=RESERVE_TOKENS
    )
    logger.info(
        "压缩配置：窗口 %d，预留 %d → 可用上限 %d token"
        "（第 1 轮的真实 usage 一旦达到它，第 2 轮就压缩）",
        config.context_window,
        config.reserve_tokens,
        config.usable_tokens,
    )

    model = LLMClient(settings.llm).model
    agent = build_review_agent(model, config)

    # ── 循环：断在这里可以 step into 中间件与工具 ─────────────────────
    final = cast(
        ReviewState,
        agent.invoke(
            state,
            context=ReviewContext(project_root=str(one_file_repo), head_rev=HEAD_BRANCH),
            # 调试脚本里把递归上限放小：这个脚本就是要逼出压缩那一轮，跑飞了
            # 反而慢
            config={"recursion_limit": 400},
        ),
    )

    # ── 结果 ────────────────────────────────────────────────────────
    messages = final["messages"]
    summaries = [message for message in messages if "<compression_summary>" in str(message.content)]

    logger.info(
        "循环结束：%d 轮，剩余消息 %d 条，其中摘要 %d 条，未审 hunk %s",
        final["round_count"],
        len(messages),
        len(summaries),
        pending_ids(final["hunk_status"]) or "无",
    )
    dump_messages(messages)

    assert summaries, "可用上限取得这么小，压缩应当已经触发 —— 没触发就调小 CONTEXT_WINDOW"
    assert "<hunk id=" in str(messages[0].content), "A 段（指导 + diff）必须留在最前"
    assert any("<review_status>" in str(message.content) for message in messages), (
        "压缩后要有一个权威进度 —— 它只是**压缩那一刻**的最后一条，"
        "循环继续跑完之后末尾是 end_done 的工具结果"
    )
