"""最小闭环的**走查用例**：在真实项目上跑一轮完整审查，供 IDE 断点调试。

与 `test_smoke.py` 的区别：

- `test_smoke.py` 起子进程跑 CLI，断言「命令跑通」，适合当验收
- 本用例在**进程内**把链路的每一段摊开调用 —— 断点能落在 middleware、
  工具、状态更新上，跟着数据一路走

被审查的项目是 `~/Projects/python/prism-tests/prism-min-test`（可用环境变量
`PRISM_MIN_TEST_REPO` 覆盖）：三个 Go 文件（`store.go` / `format.go` / `lint.go`，
同一个包）、15 个 hunk、**一个审查单元**，一次大约 4 轮 ReAct。
其中 `lint.go` 是专门让静态分析有东西可报的（5 条发现 / 5 个 linter / 三档优先级）。

那个项目是独立于本仓库的 git 仓库（`main` 是干净版，`feature` 是改动版），
不在本仓库里 —— 主仓库跟踪不了嵌套仓库。

跑法（真实调用 DeepSeek，会花钱）：

    PRISM_LOG_LEVEL=INFO uv run pytest tests/e2e/test_min_walkthrough.py -s

想只看审查前流程、不花钱：在 `[4/4]` 那行打个断点，跑到就停。

## 日志怎么看：两个开关都要给

    环境变量 PRISM_LOG_LEVEL=INFO     ← 开日志（conftest 夹具看到它才挂 handler）
    命令行   -s                       ← 关 pytest 捕获（否则日志被吞）

少任何一个都**看不到**，而且都是静默的。conftest 的
`pytest_collection_finish` 会在收集阶段自诊断这两种错配，直接告诉你差哪一步。

两个都被需要的原因不同：

- 不给 `PRISM_LOG_LEVEL`：pytest 的 logging 插件会把**根 logger** 的 handler
  换成自己的捕获 handler，我们装的那个被摘掉
- 不给 `-s`：pytest 的捕获是 **fd 级**的（`dup2` 重定向 fd 2），连
  `sys.__stderr__` 都一起吞 —— 绕不过去，只能关掉捕获

PyCharm：Run/Debug Configurations → 选中该用例的配置 → **Environment variables**
填 `PRISM_LOG_LEVEL=INFO`，**Additional Arguments** 加上 `-s`。

## 别用 `--log-cli-level`

它走的是 **pytest 自己的格式**（`级别 logger名:文件:行号`），和项目约定的
`时间 级别 [短路径:行号]` 完全是两回事 —— 而且它不需要 `-s`，所以很容易
「看起来有日志」但格式全错。conftest 检测到它会直接提示。

## 一个顺手的旋钮

**`COLUMNS`**：rich 在非终端环境下按 80 列折行（PyCharm 的控制台就被判成非终端）。
嫌折得难看就在同一处设 `COLUMNS=120`。
"""

import os
from pathlib import Path

import pytest

from prism.config.settings import settings
from prism.core.agent.agent import run_review_agent
from prism.core.agent.state import EXIT_END_DONE, EXIT_END_DONE_WITH_GAPS
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.models.llm import API_KEY_ENV

# 默认待审查项目；换个项目就设 PRISM_MIN_TEST_REPO
DEFAULT_REPO = Path.home() / "Projects" / "python" / "prism-min-test"
REPO_ENV = "PRISM_MIN_TEST_REPO"

BASE_REV = "main"
HEAD_REV = "feature"


@pytest.fixture
def demo_repo() -> Path:
    """定位待审查项目。

    Returns:
        仓库路径

    Raises:
        pytest.skip.Exception: 项目不存在
    """
    repo = Path(os.getenv(REPO_ENV, str(DEFAULT_REPO))).expanduser()
    if not (repo / ".git").exists():
        pytest.skip(f"待审查项目不存在：{repo}（可用 {REPO_ENV} 指定）")
    return repo


@pytest.fixture
def llm() -> LLMClient:
    """构造真实 LLM 客户端。

    Returns:
        LLMClient 实例

    Raises:
        pytest.skip.Exception: 未设置 API key
    """
    if not os.getenv(API_KEY_ENV):
        pytest.skip(f"未设置 {API_KEY_ENV}")
    return LLMClient(settings.llm)


def test_min_loop_walkthrough(demo_repo: Path, llm: LLMClient) -> None:
    """在 prism-min-test 上跑一组完整审查。

    阶段与 `prism review` 完全一致，只是把每一步摊开、逐段打印，
    方便在 IDE 里边看边停。

    断言只锁「闭环跑通了」（正常收尾），**不锁评论内容** ——
    评论质量本身不稳定（没有定位 / 去重 / 反思），那是阶段 3 的事。
    """
    print(f"\n[1/4] 加载 diff：{demo_repo}  {BASE_REV}..{HEAD_REV}")
    diffs = load_diffs(demo_repo, BASE_REV, HEAD_REV)
    print(f"      {len(diffs)} 个文件，{sum(len(d.hunks) for d in diffs)} 个 hunk")

    print("[2/4] 过滤文件")
    reviewable = filter_files(diffs).reviewable
    print(f"      可审查 {len(reviewable)} 个文件")

    print("[3/4] hunk → 符号 → 分组")
    groups = group_by_package(map_hunks_to_symbols(demo_repo, BASE_REV, HEAD_REV, reviewable))
    assert len(groups) == 1, "这个项目是照着「一个审查单元」设计的，分组变了说明项目被动过"
    group = groups[0]
    print(
        f"      Group 1：{group.package_path}（{len(group.files)} 文件，{len(group.hunks)} hunk）"
    )

    # BaseChatModel 的协议里没有 model_name / reasoning_effort，用 getattr 取值只为打印
    model_desc = (
        f"{getattr(llm.model, 'model_name', type(llm.model).__name__)}"
        f"（effort={getattr(llm.model, 'reasoning_effort', '?')}）"
    )
    print(f"[4/4] 跑 Main 循环（真实调用 DeepSeek，{model_desc}）")
    result = run_review_agent(group, llm.model, str(demo_repo), HEAD_REV)

    print(f"\n退出原因  ：{result.exit_reason}")
    print(f"模型轮次  ：{result.round_count}")
    print(f"未审 hunk ：{result.pending_hunk_ids or '无'}")
    print(f"评论      ：{len(result.comments)} 条")
    for hunk_id, comments in result.hunk_comments.items():
        for comment in comments:
            print(f"  [{hunk_id}] {comment.severity} · {comment.category}｜{comment.content[:70]}")

    assert result.exit_reason in {EXIT_END_DONE, EXIT_END_DONE_WITH_GAPS}, (
        f"闭环没有正常收尾：{result.exit_reason}（轮次 {result.round_count}）"
    )
