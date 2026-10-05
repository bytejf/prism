"""审查后处理的单步走查：定位 → 反思 → 排序 → 渲染。

**这个文件是给 IDE 断点用的**，每个阶段的结果都留在具名变量里，在任意一行
下断点就能看清那一步的输入与输出。审查前 / 审查中那两个阶段见同目录的
`test_debug_prism_tests.py`。

    located   → list[Comment]      定位结果（补上行号与侧）
    outcome   → ReflectionOutcome  反思结果（kept / removed / misses / failure）
    ordered   → list[Comment]      排序后
    text      → str                终端渲染结果

**LLM 是真的**（审查一次 + 每组反思一次），所以：

- 一次运行约 15~25 秒、一组 token，比走查脚本贵，别放进循环里跑
- 审查结果由**模块级 fixture 缓存**，同一轮里几个用例复用同一次审查
- 断点会卡在网络上，step into `_ask` 之前想清楚

看日志（本文件**不自己打** print 或 logger，全部走既有日志）：

    PRISM_LOG_LEVEL=INFO uv run pytest tests/manual/test_debug_post_review.py -s

    反思 · 请求 · system / user   ← 送给复核者的完整提示词（INFO）
    反思 · 响应 · 耗时 Nms         ← 模型返回了什么（INFO，失败时这条尤其重要）
    反思 · 完成 · 删除 N / 保留 M  ← 删了谁、依据是什么（INFO）
    定位完成                       ← 各层命中数、未定位数、成功率（INFO）
    轮次 / 工具调用 / token         ← llm_trace 中间件打的（INFO）

用别的仓库：

    PRISM_MIN_TEST_DIR=~/Projects/foo uv run pytest tests/manual/test_debug_post_review.py -s
"""

import logging
import os
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from prism.config.settings import settings
from prism.core.agent.agent import run_review_agent
from prism.core.agent.state import MainLoopResult
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.core.output.locate import locate_group
from prism.core.output.renderer import TerminalRenderer
from prism.core.output.sort import sort_comments
from prism.core.reflection.reflect import reflect
from prism.models.comment import ReviewStats
from prism.models.grouping import Group

# 走查目标：2 个 Go 文件、6 个 hunk、**单审查单元**、约 5 轮收尾 —— 最小的那个
# 仓库，一次审查的 token 成本最低。`NOTES.md` 列了埋的 6 处问题
REPO = Path(os.getenv("PRISM_MIN_TEST_DIR", Path.home() / "Projects/python/prism-min-test"))
BASE = "main"
HEAD = "HEAD"

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not (REPO / ".git").is_dir() or not settings.llm.api_key,
    reason=f"需要走查仓库 {REPO}（设 PRISM_MIN_TEST_DIR）与 DEEPSEEK_API_KEY",
)


@pytest.fixture(scope="module")
def model() -> BaseChatModel:
    """真实模型。模块级 —— 构造一次，几个用例共用。

    Returns:
        LangChain ChatModel
    """
    return LLMClient(settings.llm).model


@pytest.fixture(scope="module")
def reviewed(model: BaseChatModel) -> tuple[Group, MainLoopResult]:
    """跑一次真实审查，本模块内复用 —— **只花一次审查的钱**。

    断点看这两步：

    - `filter_files(...)` 之后：可审查文件清单
    - `group_by_package(...)` 之后：`groups[0].hunks` 是本组全部 hunk，
      每条带 `hunk_id` / `file_path` / `part.content`（diff 原文）/ `symbol`
    - `run_review_agent(...)` 之后：`result.hunk_comments` 是
      `{hunk_id: [CommentInput]}` —— **评论还没有行号、没有文件路径**，
      这一步只有 severity / category / content / existing_code

    Args:
        model: 真实模型

    Returns:
        (审查单元, 循环产出)
    """
    reviewable = filter_files(load_diffs(REPO, BASE, HEAD)).reviewable
    groups = group_by_package(map_hunks_to_symbols(REPO, BASE, HEAD, reviewable))
    assert groups, f"{REPO} 在这个 range 里没有可审查的 hunk"

    group = groups[0]
    return group, run_review_agent(group, model, str(REPO), HEAD)


def test_debug_locate(reviewed: tuple[Group, MainLoopResult]) -> None:
    """定位：把模型报的评论挂到行号上（零 LLM）。

    断点建议：`locate_group` 里 → `locate` 里 → `_side_lines`（行号怎么数的）。

    Args:
        reviewed: 审查单元与循环产出
    """
    group, result = reviewed
    # 这行是**分界线**，用来分辨「测试体没跑」和「日志被吞了」：
    #   看得到它 → 测试体真的跑了（没被 skip、PRISM_LOG_LEVEL 也进了进程），
    #              那要是还缺日志，就是上面 fixture 阶段被吞（夹具顺序）
    #   看不到它 → 根本没跑到这里，往上看：skip（缺 key）或环境变量没进去
    logger.info(
        "走查 · 定位开始 · %d 个 hunk / %d 条评论",
        len(group.hunks),
        sum(len(items) for items in result.hunk_comments.values()),
    )
    located = locate_group(group, result.hunk_comments)

    # 看 located：每条 Comment 比 CommentInput 多了这几个字段 ——
    #   path            文件路径（**组内命中时是命中的那个 hunk 的路径**）
    #   locate_level    hunk=在自己报的 hunk 里命中 / group=在本组别处 /
    #                   failed=没定位上
    #   side            RIGHT=新增行，行号是**被审查版本**的坐标
    #                   LEFT =删除行，行号是**基准版本**的坐标
    #   start_line/end_line   按 side 解释；failed 时为 0
    #   locate_reason   仅 failed 非空：no_match / multi_match / empty_existing_code
    #
    # 重点看 failed 的那几条：它们**没有**被排除，照样进下一步反思，
    # 而且提示词里专门给 failed:no_match 标了「重点看 Ground A」
    assert len(located) == sum(len(items) for items in result.hunk_comments.values())


def test_debug_reflect(reviewed: tuple[Group, MainLoopResult], model: BaseChatModel) -> None:
    """反思：一次 LLM 调用做证伪，只删不增。

    断点建议：`reflect` 里 → `_render_comment`（评论怎么进提示词，看 `locate` /
    `side` 属性）→ `_ask`（**真实网络调用**）→ `_parse`（回参校验）。

    Args:
        reviewed: 审查单元与循环产出
        model: 真实模型
    """
    group, result = reviewed
    located = locate_group(group, result.hunk_comments)
    outcome = reflect(model, group, located)

    # 看 outcome 的四个字段 ——
    #   kept      留下的评论（排序 + 渲染只用它）
    #   removed   被删掉的，每条带 comment_id / path / ground / reasoning，
    #             **这是删评论唯一的留档**，日志里也有全文
    #   misses    疑似漏报（hunk_id / path / description），单独一段渲染
    #             —— 它不是评论，不进 kept
    #   failure   非空表示 fail-open：本次没删任何东西，kept 是原样全部
    assert len(outcome.kept) + len(outcome.removed) == len(located)


def test_debug_render(reviewed: tuple[Group, MainLoopResult], model: BaseChatModel) -> None:
    """排序 + 渲染：产出发给用户的那段文本。

    断点建议：`sort_comments` 的 `_sort_key`（排序键：severity → 路径 → 侧 → 行号）。

    Args:
        reviewed: 审查单元与循环产出
        model: 真实模型
    """
    group, result = reviewed
    located = locate_group(group, result.hunk_comments)
    outcome = reflect(model, group, located)
    ordered = sort_comments(outcome.kept)

    stats = ReviewStats(
        base_branch=BASE,
        head_sha=HEAD,
        files_count=len(group.files),
        comments_count=len(ordered),
        duration_seconds=0.0,
    )
    text = TerminalRenderer().render(ordered, stats, outcome.misses)

    # 看 text 的三段结构 ——
    #   ─── path:行号 ───          已定位；删除侧的会标（已删除）
    #   === 未定位评论（N 条）===    locate_level=failed 的那些，每条带原因
    #   === 疑似漏报（N 条）===      反思的 misses，没有就不出现这一段
    assert "=== 统计 ===" in text
