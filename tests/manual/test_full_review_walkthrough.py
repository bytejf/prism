"""完整审查流程的走查（**真调 DeepSeek**）—— 给人挂断点用的启动器。

**这不是断言套件，是个启动器。** 它只做一件事：把跟 `prism review` 完全相同
的那条链路在**当前进程里**跑一遍 —— 子进程调 CLI 的话，你在源码里打的断点
一辈子不会命中。

**为什么走查仓库用 `prism-min-test`**：2 个 Go 文件 / 6 个 hunk / **单个审查单元**，
跑完约 5~8 轮。小到能一眼看懂每一步的中间产物，又完整覆盖
`diff → 过滤 → hunk 映射 → 静态分析 → 分组 → LLM 审查 → 定位 → 反思 → 渲染`
九段 —— 大仓库跑一次要几分钟几百个 hunk，反而看不出东西。

跑法：

    uv run pytest tests/manual/test_full_review_walkthrough.py -s

    # 想看逐轮轨迹（每轮的提醒、闸、工具调用）
    PRISM_LOG_LEVEL=DEBUG uv run pytest tests/manual/test_full_review_walkthrough.py -s

    # 想看 OTel trace（跑完用 `jq` 或编辑器看那份 JSONL）
    PRISM_TRACE_FILE=/tmp/walkthrough.trace.jsonl uv run pytest \
        tests/manual/test_full_review_walkthrough.py -s

    # 换仓库 / 换版本（默认 prism-min-test 的 main → HEAD）
    PRISM_WALKTHROUGH_REPO=~/Projects/python/prism-tests \
    PRISM_WALKTHROUGH_BASE=main PRISM_WALKTHROUGH_HEAD=HEAD \
        uv run pytest tests/manual/test_full_review_walkthrough.py -s

**花多少钱**：约 0.1~0.3 元（一个审查单元 + 一次反思调用）。
`prism-min-test` 的干跑是 6 个 hunk / 1 组 / 约 1839 token，代价很低。

**`-s` 不能省**：三个阶段的进度走 `typer.echo`（stdout），日志走 `PrismHandler`
（stderr），而 pytest 的捕获是 **fd 级**的 —— 不加 `-s` 两样都看不见。

看什么（终端上按顺序出现的九段）：

    加载 diff…            2 个文件，6 个 hunk
    过滤文件…             可审查 2 个文件，跳过 0 个
    （映射、静态分析）      1.2 秒，可用发现 0 条
    （分组）              1 个包，6 个 hunk → 1 组
    开始审查 group-001…   每轮的 ai/tool 行、提醒、收尾闸
    结束审查 group-001…   exit_reason / 轮次 / 评论数 / 未审 / 耗时 / 用量
    （定位、反思、排序、渲染）最后是渲染出来的评论

**断点打在哪**（想弄清哪一段就在哪一段断，全都在主调用的同一条调用栈上）：

| 想看什么 | 断在哪 |
|---|---|
| 每一步的中间产物 | `cli/main.py` 的 `_review`（九段都在它函数体里，逐行往下走） |
| **真正发出去的提示词** | `core/agent/agent.py` 的 `initial_state` —— 它就是用户消息 |
| 装箱为什么这么切 | `core/grouping/packages.py` 的 `_pack`（`group_cost` 与预算比较那一行） |
| 每轮的判定与注入 | `core/agent/middleware.py` 的 `ReviewProgress.before_model` |
| 模型每一次报告 | `core/tools/report_hunks.py` 的 `report_hunks`（`changed` 是写回的增量） |
| 评论怎么挂到行号上 | `core/output/locate.py` 的 `locate_group` |
| 反思删了哪几条 | `core/reflection/reflect.py` 的 `reflect` |

**想先零成本对账再花钱**：`uv run python tests/manual/audit_pipeline.py
~/Projects/python/prism-min-test` —— 它跑前半段并核对四条不变量（hunk 不重不漏 /
预算账 == 真实渲染 / 提示词自洽 / 定位同源），秒级、零 token。
"""

import os
from pathlib import Path

from prism.cli.main import _review
from prism.models.comment import ReviewStats
from prism.observability.logging import setup_logging
from prism.observability.tracing import setup_tracing

# 走查目标与版本。都用环境变量兜一层，方便换仓库时不用改文件
REPO = Path(os.getenv("PRISM_WALKTHROUGH_REPO", "~/Projects/python/prism-min-test")).expanduser()
BASE = os.getenv("PRISM_WALKTHROUGH_BASE", "main")
HEAD = os.getenv("PRISM_WALKTHROUGH_HEAD", "HEAD")


def test_full_review_walkthrough() -> None:
    """把完整的 `prism review` 在主进程里跑一遍，然后打印统计。

    入口刻意用 `cli.main._review` 而不是 `prism review` 子进程：**断点要能命中**。
    它带下划线是「模块私有」，但对这个走查脚本来说，走私有入口恰恰是目的 ——
    我们要看的就是真实那条链，而不是照着它再写一份（写第二份就会漂移，
    而漂移的走查比没有走查更坏）。
    """
    assert (REPO / ".git").exists(), f"走查仓库不存在或不是 git 仓库：{REPO}"

    # 这两句平时由 CLI 的 `main` 回调做（见 cli/main.py 的 app.callback）。
    # 直接调 `_review` 就绕过了它 —— 漏掉的话日志一条都不打，很容易误以为
    # 「日志功能坏了」。`setup_tracing` 没设环境变量时是 no-op
    setup_logging()
    setup_tracing()

    stats = _review(BASE, HEAD, REPO)

    _print_stats(stats)


def _print_stats(stats: ReviewStats) -> None:
    """把统计打出来。

    **缺口三项要盯着看**：`skipped_groups` / `pending_hunks` / `failed_groups`
    与「审完了、没问题」在输出上长得几乎一样，只有这三个数能分开它们
    （见 `ReviewStats` 的 docstring）。

    Args:
        stats: `_review` 的返回值
    """
    print("\n=== 走查结束 ===")
    print(f"  仓库      {REPO}（{stats.base_branch} → {stats.head_sha[:10]}）")
    print(f"  规模      {stats.files_count} 个文件，{stats.comments_count} 条评论")
    print(f"  耗时      {stats.duration_seconds:.1f} 秒")
    print(
        f"  缺口      未跑的组 {stats.skipped_groups}，"
        f"未审的 hunk {stats.pending_hunks}，异常退出的组 {stats.failed_groups}"
    )
    print("\n  提示：上面正文里的评论才是产出；缺口三个数全 0 才说明这一轮真的审完了")
