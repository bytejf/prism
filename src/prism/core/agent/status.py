"""`<review_status>` 渲染。

每个 `report_hunks` 之后追加一条 user 消息，另外预算跨过里程碑时也追加一条，
让 LLM 看到**覆盖缺口**与**预算进度**。

hunk 进度的**唯一权威来源**就是本模块渲染的这条消息：它是代码确定性生成的，
不会错。压缩摘要里的 `Completed Hunks` 段只是 LLM 写的历史快照，两者冲突时
以本消息为准（见 `01-上下文压缩/9-上下文压缩.md` §3.2）。

**为什么内容是这个形状**：实测 22 个 PR 的全量跑里，模型的典型形态是「读 41 轮、
最后一次性报 49 个 hunk、写 1 条评论」——它自始至终**没看到过覆盖缺口**
（旧的快照只在「上一轮调过 report_hunks」时才注入，而它第 41 轮才第一次报）。
所以这条消息的第一句必须是「已给结论 x/N」而不是「x/N 完成」：
前者是**还缺多少**，后者读起来像是进程状态。
"""

from prism.core.agent.state import (
    STATUS_DONE,
    HunkStatus,
)

# 包住进度快照的标签
REVIEW_STATUS_TAG_OPEN = "<review_status>"
REVIEW_STATUS_TAG_CLOSE = "</review_status>"

# 未报 hunk 的展示上限：全列出来在大组上会长到几百 token，而**数得清**比
# **列得全**重要 —— 差多少个看数字，具体是哪些看编号
MAX_PENDING_LISTED = 20

# 收尾要求：它是**唯一一条硬要求**（`end_done` 的前置条件），必须出现在每次快照里
STATUS_FOOTER = "每个 hunk 都要报过一次（没问题就报空评论列表），全部报完之后才能调 end_done。"


def build_review_status(
    hunk_status: dict[str, HunkStatus], *, used_ratio: float | None = None
) -> str:
    """渲染 hunk 审查进度快照。

    Args:
        hunk_status: hunk 状态表
        used_ratio: 本组预算已用比例（0~1），由 `consumed_ratio` 算出。
            **传 None 表示调用方拿不到这个数**（压缩器那条路径没有用量），
            那时不写预算 —— 写一个 0% 是错的，比不写更坏

    Returns:
        `<review_status>...</review_status>` 文本
    """
    ordered = sorted(hunk_status)
    done = [hunk_id for hunk_id in ordered if hunk_status[hunk_id].status == STATUS_DONE]
    pending = [hunk_id for hunk_id in ordered if hunk_status[hunk_id].status != STATUS_DONE]

    headline = f"### 已给结论 {len(done)}/{len(ordered)}"
    if used_ratio is not None:
        headline += f" · 本组预算已用 {used_ratio:.0%}"

    lines = [
        REVIEW_STATUS_TAG_OPEN,
        headline,
        "",
        f"已报过（{len(done)}）：{', '.join(done) if done else '无'}",
        "",
        f"**还没报告（{len(pending)}）**：{_pending_text(pending)}",
        "",
        STATUS_FOOTER,
        REVIEW_STATUS_TAG_CLOSE,
    ]
    return "\n".join(lines)


def _pending_text(pending: list[str]) -> str:
    """把未报的 hunk 编号渲染成一行（过长时截断）。

    Args:
        pending: 未报的 hunk 编号

    Returns:
        编号清单；超过上限时带「等 N 个」
    """
    if not pending:
        return "无"
    shown = pending[:MAX_PENDING_LISTED]
    text = ", ".join(shown)
    if len(pending) > len(shown):
        text += f" …（等 {len(pending)} 个）"
    return text
