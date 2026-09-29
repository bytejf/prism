"""工具执行的运行时上下文。

有状态、有行为，与 models/tool.py 的纯数据模型区分开。
"""

from dataclasses import dataclass, field

from prism.models.tool import CommentInput

# hunk 审查状态的取值
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"


@dataclass
class HunkStatus:
    """hunk 的审查状态（运行时可变状态）。

    Attributes:
        hunk_id: hunk 编号
        status: 状态（pending / in_progress / done）
        comments: 已收集的评论
    """

    hunk_id: str
    status: str  # pending / in_progress / done
    comments: list[CommentInput] = field(default_factory=list)


@dataclass
class ToolContext:
    """工具执行上下文。

    维护跨工具调用的共享状态：hunk 状态表、已收集评论等。

    Attributes:
        hunk_status: hunk 状态表（hunk_id → HunkStatus）
        all_comments: 所有已收集的评论
        project_root: 项目根目录
        end_done_count: end_done 被调用的次数（用于拒绝机制）
    """

    hunk_status: dict[str, HunkStatus] = field(default_factory=dict)
    all_comments: list[CommentInput] = field(default_factory=list)
    project_root: str = ""
    end_done_count: int = 0

    def pending_ids(self) -> list[str]:
        """未审查完毕的 hunk 编号。

        Returns:
            编号列表，按字典序排序
        """
        return sorted(
            hunk_id for hunk_id, item in self.hunk_status.items() if item.status != STATUS_DONE
        )

    def done_count(self) -> int:
        """已审查完毕的 hunk 数。

        Returns:
            已标记 done 的 hunk 数量
        """
        return sum(1 for item in self.hunk_status.values() if item.status == STATUS_DONE)

    def init_hunks(self, hunk_ids: list[str]) -> None:
        """初始化 hunk 状态表，全部置为 pending。

        已存在的编号保留原有状态，便于复用同一个 context 重跑。

        Args:
            hunk_ids: 本轮要审查的 hunk 编号
        """
        for hunk_id in hunk_ids:
            self.hunk_status.setdefault(hunk_id, HunkStatus(hunk_id=hunk_id, status=STATUS_PENDING))
