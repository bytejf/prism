"""工具执行的运行时上下文。

有状态、有行为，与 models/tool.py 的纯数据模型区分开。
"""

import logging
from dataclasses import dataclass, field

import pygit2

from prism.core.diff.loader import read_blob, resolve_commit
from prism.models.tool import CommentInput

logger = logging.getLogger(__name__)

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
        head_rev: 被审查的版本（分支名 / SHA）。工具必须读这一版而不是工作区，
            否则「审别人的 PR、自己 checkout 在 main」时会读到改之前的代码。
            为空表示只能读工作区。
    """

    hunk_status: dict[str, HunkStatus] = field(default_factory=dict)
    all_comments: list[CommentInput] = field(default_factory=list)
    project_root: str = ""
    end_done_count: int = 0
    head_rev: str = ""
    # 懒加载的 (仓库, 被审查版本的 commit)，只解析一次
    _head_cache: "tuple[pygit2.Repository, pygit2.Commit] | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def read_source(self, path: str) -> str | None:
        """读取「被审查版本」中某个文件的内容。

        Args:
            path: 仓库内相对路径

        Returns:
            文件文本；未设置 head_rev 或该版本中不存在该文件时返回 None，
            由调用方决定是否退化为读工作区
        """
        head = self.head_source()
        if head is None:
            return None
        repo, commit = head
        return read_blob(repo, commit, path)

    def head_source(self) -> "tuple[pygit2.Repository, pygit2.Commit] | None":
        """懒加载被审查版本的 (仓库, commit)。

        project_root 不是仓库或版本无法解析时返回 None 并告警，
        不抛异常 —— 工具应优雅退化而不是让整轮审查崩掉。

        Returns:
            (仓库, commit)；不可用时返回 None
        """
        if self._head_cache is not None:
            return self._head_cache
        if not self.head_rev or not self.project_root:
            return None

        try:
            repo = pygit2.Repository(str(self.project_root))
            commit = resolve_commit(repo, self.head_rev)
        except (pygit2.GitError, ValueError) as exc:
            logger.warning("无法解析被审查版本 %s，退化为读工作区：%s", self.head_rev, exc)
            return None

        self._head_cache = (repo, commit)
        return self._head_cache

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
