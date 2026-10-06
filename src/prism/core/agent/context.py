"""审查的静态上下文。

走 LangGraph 的 `context_schema` / `invoke(..., context=...)`：**一次审查内不变**，
不进 state、不进消息历史、不占 checkpoint。工具与 middleware 都通过
`runtime.context` 读它（已实测两边都能拿到）。

旧实现把静态量（project_root / head_rev）和可变量（hunk 状态、各类计数）混在
同一个可变对象 `ToolContext` 里塞进 state —— 那样 context 类型只能填 `None`
（`ToolRuntime[None, MainState]`），是「该填没填」而不是「没得填」。
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import pygit2

from prism.core.diff.loader import read_blob, resolve_commit
from prism.models.diff import Diff

logger = logging.getLogger(__name__)


@dataclass
class ReviewContext:
    """审查的静态上下文。

    Attributes:
        project_root: 项目根目录
        head_rev: 被审查的版本（分支名 / SHA）。工具必须读这一版而不是工作区，
            否则「审别人的 PR、自己 checkout 在 main」时会读到改之前的代码。
            为空表示只能读工作区。
        diffs: `路径 → Diff` 索引（只含可审查文件，含改名文件的旧路径别名）。
            `file_read_diff` 靠它回答「别的文件改了什么」—— 分组把跨包改动切成了
            互不可见的审查单元，这是唯一的补足手段。由 `build_diff_index` 建。
    """

    project_root: str = ""
    head_rev: str = ""
    diffs: Mapping[str, Diff] = field(default_factory=dict)
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
