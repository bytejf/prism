"""终端渲染器。

把排序后的评论列表渲染成人类可读的终端文本。
V1 简化：不渲染 suggestion_code 的 diff、不展示 rule_ref、不展示定位元数据。
"""

import logging
from collections import Counter

from prism.core.output.sort import SEVERITY_RANK
from prism.models.comment import Comment, ReviewStats

logger = logging.getLogger(__name__)

# 头部标题
HEADER_TITLE = "=== Prism Code Review ==="

# 尾部标题
FOOTER_TITLE = "=== 统计 ==="

# 评论分隔符，如 ─── internal/agent/agent.go ───
SEPARATOR_TEMPLATE = "─── {path} ───"

# 徽章，如 [concurrency · high]
BADGE_TEMPLATE = "[{category} · {severity}]"

# 尾部统计的展示顺序：high → medium → low
SEVERITY_ORDER: tuple[str, ...] = tuple(sorted(SEVERITY_RANK, key=lambda name: SEVERITY_RANK[name]))


class TerminalRenderer:
    """终端渲染器。

    将评论列表渲染为人类可读的终端文本。
    V1 简化：不渲染 suggestion_code 的 diff，不展示 rule_ref。
    """

    def render(self, comments: list[Comment], stats: ReviewStats) -> str:
        """渲染评论列表为终端文本。

        Args:
            comments: 排序后的评论列表
            stats: 审查统计信息

        Returns:
            格式化后的终端文本
        """
        logger.debug("渲染开始：%d 条评论", len(comments))
        blocks = [self._render_header(stats)]
        blocks.extend(self._render_comment(comment) for comment in comments)
        blocks.append(self._render_footer(comments))

        text = "\n\n".join(blocks)
        logger.info("渲染完成：输出 %d 字符", len(text))
        return text

    def _render_header(self, stats: ReviewStats) -> str:
        """渲染头部：审查范围 + 文件数 + 评论数 + 耗时。

        Args:
            stats: 审查统计信息

        Returns:
            头部文本
        """
        return "\n".join(
            (
                HEADER_TITLE,
                f"审查范围：merge-base({stats.base_branch}, {stats.head_sha})..{stats.head_sha}",
                f"文件数：{stats.files_count} | 评论数：{stats.comments_count}"
                f" | 耗时：{stats.duration_seconds:.1f}s",
            )
        )

    def _render_comment(self, comment: Comment) -> str:
        """渲染单条评论。

        格式：
        ─── path ───
        [category · severity]
        评论正文

        Args:
            comment: 单条评论

        Returns:
            渲染后的评论文本
        """
        return "\n".join(
            (
                SEPARATOR_TEMPLATE.format(path=comment.path),
                BADGE_TEMPLATE.format(category=comment.category, severity=comment.severity),
                comment.content,
            )
        )

    def _render_footer(self, comments: list[Comment]) -> str:
        """渲染尾部统计。

        非 high/medium/low 的 severity 也计入，排在三个已知值之后，
        避免出现「评论数 5 但统计只有 3」的对不上。

        Args:
            comments: 评论列表

        Returns:
            尾部统计文本
        """
        counts = Counter(comment.severity for comment in comments)
        names = [*SEVERITY_ORDER, *(name for name in sorted(counts) if name not in SEVERITY_ORDER)]
        summary = " | ".join(f"{name}: {counts.get(name, 0)}" for name in names)
        return "\n".join((FOOTER_TITLE, summary))
