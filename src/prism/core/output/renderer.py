"""终端渲染器。

把排序后的评论列表渲染成人类可读的终端文本。

未定位的评论**单独成段**放在末尾：它们的位置是「不知道」，混在正常评论里
会让人以为行号已经核实过。各条带上失败原因，用户一眼能看出是模型没给锚点、
还是 diff 里找不到、还是歧义。

已定位的评论里，挂在**删除行**上的会标 ``（已删除）``：它的行号是基准版本的
坐标，不标的话读者会拿它去当前文件里找，找不到还以为定位错了。

**疑似漏报**是反思阶段顺带发现的信号（「这处改动可能没被覆盖」），也单独成段放
最后。它比未定位评论更弱 —— 它连评论都不是，只是提示。

V1 简化：不渲染 suggestion_code 的 diff、不展示 rule_ref。
"""

import logging
from collections import Counter
from collections.abc import Sequence

from prism.core.output.sort import SEVERITY_RANK
from prism.models.comment import (
    LOCATE_FAILED,
    REASON_EMPTY_EXISTING_CODE,
    REASON_MULTI_MATCH,
    REASON_NO_MATCH,
    SIDE_LEFT,
    Comment,
    ReviewStats,
)
from prism.models.reflection import PotentialMiss
from prism.models.static import StaticFinding

logger = logging.getLogger(__name__)

# 头部标题
HEADER_TITLE = "=== Prism Code Review ==="

# 尾部标题
FOOTER_TITLE = "=== 统计 ==="

# 已定位评论的分隔符，如 ─── internal/agent/agent.go:245-248 ───
SEPARATOR_TEMPLATE = "─── {path}:{span}{note} ───"

# 删除侧行号的标注。**必须标**：LEFT 的行号是**基准版本**的坐标，拿去
# 当前文件里找是找不到的 —— 不标的话读者会以为定位错了
DELETED_NOTE = "（已删除）"

# 徽章，如 [concurrency · high]
BADGE_TEMPLATE = "[{category} · {severity}]"

# 未定位聚合段的标题与提示
UNLOCATED_TITLE = "=== 未定位评论（{count} 条）==="
UNLOCATED_NOTICE = "⚠️ 以下评论无法确定具体行号，请人工确认："

# 疑似漏报段的标题与提示。措辞刻意比未定位那段软 —— 它不是评论，只是提示，
# 而且反思者「编造漏报」是已知风险（见设计文档 §八）
MISSES_TITLE = "=== 疑似漏报（{count} 条）==="
MISSES_NOTICE = "⚠️ 反思时发现下面这些改动可能没有被任何评论覆盖，仅供参考："

# 静态发现未处理段的标题与提示。
#
# **它不是「静态发现」的输出通道**：静态发现本身依然不进评论列表，这里报的是
# 「哪几条没人管」—— 用户据此决定要不要人工核一下（设计文档 18 号 §2.1 的兜底）。
# 只列高优先级的：低优先级没被引用是常态，列出来只会让这段变成噪音。
STATIC_GAPS_TITLE = "=== 静态发现未处理（{count} 条）==="
STATIC_GAPS_NOTICE = "⚠️ 以下高优先级静态发现没有被任何评论引用，建议人工核查："

# 未定位原因 → 给人看的一句话。**报的是我们这边查不出匹配的原因**，
# 不是「模型错了」——空片段和歧义都不一定是模型的锅
REASON_LABELS: dict[str, str] = {
    REASON_NO_MATCH: "existing_code 未在 diff 中匹配到",
    REASON_MULTI_MATCH: "existing_code 匹配到多处，无法确定是哪一处",
    REASON_EMPTY_EXISTING_CODE: "existing_code 为空",
}

# 原因缺失或不在上表时的兜底说明
UNKNOWN_REASON_LABEL = "未定位"

# 尾部统计的展示顺序：high → medium → low
SEVERITY_ORDER: tuple[str, ...] = tuple(sorted(SEVERITY_RANK, key=lambda name: SEVERITY_RANK[name]))


class TerminalRenderer:
    """终端渲染器。

    将评论列表渲染为人类可读的终端文本。
    V1 简化：不渲染 suggestion_code 的 diff，不展示 rule_ref。
    """

    def render(
        self,
        comments: list[Comment],
        stats: ReviewStats,
        misses: Sequence[PotentialMiss] = (),
        static_gaps: Sequence[StaticFinding] = (),
    ) -> str:
        """渲染评论列表为终端文本。

        Args:
            comments: 排序后的评论列表
            stats: 审查统计信息
            misses: 反思阶段的疑似漏报信号（可为空）
            static_gaps: 没被任何评论引用的高优先级静态发现（可为空）。
                静态分析被跳过或没跑出发现时都是空

        Returns:
            格式化后的终端文本
        """
        located = [item for item in comments if item.locate_level != LOCATE_FAILED]
        failed = [item for item in comments if item.locate_level == LOCATE_FAILED]
        logger.debug(
            "渲染开始：%d 条评论（未定位 %d 条，疑似漏报 %d 条，静态发现未处理 %d 条）",
            len(comments),
            len(failed),
            len(misses),
            len(static_gaps),
        )

        blocks = [self._render_header(stats)]
        blocks.extend(self._render_comment(comment) for comment in located)
        if failed:
            blocks.append(self._render_unlocated(failed))
        if misses:
            blocks.append(self._render_misses(misses))
        if static_gaps:
            blocks.append(self._render_static_gaps(static_gaps))
        blocks.append(self._render_footer(comments))

        text = "\n\n".join(blocks)
        logger.info("渲染完成：输出 %d 字符", len(text))
        return text

    @staticmethod
    def _render_header(stats: ReviewStats) -> str:
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

    @staticmethod
    def _render_comment(comment: Comment) -> str:
        """渲染单条已定位的评论。

        格式：
        ─── path:12-14 ───
        [category · severity]
        评论正文

        单行命中只显示一个行号，不写成 ``12-12``。挂在删除行上的额外标
        ``（已删除）``，并在行号里保留**基准版本**的坐标。

        Args:
            comment: 已定位的评论

        Returns:
            渲染后的评论文本
        """
        span = (
            str(comment.start_line)
            if comment.start_line == comment.end_line
            else f"{comment.start_line}-{comment.end_line}"
        )
        note = DELETED_NOTE if comment.side == SIDE_LEFT else ""
        return "\n".join(
            (
                SEPARATOR_TEMPLATE.format(path=comment.path, span=span, note=note),
                BADGE_TEMPLATE.format(category=comment.category, severity=comment.severity),
                comment.content,
            )
        )

    @staticmethod
    def _render_unlocated(comments: list[Comment]) -> str:
        """渲染未定位聚合段。

        格式（对齐设计文档 §2.4 的样例）：
        === 未定位评论（2 条）===

        ⚠️ 以下评论无法确定具体行号，请人工确认：

        1. path
           [category · severity]
           评论正文
           （原因：...）

        Args:
            comments: 未定位的评论

        Returns:
            聚合段文本
        """
        lines = [UNLOCATED_TITLE.format(count=len(comments)), "", UNLOCATED_NOTICE]
        for index, comment in enumerate(comments, start=1):
            reason = REASON_LABELS.get(comment.locate_reason or "", UNKNOWN_REASON_LABEL)
            badge = BADGE_TEMPLATE.format(category=comment.category, severity=comment.severity)
            lines.extend(
                (
                    f"{index}. {comment.path}",
                    f"   {badge}",
                    f"   {comment.content}",
                    f"   （原因：{reason}）",
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _render_misses(misses: Sequence[PotentialMiss]) -> str:
        """渲染疑似漏报段。

        格式（与未定位那段同构）：
        === 疑似漏报（1 条）===

        ⚠️ 反思时发现下面这些改动可能没有被任何评论覆盖，仅供参考：

        1. H-07 internal/store/store.go
           新增的解锁路径似乎没有被审到

        Args:
            misses: 疑似漏报信号

        Returns:
            该段文本
        """
        lines = [MISSES_TITLE.format(count=len(misses)), "", MISSES_NOTICE]
        for index, item in enumerate(misses, start=1):
            lines.extend((f"{index}. {item.hunk_id} {item.path}", f"   {item.description}"))
        return "\n".join(lines)

    @staticmethod
    def _render_static_gaps(gaps: Sequence[StaticFinding]) -> str:
        """渲染静态发现未处理段。

        格式（与疑似漏报那段同构）：

        === 静态发现未处理（1 条）===

        ⚠️ 以下高优先级静态发现没有被任何评论引用，建议人工核查：

        1. S-03 pkg/a/iface.go:15 [gosec · high]
           G107: Potential HTTP request made with variable url

        Args:
            gaps: 未被引用的高优先级静态发现

        Returns:
            该段文本
        """
        lines = [STATIC_GAPS_TITLE.format(count=len(gaps)), "", STATIC_GAPS_NOTICE]
        for index, item in enumerate(gaps, start=1):
            badge = BADGE_TEMPLATE.format(category=item.linter, severity=item.priority)
            head = f"{index}. {item.finding_id} {item.path}:{item.line} {badge}"
            lines.extend((head, f"   {item.text}"))
        return "\n".join(lines)

    @staticmethod
    def _render_footer(comments: list[Comment]) -> str:
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
