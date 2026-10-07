"""后置校验：一组审完之后，核对评论有没有引用静态发现编号。

设计文档 18 号 §2.11 / §4.5。**不做定位、不新增评论、不改评论** —— 它只产出一份
「哪些发现没人管」的清单，最终由渲染器打成一条建议人工核查的提示。

两处口径值得记一笔：

- **校验集合是本组注入过的那批**，不是全量发现。模型只能看到注入给它的编号，
  引用了本组没注入的编号（哪怕是别的组的）在我们这边同样是「凭空出现」——
  那正是要抓的信号。
- **素材取反思后的评论**（`outcome.kept`）。反思删掉一条曾经引用 S-01 的评论之后，
  S-01 就是没人处理了 —— 按最终输出来算才对得上用户看到的东西。
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from prism.models.comment import Comment
from prism.models.static import PRIORITY_HIGH, StaticFinding

logger = logging.getLogger(__name__)

# 编号引用：`[S-01]`。宽松到允许小写与不补零（`[s-1]`），再归一化回 `S-01` ——
# 大小写与补零不是校验要抓的东西，抓的是「这个编号存不存在」
REF_PATTERN = re.compile(r"\[\s*s-(\d{1,3})\s*\]", re.IGNORECASE)

# 归一化后的编号格式，必须与 analyzer.FINDING_ID_PREFIX / WIDTH 一致
_ID_FORMAT = "S-{index:02d}"


@dataclass(frozen=True)
class StaticVerdict:
    """一组评论对静态发现的处理情况。

    Attributes:
        referenced: 被引用且**确实注入过本组**的编号（去重升序）
        invalid: 引用了但不存在于本组注入集合里的编号（去重升序）。评论一律保留，
            它只是提示我们模型的编号可能是编的
        unhandled_high: 没有被任何评论引用的**高优先级**发现。低优先级的不报
            （设计 §2.11：只对高优先级告警，否则 WARNING 变噪音）
    """

    referenced: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    unhandled_high: tuple[StaticFinding, ...] = ()

    @property
    def unhandled_count(self) -> int:
        """未处理的高优先级发现数。"""
        return len(self.unhandled_high)


def verify_group(findings: Sequence[StaticFinding], comments: Sequence[Comment]) -> StaticVerdict:
    """核对一组评论引用了哪些静态发现编号。

    Args:
        findings: 注入给这一组的静态发现（不是全量）
        comments: 这一组最终保留的评论

    Returns:
        StaticVerdict
    """
    if not findings:
        return StaticVerdict()

    known = {item.finding_id for item in findings}
    cited: set[str] = set()
    invalid: set[str] = set()
    for comment in comments:
        for match in REF_PATTERN.finditer(comment.content):
            ref = _ID_FORMAT.format(index=int(match.group(1)))
            if ref in known:
                cited.add(ref)
            else:
                invalid.add(ref)

    unhandled = tuple(
        item for item in findings if item.priority == PRIORITY_HIGH and item.finding_id not in cited
    )
    logger.debug(
        "静态发现校验：引用 %d 个（无效 %d 个），高优先级未处理 %d 条",
        len(cited),
        len(invalid),
        len(unhandled),
    )
    if invalid:
        logger.warning("评论引用了不存在的静态发现编号：%s", " ".join(sorted(invalid)))
    return StaticVerdict(
        referenced=tuple(sorted(cited)),
        invalid=tuple(sorted(invalid)),
        unhandled_high=unhandled,
    )
