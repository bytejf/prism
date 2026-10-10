"""后置校验：一组审完之后，核对评论里的**两类引用**。

一类是**静态发现编号**（`S-01`），一类是**规则代号**（`EH-G-03`）。两类都只做
核对，**不做定位、不新增评论、不改评论** —— 产出「哪些发现没人管」「哪些引用
是编的」两份清单，前者由渲染器打成建议人工核查的提示。

评论里的规则代号**现在有结构化字段**（`Comment.rule_ref`），所以那一类直接读
字段；正文里的 `[S-01]` 那种写法则照旧用正则兜底扫（模型仍可能在正文里重复一遍
编号，两种都要认）。

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

from prism.core.output.sort import SEVERITY_RANK
from prism.core.rules.library import default_library
from prism.models.comment import Comment
from prism.models.static import PRIORITY_HIGH, StaticFinding

logger = logging.getLogger(__name__)

# 编号引用：`[S-01]`。宽松到允许小写与不补零（`[s-1]`），再归一化回 `S-01` ——
# 大小写与补零不是校验要抓的东西，抓的是「这个编号存不存在」
REF_PATTERN = re.compile(r"\[\s*s-(\d{1,3})\s*\]", re.IGNORECASE)

# 结构化字段里的编号：`S-01` / `s-1` / 带空格都算，归一化交给 `_normalize_finding_ref`
FIELD_PATTERN = re.compile(r"^\s*s-?(\d{1,3})\s*$", re.IGNORECASE)

# 归一化后的编号格式，必须与 analyzer.FINDING_ID_PREFIX / WIDTH 一致
_ID_FORMAT = "S-{index:02d}"


@dataclass(frozen=True)
class StaticVerdict:
    """一组评论对静态发现的处理情况。

    Attributes:
        referenced: 被引用且**确实注入过本组**的编号（去重升序）
        invalid: 引用了但不存在于本组注入集合里的编号（去重升序）。评论一律保留，
            它只是提示我们模型的编号可能是编的
        unhandled_high: 没有任何评论**提到**的**高优先级**发现（字段或正文里的编号
            都算提到）。低优先级的不报
            （设计 §2.11：只对高优先级告警，否则 WARNING 变噪音）
    """

    referenced: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    unhandled_high: tuple[StaticFinding, ...] = ()
    upgraded_rules: tuple[str, ...] = ()
    """评论**填了规则代号、但严重程度高于该规则默认值**的那些代号（去重升序）。

    设计 §2.4：规则的严重程度是**默认值**，只许下调。高了意味着「这是规则之外的
    问题」，那种情况本就不该填 `rule_ref`（应按独立发现报）。这里只记下来，
    **不改评论** —— 严重程度是模型的判断，我们只在旁边标一句。"""

    invalid_rules: tuple[str, ...] = ()
    """评论里引用了、但规则库里没有的规则代号（去重升序）。

    与 `invalid` 同一个意思，只是对象换成规则：**引用了不存在的代号**。评论一律
    保留（模型写的话不该被我们悄悄改掉），但报告里要指出来 —— 那条评论的
    「依据」在界面上看起来和真的一样。"""

    @property
    def unhandled_count(self) -> int:
        """未处理的高优先级发现数。"""
        return len(self.unhandled_high)


def verify_group(findings: Sequence[StaticFinding], comments: Sequence[Comment]) -> StaticVerdict:
    """核对一组评论里的两类引用，并算出哪些静态发现没人管。

    **没有静态发现时也照常跑**：规则代号的校验与静态发现无关。

    Args:
        findings: 注入给这一组的静态发现（不是全量）
        comments: 这一组最终保留的评论

    Returns:
        StaticVerdict
    """
    known = {item.finding_id for item in findings}
    cited: set[str] = set()
    invalid: set[str] = set()
    invalid_rules: set[str] = set()
    upgraded_rules: set[str] = set()
    for comment in comments:
        for ref in _finding_refs(comment):
            if ref in known:
                cited.add(ref)
            else:
                invalid.add(ref)
        if comment.rule_ref:
            _check_rule_ref(comment.rule_ref, invalid_rules)
            _check_rule_severity(comment, upgraded_rules)

    unhandled = tuple(
        item for item in findings if item.priority == PRIORITY_HIGH and item.finding_id not in cited
    )
    logger.debug(
        "后置校验：静态引用 %d 个（无效 %d 个）、高优先级未处理 %d 条；规则引用无效 %d 个",
        len(cited),
        len(invalid),
        len(unhandled),
        len(invalid_rules),
    )
    if invalid:
        logger.warning("评论引用了不存在的静态发现编号：%s", " ".join(sorted(invalid)))
    if invalid_rules:
        logger.warning(
            "评论引用了规则库里没有的规则代号：%s（这条评论的「依据」在界面看起来和真的一样，"
            "人工看一眼）",
            " ".join(sorted(invalid_rules)),
        )
    if upgraded_rules:
        logger.warning(
            "评论的严重程度高于它引用的规则默认值：%s（设计上「比默认更严重」意味着发现了"
            "规则之外的问题，应作为独立发现报、不填 rule_ref）",
            " ".join(sorted(upgraded_rules)),
        )
    return StaticVerdict(
        referenced=tuple(sorted(cited)),
        invalid=tuple(sorted(invalid)),
        unhandled_high=unhandled,
        invalid_rules=tuple(sorted(invalid_rules)),
        upgraded_rules=tuple(sorted(upgraded_rules)),
    )


def _finding_refs(comment: Comment) -> list[str]:
    """一条评论引用的静态发现编号（结构化字段优先，正文正则兜底）。

    两条都收：字段是**教给模型的写法**，而正文里的 `[S-01]` 是模型自己顺手写的
    习惯 —— 只认字段的话，它写在正文里我们就当没引过（实测有一条正是这样：
    模型在正文里写「S-01 的 G404 在这里不成立」，报告却据此说「S-01 没有被任何
    评论引用」，自相矛盾）。

    Args:
        comment: 评论

    Returns:
        归一化后的编号列表（可能有重复，调用方去重）
    """
    refs: list[str] = []
    if comment.finding_ref:
        match = FIELD_PATTERN.match(comment.finding_ref)
        if match is not None:
            refs.append(_ID_FORMAT.format(index=int(match.group(1))))
    refs.extend(
        _ID_FORMAT.format(index=int(match.group(1)))
        for match in REF_PATTERN.finditer(comment.content)
    )
    return refs


def _check_rule_severity(comment: Comment, upgraded: set[str]) -> None:
    """核对「填了规则代号」的评论，严重程度有没有**高于**该规则的默认值。

    规则库读不了时跳过（同 `_check_rule_ref`）。

    Args:
        comment: 评论（带 rule_ref）
        upgraded: 收集「被上调」的规则代号（原地更新）
    """
    try:
        rules = default_library().rules
    # 有意的宽except：规则库坏了不该让审查崩在后置校验上（见 `_check_rule_ref`）
    except Exception:
        logger.exception("规则库读取失败，跳过严重程度上调校验")
        return
    rule = rules.get((comment.rule_ref or "").strip())
    if rule is None:
        return
    actual = SEVERITY_RANK.get(comment.severity)
    allowed = SEVERITY_RANK.get(rule.default_severity)
    if actual is None or allowed is None:
        return
    # SEVERITY_RANK 是「排序用」的，high=0 —— **越小越严重**，所以上调是 actual < allowed
    if actual < allowed:
        upgraded.add(rule.rule_id)


def _check_rule_ref(rule_ref: str, invalid: set[str]) -> None:
    """核对一个规则代号在不在规则库里（不在就记进 `invalid`）。

    规则库读不了时不抛异常（那是 `RulesError`，在加载那条路上就该响了），
    这里只跳过 —— 后置校验不能因为规则库出问题而把整条 review 带走。

    Args:
        rule_ref: 评论里引用的规则代号
        invalid: 收集无效代号的集合（原地更新）
    """
    try:
        known = default_library().rules
    # 这里是有意的宽except：规则库坏了也不该让审查崩在这（见函数 docstring）
    except Exception:
        logger.exception("规则库读取失败，跳过规则代号校验")
        return
    if rule_ref.strip() not in known:
        invalid.add(rule_ref.strip())
