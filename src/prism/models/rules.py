"""规则库的纯数据模型。

只描述数据结构，不含行为、不含可变状态（与 `models/tool.py` 同一条规矩）。
解析与查找在 `core/rules/library.py`。

规则的**正文**（`body`）原样存着，不在解析阶段改写：它是要发给模型看的文本，
中途加工一手就说不清模型看到的到底是哪一版。
"""

from dataclasses import dataclass, field

# 两个来源前缀（设计 26 号 §2.5）：C 是私有规范，G 是公共规范，C 覆盖 G。
PREFIX_PRIVATE = "C-"
PREFIX_PUBLIC = "G-"

# 被覆盖的公共规则在磁盘上的后缀（`G-05.overridden.md`），扫描时认它
OVERRIDDEN_SUFFIX = ".overridden"


@dataclass
class Rule:
    """一条规则。

    Attributes:
        rule_id: 规则代号（G-01 / C-02）
        theme: 所属主题 —— **就是它的目录名**，不是另一套代号
        default_severity: 默认严重程度（high / medium / low）。
            **只是建议值**：模型可以在它的基础上下调，不许上调（设计 26 号 §2.4）
        keywords: 机器扫描用的关键词（当前只作记录，模型按索引里的「代码特征」命中）
        source: 来源（company / google）
        body: 规则正文（frontmatter 之后的全部内容，原样）
        path: 规则文件在仓库内的相对路径（报错与排查时用）
        overridden_by: 被哪条私有规则覆盖；None 表示没被覆盖
        remaining_parts: 被覆盖时**仍然有效**的部分（空 = 完全覆盖）
    """

    rule_id: str
    theme: str
    default_severity: str
    keywords: tuple[str, ...]
    source: str
    body: str
    path: str
    overridden_by: str | None = None
    remaining_parts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fully_overridden(self) -> bool:
        """是否被**完全**覆盖（覆盖者存在，且没有留下仍然有效的部分）。

        Returns:
            True 表示这条规则在当前规则库里已经不生效
        """
        return self.overridden_by is not None and not self.remaining_parts
