"""规则库：读盘、解析、查找。

设计见 `acr-min/08-规则注入/26-规则库组织与渐进式加载设计.md`。两层：

- **L1 索引**（`rules/index.md`，始终进提示词）：只给「代码特征 → 规则代号」，
  恒定成本
- **L2 规则正文**（`rules/themes/<主题>/<代号>.md`，按需加载）：由 `load_rules` 工具取

```
rules/
├── index.md                 # L1
├── _registry.toml           # 主题枚举（编号 + 中文标题 + 一行说明）
└── themes/<主题>/<代号>.md   # L2
```

**目录名即主题**：规则文件 frontmatter 里的 `theme` 必须等于它所在的目录名，
所以主题里不带编号 —— 编号只出现在 `_registry.toml` 里，一处，不会两头漂。

**覆盖关系只有一处来源**：写在**被取代的公共规则**自己的 frontmatter 上
（`overridden_by`）。设计文档里另有一张 `OVERRIDE_MAP`（§6.2），这里没有照做 ——
那张表与规则文件说的是同一件事，两份就会漂，而工具层要的正是「现在这条规则到底
算不算数」这个事实，它只能有一个出处。

**只有「整条被取代」，没有「部分覆盖」**（设计 §2.5 那一档已废弃）：它本该在
**整理规则时**解决 —— 私有规则把公共规则里仍然有效的内容合并进自己的正文，
索引只列私有那条。运行时不合并、不解释优先级：模型看到的索引里每一条都是自洽的
现行规则。

**frontmatter 用自带的严格解析器**，不引 PyYAML：只认 `键: 值`、`键: [值, "值"]`
这两种形状，写别的就报错。文件本身仍是合法 YAML（别处用 YAML 工具照样能读），
我们只是不接受自己用不到的语法。少一个依赖，也少一类「解析器悄悄把值读成别的」
的静默失效（`on` 被读成布尔值那种）。
"""

import logging
import tomllib
from functools import lru_cache
from pathlib import Path

from prism.models.rules import OVERRIDDEN_SUFFIX, SEGMENT_PRIVATE, Rule

logger = logging.getLogger(__name__)

# 规则库根目录。它跟着包装走（`prism/rules/`），与 `config/default.toml` 同一种放法
RULES_ROOT = Path(__file__).resolve().parents[2] / "rules"

INDEX_NAME = "index.md"
REGISTRY_NAME = "_registry.toml"
THEMES_DIR = "themes"
RULE_KEYWORDS = frozenset(
    {
        "rule_id",
        "theme",
        "default_severity",
        "keywords",
        "source",
        "overridden_by",
    }
)
REQUIRED_KEYS = frozenset({"rule_id", "theme", "default_severity", "keywords", "source"})
SEVERITIES = frozenset({"high", "medium", "low"})


class RulesError(Exception):
    """规则库读不了（文件缺失、frontmatter 不合法、规则编号重复……）。

    **一律硬报错**，不降级成「没有规则」：规则库是我们自己随包发布的，
    读不了说明打包或文件坏了。静默降级会让审查照常跑，只是**再也不会**按规则判断，
    而这种失效没有任何信号。
    """


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, object], str]:
    """拆出 frontmatter 与正文。

    Args:
        text: 规则文件全文
        path: 文件路径（报错时点名）

    Returns:
        (字段字典, 正文)

    Raises:
        RulesError: 没有 frontmatter / 没闭合 / 类型不符
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise RulesError(f"{path}: 规则文件必须以 `---` 开头（frontmatter）")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise RulesError(f"{path}: frontmatter 没有闭合的 `---`") from exc

    fields: dict[str, object] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, raw = line.partition(":")
        if not sep:
            raise RulesError(f"{path}: frontmatter 这一行不是 `键: 值`：{line!r}")
        key = key.strip()
        if key not in RULE_KEYWORDS:
            hint = ""
            if key == "remaining_parts":
                hint = (
                    "。**这个字段已废弃**：部分覆盖不再在运行时合并 —— 请把公共规则里"
                    "仍然有效的内容合并进这条私有规则自己的正文，索引只列私有那条"
                )
            raise RulesError(
                f"{path}: 未知字段 {key!r}（可用：{', '.join(sorted(RULE_KEYWORDS))}）{hint}"
            )
        fields[key] = _parse_value(raw.strip(), path, key)
    return fields, "\n".join(lines[end + 1 :]).strip()


def _parse_value(raw: str, path: Path, key: str) -> object:
    """解析一个 frontmatter 值：裸标量、带引号字符串、或字符串列表。

    Args:
        raw: 冒号右边的原文
        path: 文件路径（报错时点名）
        key: 字段名（报错时点名）

    Returns:
        字符串或字符串元组
    """
    if raw.startswith("["):
        if not raw.endswith("]"):
            raise RulesError(f"{path}: {key} 的列表没有闭合的 `]`：{raw!r}")
        items = [item.strip() for item in raw[1:-1].split(",") if item.strip()]
        return tuple(_unquote(item, path, key) for item in items)
    return _unquote(raw, path, key)


def _unquote(raw: str, path: Path, key: str) -> str:
    """去掉两端引号；空值报错。

    Args:
        raw: 待处理的值
        path: 文件路径（报错时点名）
        key: 字段名（报错时点名）

    Returns:
        处理后的字符串

    Raises:
        RulesError: 值为空
    """
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise RulesError(f"{path}: {key} 的值为空")
    return value


def _read_rule(path: Path, theme: str, prefix: str) -> Rule:
    """读一条规则文件。

    Args:
        path: 规则文件路径
        theme: 它所在主题目录的名字
        prefix: 该主题的代号前缀（两字母）

    Returns:
        规则对象

    Raises:
        RulesError: 字段缺失、类型不符、`theme` 与目录名不一致、代号前缀与主题不符
    """
    fields, body = _split_frontmatter(path.read_text(encoding="utf-8"), path)
    missing = REQUIRED_KEYS - fields.keys()
    if missing:
        raise RulesError(f"{path}: 缺少字段 {', '.join(sorted(missing))}")

    declared_theme = str(fields["theme"])
    if declared_theme != theme:
        raise RulesError(
            f"{path}: frontmatter 里的 theme={declared_theme!r} 与所在目录 {theme!r} 不一致"
        )
    severity = str(fields["default_severity"])
    if severity not in SEVERITIES:
        raise RulesError(f"{path}: default_severity 只能是 {sorted(SEVERITIES)}，收到 {severity!r}")
    if not body:
        raise RulesError(f"{path}: frontmatter 之后没有正文")

    keywords = fields["keywords"]
    rule_id = str(fields["rule_id"])
    # **代号前缀必须与主题一致**：不然「EH-G-03 出现在并发目录下」这种事只能靠
    # 人眼发现，而它会让索引、目录、加载回来的正文三处互相打架
    if not rule_id.startswith(f"{prefix}-"):
        raise RulesError(f"{path}: 代号 {rule_id} 的前缀与主题 {theme!r} 不符（应为 {prefix}-…）")
    return Rule(
        rule_id=rule_id,
        theme=declared_theme,
        default_severity=severity,
        keywords=tuple(keywords) if isinstance(keywords, tuple) else (str(keywords),),
        source=str(fields["source"]),
        body=body,
        path=str(path),
        overridden_by=str(fields["overridden_by"]) if fields.get("overridden_by") else None,
    )


class RuleLibrary:
    """一整套规则：索引文本 + 按代号的查找 + 覆盖关系的落地。

    Attributes:
        index_body: 索引里**从第一个主题标题开始**的部分（提示词直接用它那段，
            这份文件自己的 H1 与优先级说明留在文件里给人看）
        themes: 主题名列表，顺序来自 `_registry.toml`
        rules: 规则代号 → 规则
    """

    def __init__(self, index_body: str, themes: list[str], rules: dict[str, Rule]) -> None:
        """由加载函数调用，别在别处直接构造。

        Args:
            index_body: 索引正文（去掉 H1 与优先级说明）
            themes: 主题顺序
            rules: 规则表
        """
        self.index_body = index_body
        self.themes = themes
        self.rules = rules

    def active(self) -> list[Rule]:
        """当前生效的规则（被完全覆盖的不算）。

        Returns:
            按代号排序的规则列表
        """
        return [item for _, item in sorted(self.rules.items()) if not item.is_overridden]

    def render(self, rule_ids: list[str]) -> str:
        """渲染 `load_rules` 的返回文本。

        两种情况（设计 26 号的 §6.1 那条已废弃，见 `render` 里那段注释）：
        **生效的**直接给正文；**没见过的代号**（含已被取代的）点名让它回索引看。

        Args:
            rule_ids: 模型给的规则代号列表

        Returns:
            给模型看的文本；全部无效时也只说事实，不抛异常
        """
        blocks: list[str] = []
        unknown: list[str] = []
        for rule_id in rule_ids:
            rule = self.rules.get(rule_id)
            if rule is None:
                unknown.append(rule_id)
                continue
            if rule.is_overridden:
                # **按「没有这条规则」处理**：索引里列的都是**现行**规则 ——
                # 有私有规则覆盖时，那一行的代号在整理规则时就换成了私有代号，
                # 所以模型正常问不到被覆盖的代号。真问到，只说明它手里的是旧
                # 信息（或猜的），让它回索引重看就行。
                #
                # **这里不解释「谁覆盖了谁」**：那套优先级语义模型不需要理解，
                # 端出来只会让它去推理两套来源的关系，而它要的只是现行内容
                unknown.append(rule_id)
                continue
            blocks.append(rule.body)

        parts = list(blocks)
        if unknown:
            parts.append(
                f"未找到规则：{', '.join(unknown)}。"
                f"可用的代号在「代码审查规则索引」那张表里，照表里的写。"
            )
        return "\n\n".join(parts) if parts else "没有可返回的规则。"


def _read_registry(root: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    """读主题枚举（顺序与代号前缀都来自它）。

    Args:
        root: 规则库根目录

    Returns:
        (主题名顺序, 主题名 → {title, summary, prefix})

    Raises:
        RulesError: 文件缺失或形状不对
    """
    path = root / REGISTRY_NAME
    if not path.exists():
        raise RulesError(f"规则库缺少 {REGISTRY_NAME}：{path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    order = list(data.get("order", []))
    themes: dict[str, dict[str, str]] = dict(data.get("themes", {}))
    if not order or set(order) != set(themes):
        raise RulesError(f"{path}: order 与 [themes] 的键必须一致（不重不漏）")
    missing = [name for name in order if not themes[name].get("prefix")]
    if missing:
        raise RulesError(f"{path}: 这些主题没有 prefix：{', '.join(missing)}")
    prefixes = [themes[name]["prefix"] for name in order]
    if len(set(prefixes)) != len(prefixes):
        raise RulesError(f"{path}: 主题前缀有重复，代号会撞车：{prefixes}")
    return order, themes


def _read_index(root: Path) -> str:
    """读 L1 索引：截掉文件自己的标题，并把主题标题**降一级**。

    降级的理由：这段文本进提示词时挂在 `## 代码审查规则索引` 下面，而文件里主题
    用的是 `##` —— 不降级的话，九个主题会与「代码审查规则索引」**同级**，
    读起来像九段并列的内容，而不是「索引里的九个主题」。

    只动**标题层级**，正文一个字不改（表格、说明、代号都可能被别处核对）。

    Args:
        root: 规则库根目录

    Returns:
        从第一个主题标题开始、且标题已降一级的正文

    Raises:
        RulesError: 文件缺失或没有主题段落
    """
    path = root / INDEX_NAME
    if not path.exists():
        raise RulesError(f"规则库缺少 {INDEX_NAME}：{path}")
    text = path.read_text(encoding="utf-8")
    index = text.find("\n## ")
    if index < 0:
        raise RulesError(f"{path}: 找不到主题段落（`## 主题名`）")
    body = text[index + 1 :].strip()
    return "\n".join(
        line if not line.startswith("## ") else "#" + line for line in body.split("\n")
    )


def load_library(root: Path | None = None) -> RuleLibrary:
    """加载整套规则库。

    Args:
        root: 规则库根目录；None 表示随包发布的那个

    Returns:
        规则库

    Raises:
        RulesError: 任何一处读不了 —— 见 `RulesError` 的说明
    """
    base = RULES_ROOT if root is None else root
    themes, registry = _read_registry(base)
    rules: dict[str, Rule] = {}
    for theme in themes:
        theme_dir = base / THEMES_DIR / theme
        if not theme_dir.is_dir():
            raise RulesError(f"主题目录不存在：{theme_dir}")
        prefix = str(registry[theme]["prefix"])
        for path in sorted(theme_dir.glob("*.md")):
            rule = _read_rule(path, theme, prefix)
            if rule.rule_id in rules:
                raise RulesError(f"规则代号重复：{rule.rule_id}（{path}）")
            rules[rule.rule_id] = rule
    if not rules:
        raise RulesError(f"规则库是空的：{base / THEMES_DIR}")

    # `.overridden` 这个后缀只给**已被取代**的规则（设计 §三）：它们不再生效、
    # 只作备查。部分是部分覆盖的仍然生效，文件名保持原样 —— 后缀与「还生效吗」
    # 必须一一对应，否则「这条规则算不算数」就要靠读 frontmatter 才知道，
    # 而 glob 出来的文件名list 会给出相反的印象
    for rule in rules.values():
        marked = Path(rule.path).stem.endswith(OVERRIDDEN_SUFFIX)
        if marked != rule.is_overridden:
            raise RulesError(
                f"{rule.path}: 文件名与它的覆盖声明对不上"
                f"（文件名带 .overridden={marked}，frontmatter overridden_by={rule.overridden_by}）"
            )
        if rule.overridden_by is not None:
            _check_override_target(rule, rules)
    return RuleLibrary(index_body=_read_index(base), themes=themes, rules=rules)


def _check_override_target(rule: Rule, rules: dict[str, Rule]) -> None:
    """覆盖者必须存在，而且不能是公共规则（C 覆盖 G，不是反过来）。

    Args:
        rule: 被覆盖的规则
        rules: 全部规则

    Raises:
        RulesError: 覆盖者不存在，或它不是私有规则
    """
    target = str(rule.overridden_by)
    if target not in rules:
        raise RulesError(f"{rule.path}: 声明的覆盖者 {target} 不存在")
    if SEGMENT_PRIVATE not in rule.rule_id and SEGMENT_PRIVATE not in target:
        raise RulesError(
            f"{rule.path}: 只有私有规则（<主题>-C-xx）能覆盖公共规则，{target} 不是私有规则"
        )


@lru_cache(maxsize=1)
def default_library() -> RuleLibrary:
    """随包发布的规则库（进程内缓存一次）。

    用「缓存一份」而不是「每次读盘」：`build_system()` 在分组装箱里会被反复调用
    （O(k²) 次渲染量成本，见「分组与预算」），每次都读盘会把零 token 的干跑拖慢。
    **缓存的是不可变的一份**，改了规则文件要重启进程才生效 —— 规则库随包发布，
    本来就不会在运行中变。

    Returns:
        规则库
    """
    return load_library()


def reset_cache() -> None:
    """清掉缓存（用例改规则文件时用）。"""
    default_library.cache_clear()
