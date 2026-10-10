"""规则库的单元测试：解析、覆盖关系、索引漂移守卫。

**漂移守卫是这里最主要的内容**：规则的**内容**没有对错，但「索引里写了什么」与
「磁盘上有什么」这两份东西一旦漂开，模型会去加载一个不存在的规则、或者**永远
看不到**某条已经写好的规则 —— 两种失效都不报错。
"""

import re
from pathlib import Path

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from prism.core.guidance.builder import build_system
from prism.core.rules.library import (
    RULES_ROOT,
    THEMES_DIR,
    RulesError,
    load_library,
)
from prism.core.tools.load_rules import load_rules
from prism.models.rules import OVERRIDDEN_SUFFIX

# 索引里出现的规则代号（`EH-G-01` / `CX-C-05` 这种形态：主题前缀 + 来源 + 组内序号）
RULE_CODE = re.compile(r"\b[A-Z]{2}-[CG]-\d{2}\b")


PREFIX_BY_THEME = {"error-handling": "EH", "concurrency": "CC", "context": "CX", "naming": "NM"}


def make_library(tmp_path: Path, *, rules: dict[str, str], index: str) -> Path:
    """搭一个最小规则库，供各种形状的用例用。

    Args:
        tmp_path: 临时目录
        rules: 相对路径 → 规则文件内容（如 `{"error-handling/EH-G-01.md": "..."}`）
        index: 索引文件内容

    Returns:
        规则库根目录
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    themes = sorted({Path(name).parent.name for name in rules})
    (tmp_path / "index.md").write_text(index, encoding="utf-8")
    registry = "\n".join(
        [
            f"order = [{', '.join(f'"{name}"' for name in themes)}]",
            *[
                f'[themes."{name}"]\nprefix = "{PREFIX_BY_THEME.get(name, name[:2].upper())}"'
                f'\ntitle = "{name}"\nsummary = "说明"'
                for name in themes
            ],
        ]
    )
    (tmp_path / "_registry.toml").write_text(registry + "\n", encoding="utf-8")
    for name, content in rules.items():
        path = tmp_path / THEMES_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def rule_text(
    rule_id: str = "EH-G-01",
    *,
    theme: str = "error-handling",
    severity: str = "medium",
    extra: str = "",
) -> str:
    """造一份合法的规则文件内容。

    Args:
        rule_id: 规则代号
        theme: 主题（要等于它的目录名）
        severity: 默认严重程度
        extra: 额外插进 frontmatter 的行

    Returns:
        规则文件全文
    """
    return (
        f"---\nrule_id: {rule_id}\ntheme: {theme}\ndefault_severity: {severity}\n"
        f"keywords: [err]\nsource: google\n{extra}---\n\n# {rule_id}：标题\n\n正文\n"
    )


def _private(content: str) -> str:
    """把一条规则的来源改成 company（私有规则）。

    Args:
        content: 规则文件全文

    Returns:
        改过来源的全文
    """
    return content.replace("source: google", "source: company")


INDEX = (
    "# 索引\n\n## 错误处理\n说明\n\n| 代码特征 | 规则代号 |\n|---|---|\n| 忽略 err | EH-G-01 |\n"
)
INDEX_PRIVATE = (
    "# 索引\n\n## 错误处理\n说明\n\n| 代码特征 | 规则代号 |\n|---|---|\n| 忽略 err | EH-C-01 |\n"
)
INDEX_CONTEXT = (
    "# 索引\n\n## Context\n说明\n\n| 代码特征 | 规则代号 |\n|---|---|\n| ctx | CX-G-01 |\n"
)


# ---------- 随包发布的规则库 ----------


def test_shipped_library_loads_and_is_self_consistent() -> None:
    """随包的规则库能加载，且九个主题都真的有规则。

    「主题目录存在但一条规则都没有」会渲染出一个空标题 —— 模型看到「并发」
    这个标题下面什么都没有，只会以为规则库坏了。
    """
    library = load_library()
    assert library.themes[0] == "error-handling"
    assert len(library.themes) == 9
    assert len(library.active()) >= 20, "初始规则集要有规模，否则索引撑不起导航"
    for theme in library.themes:
        assert any(item.theme == theme for item in library.active()), f"{theme} 下没有规则"
    assert all(item.source == "google" for item in library.rules.values()), (
        "私有规则（company）还没整理 —— 见设计文档 §十二 第 8 项"
    )


def test_index_lists_every_active_rule_exactly_as_it_is_named() -> None:
    """**索引与磁盘不漂**：索引里每个代号都存在，每条生效规则也都出现在索引里。

    两边的失效方向相反，都要挡：

    - 索引里有、磁盘上没有 → 模型去加载一个不存在的规则，白费一轮
    - 磁盘上有、索引里没有 → 这条规则**永远加载不了**（模型不知道它存在）,
      而它写在那儿看起来完全正常
    """
    library = load_library()
    listed = set(RULE_CODE.findall(library.index_body))
    assert listed, "索引里一个代号都没有？"
    assert listed <= set(library.rules), f"索引里有不存在的代号：{listed - set(library.rules)}"
    missing = {item.rule_id for item in library.active()} - listed
    assert not missing, f"这些规则没进索引，模型看不到它们：{sorted(missing)}"
    assert listed <= {item.rule_id for item in library.active()}, "被完全覆盖的规则不该出现在索引里"


def test_index_stays_within_the_line_budget() -> None:
    """索引 ≤ 80 行（设计 §4.2）。

    这条是**给未来的自己设的闸**：规则库一涨，索引就会跟着涨，而它是**每次都进
    提示词**的固定成本。超了不是改这个数字，是往下挪规则（只留 blocking + 高频）。
    """
    lines = (RULES_ROOT / "index.md").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 80, f"索引 {len(lines)} 行，超预算了"


def test_index_sections_follow_the_registry_order() -> None:
    """索引里的主题顺序就是注册表里的顺序（人改目录时最先忘的就是它）。"""
    library = load_library()
    titles = [f"## {title}" for title in _titles(library.themes)]
    positions = [library.index_body.index(title) for title in titles]
    assert positions == sorted(positions)


def _titles(themes: list[str]) -> list[str]:
    """主题名 → 中文标题（从注册表读）。

    Args:
        themes: 主题名列表

    Returns:
        中文标题列表
    """
    import tomllib

    data = tomllib.loads((RULES_ROOT / "_registry.toml").read_text(encoding="utf-8"))
    return [str(data["themes"][name]["title"]) for name in themes]


# ---------- frontmatter 解析 ----------


def test_frontmatter_rejects_unknown_fields_and_bad_values(tmp_path: Path) -> None:
    """frontmatter 只认自己那几个字段 —— 写错了**当场报错**，不静默忽略。

    静默忽略的后果是「加了个字段以为生效了，其实从来没被读过」，而规则库是随包
    发布的、没人会去核对每一条。
    """
    for index, (name, content, message) in enumerate(
        [
            ("error-handling/EH-G-01.md", rule_text(extra="unknown_key: 1\n"), "未知字段"),
            ("error-handling/EH-G-01.md", rule_text(severity="critical"), "default_severity"),
            ("naming/NM-G-01.md", rule_text("NM-G-01", theme="error-handling"), "不一致"),
            ("error-handling/G-03.md", "没有 frontmatter\n", "必须"),
            ("error-handling/G-04.md", "---\nrule_id: G-04\n", "没有闭合"),
        ]
    ):
        root = make_library(tmp_path / f"case{index}", rules={name: content}, index=INDEX)
        with pytest.raises(RulesError, match=message):
            load_library(root)


def test_missing_theme_directory_is_reported(tmp_path: Path) -> None:
    """注册表里写了主题、磁盘上没有目录 → 报错（不是当成「这个主题没规则」）。"""
    root = make_library(tmp_path, rules={"error-handling/EH-G-01.md": rule_text()}, index=INDEX)
    (root / "_registry.toml").write_text(
        'order = ["error-handling", "concurrency"]\n'
        '[themes.error-handling]\nprefix = "EH"\ntitle = "错误处理"\nsummary = "x"\n'
        '[themes.concurrency]\nprefix = "CC"\ntitle = "并发"\nsummary = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(RulesError, match="主题目录不存在"):
        load_library(root)


def test_duplicate_rule_id_is_reported(tmp_path: Path) -> None:
    """两条规则用同一个代号 → 报错（后一条会静默盖掉前一条）。

    **要造出这个形态得靠一个名字不规范的文件**：代号自带主题前缀之后，跨主题
    撞号已经不可能（前缀就不同），同主题同序号只可能来自「文件名没跟着改」
    —— 比如有人复制了一份改名时漏了，于是 `EH-G-01-old.md` 里还写着 `EH-G-01`。
    这正是这条校验要挡的现场：不报的话，前一条规则会被后一条**静默替代**，
    而加载日志里看不出任何异常。
    """
    root = make_library(
        tmp_path,
        rules={
            "error-handling/EH-G-01.md": rule_text(),
            "error-handling/EH-G-01-old.md": rule_text(),
        },
        index=INDEX,
    )
    with pytest.raises(RulesError, match="代号重复"):
        load_library(root)


# ---------- 覆盖关系 ----------


def test_fully_overridden_rule_drops_out_of_the_index_and_points_elsewhere(
    tmp_path: Path,
) -> None:
    """完全覆盖：索引里不再出现，加载它时指路到覆盖者（设计 §2.5）。"""
    root = make_library(
        tmp_path,
        rules={
            "error-handling/EH-G-01.overridden.md": rule_text(extra="overridden_by: EH-C-01\n"),
            "error-handling/EH-C-01.md": _private(rule_text("EH-C-01")),
        },
        index=INDEX_PRIVATE,
    )
    library = load_library(root)

    assert library.rules["EH-G-01"].fully_overridden
    assert "EH-G-01" not in {item.rule_id for item in library.active()}
    # 问它要一条被取代的规则 → **直接把现行那一条的正文给回去**，不让模型再发
    # 一次请求（设计 §6.1 写的是「告诉它去加载 X」，那一版要多花一轮）
    text = library.render(["EH-G-01"])
    assert "已被取代，以下是现行规则" in text
    assert "# EH-C-01" in text, "现行规则的正文要一并发出来"
    assert "# EH-G-01" not in text, "被取代的那条内容不再生效，不要混进来"


def test_partially_overridden_rule_returns_both_sides(tmp_path: Path) -> None:
    """部分覆盖：给出仍然有效的部分 + 覆盖者的正文（不做机械拼接，见实现注释）。"""
    root = make_library(
        tmp_path,
        rules={
            "context/CX-G-01.md": rule_text(
                "CX-G-01",
                theme="context",
                extra='overridden_by: CX-C-05\nremaining_parts: ["值传递"]\n',
            ),
            "context/CX-C-05.md": _private(rule_text("CX-C-05", theme="context")),
        },
        index=INDEX_CONTEXT,
    )
    library = load_library(root)

    assert not library.rules["CX-G-01"].fully_overridden
    text = library.render(["CX-G-01"])
    assert "部分覆盖" in text
    assert "值传递" in text, "仍然有效的部分要说出来"
    assert "# CX-C-05" in text, "覆盖者也一并发给模型"


def test_override_declaration_must_match_the_filename(tmp_path: Path) -> None:
    """文件名带 `.overridden` 而 frontmatter 没声明（或反过来）→ 报错。

    这两处对不上说明改名与内容脱节了 —— 那时「这条规则还生效吗」谁也答不上来。
    """
    root = make_library(
        tmp_path,
        rules={"error-handling/EH-G-01.md": rule_text()},
        index=INDEX,
    )
    (root / THEMES_DIR / "error-handling" / "EH-G-01.md").rename(
        root / THEMES_DIR / "error-handling" / f"EH-G-01{OVERRIDDEN_SUFFIX}.md"
    )
    with pytest.raises(RulesError, match="对不上"):
        load_library(root)


def test_only_private_rules_may_cover_public_ones(tmp_path: Path) -> None:
    """公共规则不能声明覆盖另一条公共规则 —— 覆盖方向只有 C → G。"""
    root = make_library(
        tmp_path,
        rules={
            "error-handling/EH-G-01.md": rule_text(),
            "error-handling/EH-G-02.overridden.md": rule_text(
                "EH-G-02", extra="overridden_by: EH-G-01\n"
            ),
        },
        index=INDEX,
    )
    with pytest.raises(RulesError, match="不是私有规则"):
        load_library(root)


def test_override_target_must_exist(tmp_path: Path) -> None:
    """声明的覆盖者不存在 → 报错。"""
    root = make_library(
        tmp_path,
        rules={
            "error-handling/EH-G-01.overridden.md": rule_text(extra="overridden_by: EH-C-99\n"),
        },
        index=INDEX,
    )
    with pytest.raises(RulesError, match="不存在"):
        load_library(root)


# ---------- 与提示词的集成 ----------


def test_system_carries_the_whole_index_and_the_load_hint() -> None:
    """索引整段进 system，并且带上「怎么加载」的说明。

    只注入表格、不说 `load_rules`，模型会以为**表里那几行就是规则全文**。
    """
    system = build_system()
    library = load_library()

    assert "## 代码审查规则索引" in system
    assert "load_rules" in system
    assert "一次最多 5 条" in system
    # 索引正文逐字节进提示词（去掉文件自己那几行标题）
    assert library.index_body in system
    # 每个主题标题都在（主题说明也跟着走）
    for title in _titles(library.themes):
        assert f"## {title}" in system
    # 文件自己的 H1 不进提示词：段标题已经有了，重复一遍是同一句话说两次
    assert system.count("# 代码审查规则索引") == 1


# ---------- load_rules 工具 ----------


def test_load_rules_returns_the_rule_body() -> None:
    """工具返回规则正文（模型看到的就是规则文件里的那段）。"""
    text = load_rules.invoke({"rule_ids": ["EH-G-01"]})

    assert "# EH-G-01" in text
    assert "正例" in text and "反例" in text and "边界情况" in text
    assert "**默认严重程度**：medium" in text


def test_load_rules_rejects_more_than_five_at_once() -> None:
    """超过 5 条**整次拒绝**，并说清为什么 —— 不悄悄截断。

    截断会让模型以为自己看到了全部，而它恰恰是因为「命中太多」才需要被拦一下：
    一次要 6 条以上，说明这一处涉及的主题太杂，该先挑最对得上的那几条。
    """
    six = ["EH-G-01", "EH-G-02", "CC-G-01", "CX-G-01", "CM-G-01", "TS-G-01"]
    text = load_rules.invoke({"rule_ids": six})

    assert "最多加载 5 条" in text
    assert "# EH-G-01" not in text, "拒绝就是整次不执行，不能给一半"


def test_load_rules_reports_unknown_ids_without_losing_the_valid_ones() -> None:
    """未知代号单独点出来，认得的那几条照常返回（部分成功好过整份失败）。"""
    text = load_rules.invoke({"rule_ids": ["EH-G-01", "EH-G-99"]})

    assert "# EH-G-01" in text
    assert "未找到规则：EH-G-99" in text
    assert "索引" in text, "要告诉它去哪儿找对的代号"


def test_load_rules_on_empty_list_says_so() -> None:
    """空列表给一句明确的话，不返回空串（空串会被读成「没有规则」）。"""
    assert "没有给规则代号" in load_rules.invoke({"rule_ids": []})


def test_load_rules_description_explains_the_cap() -> None:
    """上限写在**工具描述**里：模型不看代码，它只看描述。"""
    description = convert_to_openai_tool(load_rules)["function"]["description"]

    assert "一次最多 5 条" in description
    assert "代码审查规则索引" in description


def test_index_themes_are_demoted_one_level_when_injected() -> None:
    """索引里的主题标题注进提示词时**降一级**（`## 错误处理` → `### 错误处理`）。

    不降级的话，九个主题会与「## 代码审查规则索引」**同级** —— 读起来像九段并列
    的内容，而不是「索引里的九个主题」。文件里保留 `##`（它自己的 H1 是
    `# 代码审查规则索引`，对一份独立文档来说层级是对的）。
    """
    library = load_library()
    system = build_system()

    assert "### 错误处理" in library.index_body
    assert "\n## 错误处理" not in library.index_body, "注入的正文里不该再有一级标题"
    assert "\n## 错误处理" not in system
    assert "### 错误处理" in system, "降级后的标题要真的进到提示词"
    # 文件自身仍是 `##`（给人读的那份没被动）
    raw = (RULES_ROOT / "index.md").read_text(encoding="utf-8")
    assert "\n## 错误处理\n" in raw
