"""配置读取的单元测试。

这里守的主要不是「读得对」，而是**默认值只有一份**这件事不会被悄悄破坏：
toml 里加了键没人读、dataclass 加了字段没人填，两个方向都要红。
"""

import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

from prism.config.settings import (
    CONTAINERS,
    PATH,
    SCHEMA,
    load,
    settings,
)


def toml_section(path: str) -> dict[str, object]:
    """取出 toml 里某一段的原始字典。

    Args:
        path: 点分键路径，如 ``tools.file_read``

    Returns:
        该段的字典
    """
    node: object = raw_config()
    for part in path.split("."):
        assert isinstance(node, dict)
        node = node[part]
    assert isinstance(node, dict)
    return node


def raw_config() -> dict[str, object]:
    """读出未加工的 toml 字典。

    Returns:
        配置文件的原始结构
    """
    return tomllib.loads(PATH.read_text(encoding="utf-8"))


def with_line(old: str, new: str) -> str:
    """把配置文本里的一行换掉，供「配错」类用例用。

    Args:
        old: 要被替换的原文（必须存在且唯一）
        new: 替换后的文本

    Returns:
        替换后的完整配置文本
    """
    text = PATH.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"配置里 {old!r} 不是唯一出现"
    return text.replace(old, new)


def write_config(tmp_path: Path, text: str) -> Path:
    """把配置文本落到临时文件。

    Args:
        tmp_path: 临时目录
        text: 配置文本

    Returns:
        文件路径
    """
    target = tmp_path / "config.toml"
    target.write_text(text, encoding="utf-8")
    return target


# ---------- 键与字段的一致性 ----------


@pytest.mark.parametrize("path", sorted(SCHEMA))
def test_toml_keys_match_dataclass_fields(path: str) -> None:
    """测试：每段的 toml 键与它的 dataclass 字段**完全相等**。

    两边都算一遍，挡住两个方向的漂移：

    - dataclass 加了字段而 toml 没加 → 右边多出一个键，不等
    - toml 多写了键 → 左边多出一个键，不等
      （这条 `load()` 自己也会报未知键，但那里要跑到读取才炸；
      这里在收集阶段就能发现）

    `SCHEMA` 是读取与校验**共用**的那份映射，所以它自己不可能偏 —— 偏了
    `load()` 立刻报未知键。
    """
    model, excluded = SCHEMA[path]
    expected = {field.name for field in fields(model)} - excluded

    assert set(toml_section(path)) == expected


def test_container_sections_match() -> None:
    """测试：`[tools]` 这类只当容器的段，键就是它声明的子段名。

    挡住「加了新子段却忘了写进 `CONTAINERS`」—— 那样 `load()` 会把它当未知键
    报掉，但报出来的是「未知配置项 file_write」，不如这里直接说清。
    """
    assert set(toml_section("tools")) == set(CONTAINERS["tools"])


# ---------- 读到的东西 ----------


def test_load_maps_every_value() -> None:
    """测试：每一段的值都真的落到了 Settings 上。

    逐字段比对，而不是只看 `load()` 不抛异常 —— 后者在「读了但接错了字段」
    时照样通过。
    """
    table = raw_config()
    assert isinstance(table, dict)
    llm_raw = table["llm"]
    compression_raw = table["compression"]
    review_raw = table["review"]
    log_raw = table["log"]
    assert isinstance(llm_raw, dict)
    assert isinstance(compression_raw, dict)
    assert isinstance(review_raw, dict)
    assert isinstance(log_raw, dict)

    assert settings.llm.model == llm_raw["model"]
    assert settings.llm.base_url == llm_raw["base_url"]
    assert settings.llm.reasoning_effort == llm_raw["reasoning_effort"]
    assert settings.llm.max_retries == llm_raw["max_retries"]
    assert settings.llm.timeout == llm_raw["timeout"]
    assert settings.llm.context_window == llm_raw["context_window"]

    assert settings.compression.reserve_tokens == compression_raw["reserve_tokens"]
    assert settings.compression.max_compressions == compression_raw["max_compressions"]
    assert (
        settings.compression.max_consecutive_failures == compression_raw["max_consecutive_failures"]
    )

    assert settings.review.max_group_tokens == review_raw["max_group_tokens"]
    assert settings.review.max_super_steps == review_raw["max_super_steps"]
    assert settings.review.explore_close_ratio == review_raw["explore_close_ratio"]
    assert settings.review.max_rejections == review_raw["max_rejections"]
    # toml 数组读进来是 list，配置里存成 tuple
    assert (
        list(settings.review.no_progress_remind_rounds) == review_raw["no_progress_remind_rounds"]
    )

    assert settings.log.level == log_raw["level"]


def test_context_window_comes_from_llm_section(tmp_path: Path) -> None:
    """测试：压缩的窗口**跟着 `[llm]` 走**，不是一个独立的值。

    注意不能只断言 `settings.compression.context_window == settings.llm.context_window`
    —— 两边都等于默认的 1_000_000，把实现改成写死 `1_000_000` 它照样通过，
    那就成了恒等式（**实测踩过**：按原来那种写法做变异测试不红）。

    所以这里把 `[llm]` 的窗口改成一个**不可能被写死**的数，看压缩跟不跟。
    """
    path = write_config(tmp_path, with_line("context_window = 1000000", "context_window = 123456"))

    loaded = load(path)

    assert loaded.llm.context_window == 123456
    assert loaded.compression.context_window == 123456, "压缩的窗口应跟着 [llm] 走"
    assert loaded.compression.usable_tokens == 123456 - loaded.compression.reserve_tokens


def test_usable_tokens_is_sane() -> None:
    """测试：可用额度为正。

    预留大于窗口时 `usable_tokens` 变负，触发判定会永远成立 —— 每轮都压缩。
    """
    assert settings.compression.usable_tokens > 0


# ---------- 配错了要响 ----------


def test_unknown_key_in_section_is_rejected(tmp_path: Path) -> None:
    """测试：拼错的键报的是**那个拼错的键**，不是「少了原来那个」。

    把 `reserve_tokens` 写成 `reserve_token` 时有两件事同时成立：少了一个、
    多了一个。先读值会先撞上「缺少 reserve_tokens」—— 那指向错误的方向，
    用户会去文件里找 reserve_tokens，而它其实就在那儿。所以进段先查未知键。
    """
    path = write_config(tmp_path, with_line("reserve_tokens = 20000", "reserve_token = 1"))

    with pytest.raises(ValueError, match=r"\[compression\] 存在未知配置项：reserve_token"):
        load(path)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    """测试：多出来的整段也要报错。"""
    path = write_config(tmp_path, PATH.read_text(encoding="utf-8") + "\n[nonsense]\nx = 1\n")

    with pytest.raises(ValueError, match=r"未知配置项：nonsense"):
        load(path)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    """测试：删掉一个键要报出来，而不是悄悄用上某个隐含默认值。

    与上一条的区别：这里没有「多出来的键」，只剩「少了一个」，报的就是它。
    """
    path = write_config(tmp_path, with_line("max_lines = 500\n", ""))

    with pytest.raises(ValueError, match=r"\[tools\.file_read\] 缺少配置项 max_lines"):
        load(path)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    """测试：类型不符要报出来。

    tomllib 不校验类型，`max_lines = "500"` 会一路混进工具里 —— 那时
    `end - start + 1 > "500"` 抛的是 TypeError，离原因已经很远。
    """
    path = write_config(tmp_path, with_line("max_lines = 500", 'max_lines = "500"'))

    with pytest.raises(ValueError, match=r"max_lines 应为整数，实为 str"):
        load(path)


def test_bool_is_not_an_int(tmp_path: Path) -> None:
    """测试：`true` 不能被当成整数 1。

    `bool` 是 `int` 的子类，只写 `isinstance(value, int)` 会让
    `max_lines = true` 静默变成 1 —— 一次读一行，症状离原因极远。
    """
    path = write_config(tmp_path, with_line("max_lines = 500", "max_lines = true"))

    with pytest.raises(ValueError, match=r"max_lines 应为整数，实为 bool"):
        load(path)


def test_wrong_element_type_in_array_is_rejected(tmp_path: Path) -> None:
    """测试：数组里混进非整数也要报。"""
    path = write_config(
        tmp_path,
        with_line("no_progress_remind_rounds = [5, 7]", 'no_progress_remind_rounds = ["5", 7]'),
    )

    with pytest.raises(ValueError, match=r"no_progress_remind_rounds 应为整数数组"):
        load(path)


def test_error_message_names_the_file(tmp_path: Path) -> None:
    """测试：报错里带文件名 —— 同时写了好几份配置时，得知道是哪一份错了。"""
    path = write_config(tmp_path, with_line("max_super_steps = 1400", "max_super_steps = 0"))

    # max_super_steps = 0 本身合法，这里只借它确认错误信息带文件名的那条路径
    assert load(path).review.max_super_steps == 0
    bad = write_config(
        tmp_path,
        with_line("max_super_steps = 1400", "max_super_steps = 0").replace(
            "max_super_steps = 0", "max_super_step = 0"
        ),
    )
    with pytest.raises(ValueError, match=r"config\.toml"):
        load(bad)


# ---------- 有取值范围的键 ----------


def test_concurrency_default_is_serial() -> None:
    """并发度默认 1（串行）—— 默认路径上不该有线程。"""
    assert settings.review.concurrency == 1


@pytest.mark.parametrize("value", ["0", "-1", "9", "16"])
def test_concurrency_out_of_range_is_a_hard_error(tmp_path: Path, value: str) -> None:
    """并发度越界**硬报错**，不静默夹到边界。

    夹一下看起来更「友好」，但那正是本项目最不能忍的那类：写了 16、跑出来 8，
    两者都不报错 —— 而「16 并发」与「8 并发」对 provider 限流是两回事。
    """
    path = write_config(tmp_path, with_line("concurrency = 1", f"concurrency = {value}"))

    with pytest.raises(ValueError, match=r"concurrency 必须在 1\.\.8 之间"):
        load(path)


def test_concurrency_accepts_the_boundaries(tmp_path: Path) -> None:
    """边界值本身是合法的（1 与 8 都收）。"""
    for value in (1, 8):
        path = write_config(tmp_path, with_line("concurrency = 1", f"concurrency = {value}"))
        assert load(path).review.concurrency == value


def test_reasoning_effort_default_is_low() -> None:
    """默认档位是 low —— 显式压下来，不让 API 的默认值决定我们花多少钱。"""
    assert settings.llm.reasoning_effort == "low"


@pytest.mark.parametrize("value", ["ultra", "LOW", "off", ""])
def test_reasoning_effort_out_of_set_is_a_hard_error(tmp_path: Path, value: str) -> None:
    """取值不在 API 认的集合里 → **硬报错**，不静默按默认档跑。

    这条是实测出来的：往 API 传一个它不认的档位（`ultra`）**不报错**，请求照发，
    只是安静地按默认档执行。而「跑的是哪个档」直接决定花多少钱 ——
    配错了看不见，正是本项目最不能忍的那类失效。
    """
    path = write_config(
        tmp_path, with_line('reasoning_effort = "low"', f'reasoning_effort = "{value}"')
    )

    with pytest.raises(ValueError, match=r"reasoning_effort 必须是"):
        load(path)


@pytest.mark.parametrize("value", ["none", "high", "max", "minimal"])
def test_reasoning_effort_accepts_every_documented_value(tmp_path: Path, value: str) -> None:
    """官方定义的四个值 + 兼容别名都要能过 —— 校验不比 API 自己更窄。"""
    path = write_config(
        tmp_path, with_line('reasoning_effort = "low"', f'reasoning_effort = "{value}"')
    )

    assert load(path).llm.reasoning_effort == value
