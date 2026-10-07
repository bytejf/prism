"""配置读取。

**默认值只有一份，在 `default.toml` 里**。代码里不再写第二份 —— 想改某个值，
改那个 toml 即可，改完立刻全局生效。

用法：

    from prism.config.settings import settings

    max_rounds = max(settings.review.min_rounds, n * settings.review.rounds_per_hunk)

## 优先级

    环境变量 > default.toml

唯一走环境变量的是 ``DEEPSEEK_API_KEY``（密钥不进文件）。它**不在导入期校验**：
缺了只给空串，报错留给 `LLMClient.__init__` —— 否则没配密钥时连
``prism --help`` / ``prism diff`` 都会崩，而这两个命令根本不需要密钥。

## 为什么不用 pydantic / 配置框架

要读的是一份**自己随包发布**的 toml，结构固定、不到 20 个键，而且必须报错时指出
具体是哪个键。`tomllib` 是标准库，加上下面那点键记账就够，不值得为此多一个依赖或
一层框架。

## 为什么未知键要硬报错

`max_lines = "500"` 这种写法 tomllib 照收不误，字符串会一路混到工具里才炸，
而那时离原因已经很远了。本项目对「静默失效」的容忍度是零
（见 CLAUDE.md「诊断经验」），所以这里在**读的那一刻**就拦下来，
并报出文件名与键路径。
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from prism.models.compression import CompressionConfig
from prism.models.cost import CostConfig
from prism.models.llm import API_KEY_ENV, LLMConfig

# 配置本体。与 `__file__` 同级 —— 走 editable 安装时是 `src/prism/config/`，
# 打进 wheel 时也在包里，两种情况都读得到
PATH = Path(__file__).parent / "default.toml"


@dataclass(frozen=True)
class ReviewSettings:
    """审查循环的轮次与节奏。

    Attributes:
        min_rounds: 单组最少给的轮次
        rounds_per_hunk: 每个 hunk 追加的轮次
        no_progress_remind_rounds: 连续无产出到这些轮次时注入提醒
        no_progress_exit_rounds: 连续无产出到该轮次强制退出
        max_rejections: end_done 被拒绝的上限，超过就接受并带缺口退出
        max_group_tokens: 单个审查组的提示词预算（估算 token），分组按它装箱
        max_group_cost: 单组花费上限（元）
        max_group_seconds: 单组时长上限（秒）
        max_review_cost: 整条 review 的花费上限（元）
        max_review_seconds: 整条 review 的时长上限（秒）
    """

    min_rounds: int
    rounds_per_hunk: int
    no_progress_remind_rounds: tuple[int, ...]
    no_progress_exit_rounds: int
    max_rejections: int
    max_group_tokens: int
    max_group_cost: float
    max_group_seconds: float
    max_review_cost: float
    max_review_seconds: float


@dataclass(frozen=True)
class FileReadSettings:
    """file_read 工具的限制。

    Attributes:
        max_lines: 单次最多读取的行数
    """

    max_lines: int


@dataclass(frozen=True)
class CodeSearchSettings:
    """code_search 工具的限制。

    Attributes:
        max_results: 单次最多返回的匹配条数
        timeout_seconds: git grep 子进程超时
    """

    max_results: int
    timeout_seconds: int


@dataclass(frozen=True)
class ToolsSettings:
    """工具的限制。

    Attributes:
        file_read: file_read 的限制
        code_search: code_search 的限制
    """

    file_read: FileReadSettings
    code_search: CodeSearchSettings


@dataclass(frozen=True)
class StaticSettings:
    """静态分析的超时。

    Attributes:
        timeout_seconds: golangci-lint / git 子进程超时
    """

    timeout_seconds: int


@dataclass(frozen=True)
class LogSettings:
    """日志配置。

    Attributes:
        level: prism 命名空间的日志级别；环境变量 PRISM_LOG_LEVEL 优先于它
    """

    level: str


@dataclass(frozen=True)
class Settings:
    """全部配置。

    这里直接复用 `models/` 里已有的两个配置类型，不再造平行的 `LLMSettings` ——
    一个概念两个类型，迟早要在它们之间写转换。

    Attributes:
        llm: 模型、端点与窗口
        compression: 压缩阈值。`context_window` 来自 `[llm]`，不重复写在 toml 里
        cost: 每百万 token 的价格
        review: 审查循环的轮次与节奏
        static: 静态分析的超时
        tools: 工具限制
        log: 日志
    """

    llm: LLMConfig
    compression: CompressionConfig
    cost: CostConfig
    review: ReviewSettings
    static: StaticSettings
    tools: ToolsSettings
    log: LogSettings


# 段路径 → (该段的数据模型, **故意不出现在 toml 里**的字段)
#
# 期望的键由**数据模型的字段**推出来，不另写一份键名清单 —— 那又会是一处会漂移
# 的东西。减掉的三个字段各有理由：
#
#   llm.api_key               只从环境变量读，密钥不落盘
#   llm.temperature           思考模式下被 API 忽略，不做成旋钮
#   compression.context_window 模型能力，只在 [llm] 写一次
SCHEMA: dict[str, tuple[type, frozenset[str]]] = {
    "llm": (LLMConfig, frozenset({"api_key", "temperature"})),
    "compression": (CompressionConfig, frozenset({"context_window"})),
    "cost": (CostConfig, frozenset()),
    "review": (ReviewSettings, frozenset()),
    "static": (StaticSettings, frozenset()),
    "log": (LogSettings, frozenset()),
    "tools.file_read": (FileReadSettings, frozenset()),
    "tools.code_search": (CodeSearchSettings, frozenset()),
}

# 只当容器的段：它的键是子段名，不是配置项
CONTAINERS: dict[str, frozenset[str]] = {"tools": frozenset({"file_read", "code_search"})}


def _expected_keys(path: str) -> frozenset[str]:
    """某一层允许出现哪些键。

    Args:
        path: 点分键路径，根为空串

    Returns:
        允许的键集合（可能为空 —— 那个段本来就不该有键）
    """
    if not path:
        return frozenset(name.split(".")[0] for name in (*SCHEMA, *CONTAINERS))
    if path in CONTAINERS:
        return CONTAINERS[path]
    model, excluded = SCHEMA[path]
    return frozenset(field.name for field in fields(model)) - excluded


class _Section:
    """一层 toml 表。

    **进这一层就先查未知键**，再逐个读值。顺序不能反：把 ``reserve_tokens``
    写成 ``reserve_token`` 时，先读值会先撞上「缺少 reserve_tokens」——
    那是个指向错误方向的提示（键并没有少，是多了个拼错的）。
    先查未知键，报出来的就是 ``reserve_token`` 本身，一眼看出是拼写问题。
    """

    def __init__(self, data: Mapping[str, Any], path: str, source: Path) -> None:
        """初始化并校验键名。

        Args:
            data: 本层的原始字典
            path: 点分键路径，用于报错。根为空串，往下是 ``compression`` /
                ``tools.file_read`` 这种（跟 toml 的段名写法一致）
            source: 配置文件路径，用于报错

        Raises:
            ValueError: 存在未知键
        """
        self._data = data
        self._path = path
        self._source = source

        expected = _expected_keys(path)
        unknown = sorted(set(data) - expected)
        if unknown:
            where = f"[{path}] " if path else ""
            raise ValueError(f"{source.name}: {where}存在未知配置项：{', '.join(unknown)}")

    def _fail(self, message: str) -> ValueError:
        """构造带文件名与键路径的错误。

        Args:
            message: 错误描述

        Returns:
            填充好上下文的 ValueError
        """
        where = f"[{self._path}] " if self._path else ""
        return ValueError(f"{self._source.name}: {where}{message}")

    def _raw(self, key: str) -> Any:
        """取出某个键的原始值。

        未知键在 `__init__` 就挡掉了，所以走到这里还缺，只可能是键被删了。

        Args:
            key: 键名

        Returns:
            原始值

        Raises:
            ValueError: 该键缺失
        """
        if key not in self._data:
            raise self._fail(f"缺少配置项 {key}")
        return self._data[key]

    def text(self, key: str) -> str:
        """读一个字符串。

        Args:
            key: 键名

        Returns:
            字符串值

        Raises:
            ValueError: 缺失或类型不符
        """
        value = self._raw(key)
        if not isinstance(value, str):
            raise self._fail(f"{key} 应为字符串，实为 {type(value).__name__}")
        return value

    def count(self, key: str) -> int:
        """读一个整数。

        `bool` 是 `int` 的子类，所以 `true` 要单独挡掉 —— 否则
        `max_lines = true` 会静默变成 1。

        Args:
            key: 键名

        Returns:
            整数值

        Raises:
            ValueError: 缺失或类型不符
        """
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._fail(f"{key} 应为整数，实为 {type(value).__name__}")
        return value

    def number(self, key: str) -> float:
        """读一个数（整数或小数都收，统一成 float）。

        预算与价格这类值天然是小数（2.0 元、0.5 元/百万）。**不接 `bool`** ——
        `true` 会静默变成 1.0，跟 `count` 那边是同一个坑。

        Args:
            key: 键名

        Returns:
            浮点值

        Raises:
            ValueError: 缺失或类型不符
        """
        value = self._raw(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._fail(f"{key} 应为数值，实为 {type(value).__name__}")
        return float(value)

    def numbers(self, key: str) -> tuple[int, ...]:
        """读一个整数数组。

        Args:
            key: 键名

        Returns:
            整数元组

        Raises:
            ValueError: 缺失或类型不符
        """
        value = self._raw(key)
        if not isinstance(value, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            raise self._fail(f"{key} 应为整数数组")
        return tuple(value)

    def section(self, key: str) -> "_Section":
        """进入下一层表。

        Args:
            key: 键名

        Returns:
            子 section

        Raises:
            ValueError: 缺失或不是表
        """
        value = self._raw(key)
        if not isinstance(value, dict):
            raise self._fail(f"{key} 应为配置段")
        return _Section(value, f"{self._path}.{key}".lstrip("."), self._source)


def load(path: Path = PATH) -> Settings:
    """读取配置文件并构造设置。

    独立成函数而不是只在模块级跑一次，是为了给测试一个接缝：传别的路径、
    或配合 monkeypatch 改环境变量后重新读。

    Args:
        path: 配置文件路径，默认包内的 `default.toml`

    Returns:
        Settings

    Raises:
        ValueError: 配置项缺失、类型不符或存在未知项
    """
    root = _Section(tomllib.loads(path.read_text(encoding="utf-8")), "", path)

    llm = root.section("llm")
    # api_key 只从环境变量来，不读文件；strip 掉粘贴时常见的首尾空白
    api_key = os.getenv(API_KEY_ENV, "").strip()
    llm_config = LLMConfig(
        model=llm.text("model"),
        base_url=llm.text("base_url"),
        api_key=api_key,
        reasoning_effort=llm.text("reasoning_effort"),
        max_retries=llm.count("max_retries"),
        timeout=llm.count("timeout"),
        context_window=llm.count("context_window"),
    )

    compression = root.section("compression")
    compression_config = CompressionConfig(
        # 窗口是「模型的能力」，只在 [llm] 写一次
        context_window=llm_config.context_window,
        reserve_tokens=compression.count("reserve_tokens"),
        max_compressions=compression.count("max_compressions"),
        max_consecutive_failures=compression.count("max_consecutive_failures"),
    )

    review = root.section("review")
    review_settings = ReviewSettings(
        min_rounds=review.count("min_rounds"),
        rounds_per_hunk=review.count("rounds_per_hunk"),
        no_progress_remind_rounds=review.numbers("no_progress_remind_rounds"),
        no_progress_exit_rounds=review.count("no_progress_exit_rounds"),
        max_rejections=review.count("max_rejections"),
        max_group_tokens=review.count("max_group_tokens"),
        max_group_cost=review.number("max_group_cost"),
        max_group_seconds=review.number("max_group_seconds"),
        max_review_cost=review.number("max_review_cost"),
        max_review_seconds=review.number("max_review_seconds"),
    )

    cost = root.section("cost")
    cost_config = CostConfig(
        cache_hit=cost.number("cache_hit"),
        cache_miss=cost.number("cache_miss"),
        output=cost.number("output"),
    )

    static = root.section("static")
    static_settings = StaticSettings(timeout_seconds=static.count("timeout_seconds"))

    tools = root.section("tools")
    file_read = tools.section("file_read")
    file_read_settings = FileReadSettings(max_lines=file_read.count("max_lines"))
    code_search = tools.section("code_search")
    code_search_settings = CodeSearchSettings(
        max_results=code_search.count("max_results"),
        timeout_seconds=code_search.count("timeout_seconds"),
    )

    log = root.section("log")
    log_settings = LogSettings(level=log.text("level"))

    return Settings(
        llm=llm_config,
        compression=compression_config,
        cost=cost_config,
        review=review_settings,
        static=static_settings,
        tools=ToolsSettings(file_read=file_read_settings, code_search=code_search_settings),
        log=log_settings,
    )


# 全局唯一实例。读的是一份随包发布的固定文件，导入期出错即配置损坏，早炸早好
settings = load()
