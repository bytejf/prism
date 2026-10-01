"""LLM 客户端的单元测试。

不真实调用 DeepSeek（慢、要钱、CI 不稳定），改为替换 ChatOpenAI 后
断言「配置是否正确下传」。
真实调用由 tests/e2e/ 的 smoke test 覆盖。

调用与工具绑定已经交给框架（`create_agent`），这里不再测 invoke。
"""

from typing import Any, cast

import pytest

from prism.core.llm.client import LLMClient
from prism.models.llm import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL, LLMConfig


class FakeChatModel:
    """假的 ChatModel：只记录构造参数。"""

    def __init__(self, **kwargs: Any) -> None:
        """记录构造参数。

        Args:
            kwargs: ChatOpenAI 的构造参数
        """
        self.kwargs = kwargs


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> list[FakeChatModel]:
    """替换 ChatOpenAI，返回捕获到的假模型实例列表。

    Returns:
        列表，第 0 项是 LLMClient 构造出的假模型
    """
    created: list[FakeChatModel] = []

    def factory(**kwargs: Any) -> FakeChatModel:
        fake = FakeChatModel(**kwargs)
        created.append(fake)
        return fake

    monkeypatch.setattr("prism.core.llm.client.ChatOpenAI", factory)
    return created


def make_config(**overrides: Any) -> LLMConfig:
    """构造测试用 LLMConfig。

    Args:
        overrides: 需要覆盖的字段

    Returns:
        LLMConfig 对象
    """
    return LLMConfig(api_key="test-key", **overrides)


def test_config_is_passed_to_chatopenai(fake_model: list[FakeChatModel]) -> None:
    """测试配置正确下传给 ChatOpenAI。"""
    LLMClient(make_config(temperature=0.3, max_retries=5, timeout=30))
    kwargs = fake_model[0].kwargs

    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["base_url"] == DEFAULT_BASE_URL
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_retries"] == 5
    assert kwargs["timeout"] == 30
    assert kwargs["api_key"].get_secret_value() == "test-key"


def test_model_property_exposes_configured_chatopenai(fake_model: list[FakeChatModel]) -> None:
    """测试 model 属性把构造好的 ChatModel 交出去。

    调用与 bind_tools 都由框架负责，这里只保证「拿得到配好的模型」。
    """
    client = LLMClient(make_config())
    # 假模型是 monkeypatch 塞进去的普通类，与 BaseChatModel 无继承关系，
    # cast 成 object 只为让 mypy 的 identity 重叠检查放行
    assert cast(object, client.model) is fake_model[0]


def test_empty_api_key_is_rejected() -> None:
    """测试缺少 api_key 时构造即失败。"""
    with pytest.raises(ValueError, match="api_key 为空"):
        LLMClient(LLMConfig(api_key=""))


def test_from_env_reads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试从环境变量读取 api_key 并使用默认 model / base_url。"""
    monkeypatch.setenv(API_KEY_ENV, "  env-key  ")
    config = LLMConfig.from_env()

    assert config.api_key == "env-key"
    assert config.model == DEFAULT_MODEL
    assert config.base_url == DEFAULT_BASE_URL
    assert config.temperature == 0.0
    assert config.max_retries == 3
    assert config.timeout == 60


def test_from_env_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试未设置环境变量时抛 ValueError。"""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=API_KEY_ENV):
        LLMConfig.from_env()
