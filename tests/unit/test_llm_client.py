"""LLM 客户端的单元测试。

不真实调用 DeepSeek（慢、要钱、CI 不稳定），改为替换 ChatOpenAI 后
断言「配置是否正确下传」与「响应/异常如何处理」。
真实调用由 tests/e2e/ 的 smoke test 覆盖。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from prism.core.llm.client import LLMClient, LLMError, message_text
from prism.models.llm import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL, LLMConfig


class FakeChatModel:
    """假的 ChatModel：记录构造参数、绑定过的工具与收到的消息。"""

    def __init__(self, **kwargs: Any) -> None:
        """记录构造参数。

        Args:
            kwargs: ChatOpenAI 的构造参数
        """
        self.kwargs = kwargs
        self.bound_tools: list[BaseTool] = []
        self.received: list[list[Any]] = []
        self.response: AIMessage | Exception = AIMessage(content="ok")

    def bind_tools(self, tools: list[BaseTool]) -> "FakeChatModel":
        """记录绑定的工具。

        Args:
            tools: 待绑定的工具

        Returns:
            自身，模拟 Runnable
        """
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        """返回预设响应或抛出预设异常。

        Args:
            messages: 消息列表

        Returns:
            预设的 AIMessage

        Raises:
            Exception: 预设的异常
        """
        self.received.append(list(messages))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


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


@tool
def echo(text: str) -> str:
    """回显输入。

    Args:
        text: 待回显的文本

    Returns:
        原样返回
    """
    return text


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


def test_invoke_without_tools_does_not_bind(fake_model: list[FakeChatModel]) -> None:
    """测试不传工具时不绑定工具。"""
    client = LLMClient(make_config())
    response = client.invoke([HumanMessage(content="你好")])

    assert response.content == "ok"
    assert fake_model[0].bound_tools == []
    assert len(fake_model[0].received) == 1


def test_invoke_with_tools_binds_them(fake_model: list[FakeChatModel]) -> None:
    """测试传工具时绑定到模型。"""
    client = LLMClient(make_config())
    fake_model[0].response = AIMessage(
        content="",
        tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "call-1", "type": "tool_call"}],
    )
    response = client.invoke([HumanMessage(content="回显 hi")], tools=[echo])

    assert [t.name for t in fake_model[0].bound_tools] == ["echo"]
    assert response.tool_calls[0]["name"] == "echo"


def test_invoke_wraps_failure_in_llm_error(fake_model: list[FakeChatModel]) -> None:
    """测试调用失败时抛 LLMError 并保留原因。"""
    client = LLMClient(make_config())
    fake_model[0].response = RuntimeError("连接超时")

    with pytest.raises(LLMError, match="连接超时"):
        client.invoke([HumanMessage(content="你好")])


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


def test_message_text_from_string() -> None:
    """测试字符串内容。"""
    assert message_text(AIMessage(content="hello")) == "hello"


def test_message_text_from_content_blocks() -> None:
    """测试内容块列表（含文本块与非文本块）。"""
    message = AIMessage(
        content=[
            {"type": "text", "text": "第一段"},
            {"type": "tool_use", "id": "x", "name": "echo", "input": {}},
            {"type": "text", "text": "第二段"},
        ]
    )
    assert message_text(message) == "第一段第二段"


def test_message_text_empty() -> None:
    """测试无文本内容。"""
    assert message_text(AIMessage(content=[])) == ""
