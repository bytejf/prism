"""LLM 客户端的单元测试。

不真实调用 DeepSeek（慢、要钱、CI 不稳定），改为替换 `ReasoningEchoChatDeepSeek`
后断言「配置是否正确下传」。
真实调用由 tests/e2e/ 的 smoke test 覆盖。

调用与工具绑定已经交给框架（`create_agent`），这里不再测 invoke。
"""

from dataclasses import replace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import SecretStr

from prism.config.settings import load, settings
from prism.core.llm.client import LLMClient, ReasoningEchoChatDeepSeek
from prism.models.llm import API_KEY_ENV, LLMConfig


class FakeChatModel:
    """假的 ChatModel：只记录构造参数。"""

    def __init__(self, **kwargs: Any) -> None:
        """记录构造参数。

        Args:
            kwargs: ChatDeepSeek 的构造参数
        """
        self.kwargs = kwargs


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> list[FakeChatModel]:
    """替换 ChatDeepSeek，返回捕获到的假模型实例列表。

    Returns:
        列表，第 0 项是 LLMClient 构造出的假模型
    """
    created: list[FakeChatModel] = []

    def factory(**kwargs: Any) -> FakeChatModel:
        fake = FakeChatModel(**kwargs)
        created.append(fake)
        return fake

    monkeypatch.setattr("prism.core.llm.client.ReasoningEchoChatDeepSeek", factory)
    return created


def make_config(**overrides: Any) -> LLMConfig:
    """构造测试用 LLMConfig。

    以配置里的那份为底再覆盖 —— 默认值只有 `default.toml` 一份，测试里也不再
    抄一遍，否则配置一改这里就对不上了。

    Args:
        overrides: 需要覆盖的字段

    Returns:
        LLMConfig 对象
    """
    return replace(settings.llm, api_key="test-key", **overrides)


def test_config_is_passed_to_chatdeepseek(fake_model: list[FakeChatModel]) -> None:
    """测试配置正确下传给 ChatDeepSeek。"""
    LLMClient(make_config(temperature=0.3, max_retries=5, timeout=30, reasoning_effort="medium"))
    kwargs = fake_model[0].kwargs

    assert kwargs["model"] == settings.llm.model
    assert kwargs["base_url"] == settings.llm.base_url
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_retries"] == 5
    assert kwargs["timeout"] == 30
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["api_key"].get_secret_value() == "test-key"


def test_model_property_exposes_configured_model(fake_model: list[FakeChatModel]) -> None:
    """测试 model 属性把构造好的 ChatModel 交出去。

    调用与 bind_tools 都由框架负责，这里只保证「拿得到配好的模型」。
    """
    client = LLMClient(make_config())
    # 假模型是 monkeypatch 塞进去的普通类，与 BaseChatModel 无继承关系，
    # cast 成 object 只为让 mypy 的 identity 重叠检查放行
    assert cast(object, client.model) is fake_model[0]


def test_empty_api_key_is_rejected() -> None:
    """测试缺少 api_key 时构造即失败。

    这是**唯一**会因缺密钥而报错的地方 —— 读取配置本身不校验（见下一个用例）。
    """
    with pytest.raises(ValueError, match="api_key 为空"):
        LLMClient(replace(settings.llm, api_key=""))


def test_load_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试 api_key 只从环境变量读，且剔除首尾空白。"""
    monkeypatch.setenv(API_KEY_ENV, "  env-key  ")

    loaded = load()

    assert loaded.llm.api_key == "env-key"
    # 其余字段来自配置文件，不是代码里的第二份默认值
    assert loaded.llm.model == settings.llm.model
    assert loaded.llm.base_url == settings.llm.base_url


def test_load_tolerates_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试没有密钥时读取配置**不抛异常**。

    否则 `prism --help` / `prism diff` 这些根本不需要密钥的命令，会在导入配置
    阶段就崩掉。校验时机交给 LLMClient（上一个用例）。
    """
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    assert load().llm.api_key == ""


# ---------- 思考内容回传 ----------


def make_echo_model() -> ReasoningEchoChatDeepSeek:
    """构造一个不联网的实例（构造不校验 key，也不发请求）。

    Returns:
        ReasoningEchoChatDeepSeek 实例
    """
    return ReasoningEchoChatDeepSeek(
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        api_key=SecretStr("test-key"),
    )


def reasoning_of(payload: dict[str, Any], index: int) -> object:
    """取出请求体里第 index 条消息的 reasoning_content。

    Args:
        payload: `_get_request_payload` 的返回
        index: 消息下标

    Returns:
        reasoning_content 的值；没有则为 None
    """
    return payload["messages"][index].get("reasoning_content")


def test_reasoning_echo_injects_only_last_assistant() -> None:
    """测试只给最后一条 assistant 补思考内容，历史 assistant 不动。

    补全部历史会让 prompt 随轮次线性膨胀（每轮都重发全量历史），
    而「一条都不补」实测也能跑通，所以取最小改动。
    """
    messages = [
        HumanMessage("北京天气？"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "北京"}, "id": "c1", "type": "tool_call"}
            ],
            additional_kwargs={"reasoning_content": "第一轮的思考"},
        ),
        ToolMessage(content="北京 晴", tool_call_id="c1"),
        AIMessage(content="晴", additional_kwargs={"reasoning_content": "最后一轮的思考"}),
    ]

    payload = make_echo_model()._get_request_payload(messages)

    assert reasoning_of(payload, 0) is None
    assert reasoning_of(payload, 1) is None, "历史 assistant 不该被补"
    assert reasoning_of(payload, 2) is None
    assert reasoning_of(payload, 3) == "最后一轮的思考"


def test_reasoning_echo_leaves_plain_messages_alone() -> None:
    """测试没有思考内容的对话不会被改动。"""
    messages = [HumanMessage("你好"), AIMessage("你好，有什么可以帮你？")]

    payload = make_echo_model()._get_request_payload(messages)

    assert reasoning_of(payload, 1) is None
    assert payload["messages"][1]["role"] == "assistant"


def test_reasoning_echo_keeps_message_count() -> None:
    """测试注入不改变消息条数，且注入的是原消息自己的思考内容。

    条数是 `client.py` 里那条「对不上就整体跳过」检查的前提 ——
    这里守住它。条数真对不上的分支没法用公开输入构造，只能靠代码审查覆盖。
    """
    messages = [HumanMessage("你好"), AIMessage("嗯", additional_kwargs={"reasoning_content": "x"})]

    payload = make_echo_model()._get_request_payload(messages)

    assert len(payload["messages"]) == len(messages)
    assert reasoning_of(payload, 1) == "x"
