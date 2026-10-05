"""评论反思的单元测试。

守三件事，每一件错了都不会报错、只会静默地改变输出：

- **只删不增** —— 保留下来的评论内容与顺序原样不动
- **绝不删错** —— 编号对不上、理由为空、模型胡说的，一律保留
- **失败 fail-open** —— 出错时全部保留，不留半截结果

用假模型（`FakeModel`）按脚本返回工具调用，不打网络。
"""

import logging
from typing import Any, override

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from prism.core.reflection.reflect import (
    TOOL_NAME,
    reflect,
)
from prism.models.ast import HunkWithSymbol, Symbol
from prism.models.comment import (
    LOCATE_FAILED,
    LOCATE_HUNK,
    REASON_NO_MATCH,
    SIDE_LEFT,
    SIDE_RIGHT,
    Comment,
)
from prism.models.diff import Hunk
from prism.models.grouping import Group

REFLECT_LOGGER = "prism.core.reflection.reflect"

SYMBOL = Symbol(name="F", kind="function", start_line=1, end_line=99)


class FakeModel(BaseChatModel):
    """按脚本返回预设响应的假 ChatModel。"""

    response: AIMessage
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    error: str | None = None

    @property
    @override
    def _llm_type(self) -> str:
        """模型类型标识。

        Returns:
            固定字符串
        """
        return "fake-reflection"

    @override
    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> "FakeModel":
        """记下绑定的工具并返回自己。

        Args:
            tools: 工具列表
            tool_choice: 选择策略（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            self
        """
        self.bound_tools.append(tools)
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回脚本响应。

        Args:
            messages: 本轮消息
            stop: 停止词（忽略）
            run_manager: 回调管理器（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            ChatResult

        Raises:
            ValueError: 脚本里设了 error
        """
        self.prompts.append(list(messages))
        if self.error:
            raise ValueError(self.error)
        return ChatResult(generations=[ChatGeneration(message=self.response)])


def verdict(*removals: dict[str, str], misses: list[dict[str, str]] | None = None) -> AIMessage:
    """造一条带 `report_incorrect_comments` 调用的响应。

    Args:
        *removals: 每条要删除的评论参数
        misses: 漏报信号

    Returns:
        AIMessage
    """
    args = {"removals": list(removals), "misses": misses or []}
    return AIMessage(
        content="",
        tool_calls=[{"name": TOOL_NAME, "args": args, "id": "c1", "type": "tool_call"}],
    )


def removal(
    comment_id: str, ground: str = "B", reasoning: str = "diff 第 8 行推翻了它"
) -> dict[str, str]:
    """造一条删除项。

    Args:
        comment_id: 评论编号
        ground: 依据
        reasoning: 指认

    Returns:
        参数字典
    """
    return {"comment_id": comment_id, "ground": ground, "reasoning": reasoning}


def make_comment(
    content: str,
    hunk_id: str = "H-01",
    path: str = "a.go",
    *,
    locate_level: str = LOCATE_HUNK,
    side: str | None = SIDE_RIGHT,
    start_line: int = 10,
    locate_reason: str | None = None,
) -> Comment:
    """构造测试用评论。

    Args:
        content: 评论正文
        hunk_id: hunk 编号
        path: 文件路径
        locate_level: 定位层级
        side: 行号属于哪一侧
        start_line: 起始行号
        locate_reason: 未定位原因

    Returns:
        Comment 对象
    """
    return Comment(
        path=path,
        severity="high",
        category="边界条件",
        content=content,
        existing_code="if err != nil {",
        hunk_id=hunk_id,
        locate_level=locate_level,
        side=side,
        locate_reason=locate_reason,
        start_line=start_line,
        end_line=start_line,
    )


def make_group(hunk_ids: dict[str, str]) -> Group:
    """构造测试用审查单元。

    Args:
        hunk_ids: hunk 编号 → 文件路径

    Returns:
        Group 对象
    """
    hunks = [
        HunkWithSymbol(
            part=Hunk(old_start=1, old_lines=1, new_start=1, new_lines=1, content="+\tfoo()"),
            file_path=path,
            symbol=SYMBOL,
            hunk_id=hunk_id,
            head_lines=(1, 9),
            renamed_from=None,
            deleted=False,
        )
        for hunk_id, path in hunk_ids.items()
    ]
    return Group(
        group_id="group-001", package_path=".", files=sorted(hunk_ids.values()), hunks=hunks
    )


def run(response: AIMessage, comments: list[Comment], group: Group | None = None) -> Any:
    """跑一次反思。

    Args:
        response: 假模型的响应
        comments: 待复核的评论
        group: 审查单元

    Returns:
        ReflectionOutcome
    """
    model = FakeModel(response=response)
    return reflect(model, group or make_group({"H-01": "a.go"}), comments)


# ---------- 只删不增 ----------


def test_approve_all_keeps_everything() -> None:
    """测试：removals 为空 = 全部保留。"""
    comments = [make_comment("第一条"), make_comment("第二条")]

    outcome = run(verdict(), comments)

    assert [item.content for item in outcome.kept] == ["第一条", "第二条"]
    assert outcome.removed == []
    assert outcome.failure is None


def test_removes_only_the_named_comment() -> None:
    """测试：只去掉被点名的那些，其余原样保留且顺序不变。"""
    comments = [make_comment("第一条"), make_comment("第二条"), make_comment("第三条")]

    outcome = run(verdict(removal("C-02", "A", "diff 里没有它说的那段")), comments)

    assert [item.content for item in outcome.kept] == ["第一条", "第三条"]


def test_removal_records_ground_and_reasoning() -> None:
    """测试：删除依据与指认要留档 —— 删评论是唯一有破坏性的动作。"""
    outcome = run(
        verdict(removal("C-01", "B", "第 8 行就是它说缺失的那个检查")), [make_comment("x")]
    )

    assert len(outcome.removed) == 1
    removed = outcome.removed[0]
    assert (removed.comment_id, removed.ground, removed.path) == ("C-01", "B", "a.go")
    assert removed.reasoning == "第 8 行就是它说缺失的那个检查"


def test_does_not_mutate_the_input_list() -> None:
    """测试：入参列表不被修改。"""
    comments = [make_comment("第一条"), make_comment("第二条")]

    run(verdict(removal("C-01")), comments)

    assert len(comments) == 2


# ---------- 绝不删错 ----------


def test_unknown_comment_id_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    """测试：编号对不上就忽略，绝不猜。

    猜错一次就是删掉一条**无关**的评论 —— 比不删严重得多。
    """
    comments = [make_comment("第一条"), make_comment("第二条")]

    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(verdict(removal("C-99")), comments)

    assert len(outcome.kept) == 2
    assert outcome.removed == []
    assert "忽略未知编号" in caplog.text


def test_blank_reasoning_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    """测试：理由为空的删除被拒绝，评论保留。

    提示词要求「指不出具体哪一处 diff 就保留」，空理由是这条规则在 schema
    层面唯一可机器判定的形态 —— 所以在这里兜一层。
    """
    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(verdict(removal("C-01", "B", "   ")), [make_comment("x")])

    assert len(outcome.kept) == 1
    assert outcome.removed == []
    assert "拒绝无依据的删除" in caplog.text


def test_duplicate_ids_are_deduped() -> None:
    """测试：同一编号报两次只删一次，且保留列表不能少算。"""
    comments = [make_comment("第一条"), make_comment("第二条")]

    outcome = run(verdict(removal("C-01"), removal("C-01", "A", "另一条理由")), comments)

    assert [item.comment_id for item in outcome.removed] == ["C-01"]
    assert [item.content for item in outcome.kept] == ["第二条"]


def test_invalid_ground_fails_open() -> None:
    """测试：ground 不是 A/B 时整份结论作废、全部保留（fail-open）。

    宁可一条都不删，也不要按半份读不懂的结论删。
    """
    outcome = run(verdict(removal("C-01", "C")), [make_comment("第一条"), make_comment("第二条")])

    assert len(outcome.kept) == 2
    assert outcome.removed == []
    assert outcome.failure is not None


# ---------- 失败 fail-open ----------


def test_no_tool_call_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    """测试：模型没调工具（回了纯文本）→ 全部保留 + 告警。"""
    comments = [make_comment("第一条"), make_comment("第二条")]

    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(AIMessage(content="我觉得都还行"), comments)

    assert len(outcome.kept) == 2
    assert outcome.failure is not None
    assert "已保留全部 2 条" in caplog.text


def test_model_error_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    """测试：调用报错（超时这类）→ 全部保留 + 告警。"""
    model = FakeModel(response=verdict(), error="connection reset")
    comments = [make_comment("第一条")]

    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = reflect(model, make_group({"H-01": "a.go"}), comments)

    assert [item.content for item in outcome.kept] == ["第一条"]
    assert outcome.failure is not None and "connection reset" in outcome.failure
    assert "反思 · 失败" in caplog.text


def test_multiple_tool_calls_fail_open() -> None:
    """测试：一次响应里调了多次工具 → 结论自相矛盾，全部保留。

    取第一个是**猜**：两次调用给出的是两份不同的结论，猜错就是删错。
    """
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": TOOL_NAME,
                "args": {"removals": [removal("C-01")]},
                "id": "c1",
                "type": "tool_call",
            },
            {"name": TOOL_NAME, "args": {"removals": []}, "id": "c2", "type": "tool_call"},
        ],
    )

    outcome = run(response, [make_comment("第一条")])

    assert len(outcome.kept) == 1
    assert outcome.removed == []
    assert outcome.failure is not None


def test_wrong_tool_name_fails_open() -> None:
    """测试：调了个不认识的工具 → 全部保留。"""
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "something_else", "args": {"removals": []}, "id": "c1", "type": "tool_call"}
        ],
    )

    outcome = run(response, [make_comment("第一条")])

    assert len(outcome.kept) == 1
    assert outcome.failure is not None


def test_no_comments_means_no_call() -> None:
    """测试：没有评论时不调模型。"""
    model = FakeModel(response=verdict())

    outcome = reflect(model, make_group({"H-01": "a.go"}), [])

    assert outcome.kept == []
    assert model.prompts == []


# ---------- 漏报信号 ----------


def test_miss_gets_path_from_hunk_id() -> None:
    """测试：漏报的路径由 hunk_id 查出，不让模型自己写。"""
    group = make_group({"H-01": "pkg/a.go", "H-02": "pkg/b.go"})

    outcome = run(
        verdict(misses=[{"hunk_id": "H-02", "description": "新加的解锁路径没人提"}]),
        [make_comment("x")],
        group,
    )

    assert [(item.hunk_id, item.path) for item in outcome.misses] == [("H-02", "pkg/b.go")]
    assert outcome.misses[0].description == "新加的解锁路径没人提"


def test_miss_with_unknown_hunk_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """测试：hunk 编号不存在的漏报直接丢并告警。"""
    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(
            verdict(misses=[{"hunk_id": "H-99", "description": "不知哪来的"}]),
            [make_comment("x")],
        )

    assert outcome.misses == []
    assert "忽略未知编号的漏报" in caplog.text


def test_miss_with_blank_description_is_dropped() -> None:
    """测试：描述为空的漏报是噪声，丢掉。"""
    outcome = run(verdict(misses=[{"hunk_id": "H-01", "description": "  "}]), [make_comment("x")])

    assert outcome.misses == []


def test_misses_do_not_become_comments() -> None:
    """测试：漏报**不进**评论列表 —— 只删不增这条约束不因它松动。"""
    comments = [make_comment("第一条")]

    outcome = run(verdict(misses=[{"hunk_id": "H-01", "description": "疑似漏报"}]), comments)

    assert [item.content for item in outcome.kept] == ["第一条"]


# ---------- 送给模型的材料 ----------


def test_prompt_carries_diff_and_locate_metadata() -> None:
    """测试：提示词里同时有 diff、评论正文与定位元数据。

    定位元数据是「先定位后反思」换来的那部分信号，漏掉它等于白做定位。
    """
    model = FakeModel(response=verdict())
    comments = [
        make_comment("删除侧的那条", locate_level=LOCATE_HUNK, side=SIDE_LEFT, start_line=7),
        make_comment(
            "未定位的那条",
            locate_level=LOCATE_FAILED,
            side=None,
            start_line=0,
            locate_reason=REASON_NO_MATCH,
        ),
    ]

    reflect(model, make_group({"H-01": "pkg/a.go"}), comments)

    prompt = str(model.prompts[0][1].content)
    assert "<diff>" in prompt and 'id="H-01"' in prompt
    assert 'id="C-01"' in prompt and 'id="C-02"' in prompt
    assert 'locate="hunk" side="LEFT" lines="7"' in prompt
    assert 'locate="failed:no_match"' in prompt
    # 未定位的评论不给位置属性 —— 宁可少一个属性，也不给一个假坐标
    assert 'locate="failed:no_match" side=' not in prompt
    # 两侧职责相反，system 必须是复核那份，不是审查那份
    assert "复核者" in str(model.prompts[0][0].content)


def test_prompt_mentions_every_comment_in_order() -> None:
    """测试：评论按传入顺序编号 C-01、C-02、C-03。"""
    model = FakeModel(response=verdict())
    comments = [make_comment("一"), make_comment("二"), make_comment("三")]

    reflect(model, make_group({"H-01": "a.go"}), comments)

    prompt = str(model.prompts[0][1].content)
    assert prompt.index('id="C-01"') < prompt.index('id="C-02"') < prompt.index('id="C-03"')


def test_only_the_verdict_tool_is_bound() -> None:
    """测试：只绑一个工具，名字与参数形状固定。

    绑两个工具（OCR 那样，多一个 `approve_all_comments`）实测行为一样，
    所以这里守住「就一个」这个更小的形状 —— 见模块 docstring 的实测记录。
    """
    model = FakeModel(response=verdict())

    reflect(model, make_group({"H-01": "a.go"}), [make_comment("x")])

    tools = model.bound_tools[0]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == TOOL_NAME
    assert set(tools[0]["function"]["parameters"]["properties"]) == {"removals", "misses"}


# ---------- 日志 ----------


def test_completion_is_logged_with_counts_and_grounds(caplog: pytest.LogCaptureFixture) -> None:
    """测试：完成日志带删除数、保留数、删除率，以及删除依据（审计用）。"""
    comments = [make_comment("第一条"), make_comment("第二条"), make_comment("第三条")]

    with caplog.at_level(logging.INFO, logger=REFLECT_LOGGER):
        run(verdict(removal("C-02", "A", "diff 里没有它说的那段")), comments)

    assert "反思 · 开始 · group-001 · 候选 3 条" in caplog.text
    assert "反思 · 完成" in caplog.text
    assert "删除 1 / 保留 2（删除率 33%）" in caplog.text
    assert "Ground A：diff 里没有它说的那段" in caplog.text


def test_request_is_logged_in_full(caplog: pytest.LogCaptureFixture) -> None:
    """测试：请求全文进日志。

    复核走的是裸模型调用，不经过 `ReviewTrace`，不打就完全看不见 ——
    与压缩请求同样的理由。
    """
    with caplog.at_level(logging.INFO, logger=REFLECT_LOGGER):
        run(verdict(), [make_comment("被复核的评论正文")])

    assert "反思 · 请求 · system" in caplog.text
    assert "反思 · 请求 · user" in caplog.text
    assert "被复核的评论正文" in caplog.text


def test_response_is_logged_with_tool_calls(caplog: pytest.LogCaptureFixture) -> None:
    """测试：响应（含工具调用）也进日志，且**先打再解析**。"""
    with caplog.at_level(logging.INFO, logger=REFLECT_LOGGER):
        run(verdict(removal("C-01")), [make_comment("x")])

    assert "反思 · 响应 · 耗时" in caplog.text
    assert f"{TOOL_NAME}(" in caplog.text


def test_no_tool_call_logs_the_raw_content(caplog: pytest.LogCaptureFixture) -> None:
    """测试：没调工具时，模型的原话必须进日志。

    这条是补出来的 —— 定标第一次跑就撞上「没调工具」，而当时的日志只说
    「期望 1 次调用，实际 0 次」，看不出模型到底是写了段散文还是空手而归。
    两者的修法完全不同，所以响应正文必须打出来再解析。
    """
    with caplog.at_level(logging.INFO, logger=REFLECT_LOGGER):
        run(AIMessage(content="这几条评论我都认为成立，无需删除。"), [make_comment("x")])

    assert "这几条评论我都认为成立" in caplog.text
