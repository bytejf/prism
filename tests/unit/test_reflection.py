"""评论反思的单元测试。

守三件事，每一件错了都不会报错、只会静默地改变输出：

- **只删不增** —— 保留下来的评论内容与顺序原样不动
- **绝不删错** —— 编号对不上、理由为空的，一律退回给模型改，改不好就全留
- **失败 fail-open** —— 出错时全部保留，不留半截结果

用假模型按脚本返回工具调用，**工具本身是真的**（走 `create_agent`，所以
`ToolRuntime` 注入、`Command` 回写、中间件日志都真跑），只是不打网络。
"""

import logging
from typing import Any, override

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from prism.core.reflection.reflect import ReflectionOutcome, reflect
from prism.core.reflection.tools import TOOL_NAME, report_incorrect_comments
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
TRACE_LOGGER = "prism.observability.llm_trace"

SYMBOL = Symbol(name="F", kind="function", start_line=1, end_line=99)


class ScriptedChatModel(BaseChatModel):
    """按脚本依次返回响应的假 ChatModel（写法同 `test_main_loop.py`）。"""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    tools_seen: list[list[BaseTool]] = Field(default_factory=list)
    error: str | None = None

    @property
    @override
    def _llm_type(self) -> str:
        """模型类型标识。

        Returns:
            固定字符串
        """
        return "scripted-reflection"

    @override
    def bind_tools(
        self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any
    ) -> "ScriptedChatModel":
        """记录被绑定的工具，原样返回自己。

        Args:
            tools: 工具列表
            tool_choice: 选择策略（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            self
        """
        self.tools_seen.append(list(tools))
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """返回下一个脚本响应。

        Args:
            messages: 本轮消息
            stop: 停止词（忽略）
            run_manager: 回调管理器（忽略）
            **kwargs: 其余参数（忽略）

        Returns:
            ChatResult

        Raises:
            ValueError: 脚本里设了 error（模拟网络/接口失败）
            AssertionError: 脚本用尽（顺带防住死循环）
        """
        self.calls.append(list(messages))
        if self.error:
            raise ValueError(self.error)
        if not self.responses:
            msg = "ScriptedChatModel 的响应脚本已用尽"
            raise AssertionError(msg)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def verdict(
    *removals: dict[str, str], misses: list[dict[str, str]] | None = None, call_id: str = "c1"
) -> AIMessage:
    """造一条调用 `report_incorrect_comments` 的响应。

    Args:
        *removals: 每条要删除的评论参数
        misses: 漏报信号
        call_id: 调用编号

    Returns:
        AIMessage
    """
    args = {"removals": list(removals), "misses": misses or []}
    return AIMessage(
        content="",
        tool_calls=[{"name": TOOL_NAME, "args": args, "id": call_id, "type": "tool_call"}],
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


def run(
    model: ScriptedChatModel, comments: list[Comment], group: Group | None = None
) -> ReflectionOutcome:
    """跑一次反思。

    Args:
        model: 假模型
        comments: 待复核的评论
        group: 审查单元

    Returns:
        ReflectionOutcome
    """
    return reflect(model, group or make_group({"H-01": "a.go"}), comments)


def request_text(model: ScriptedChatModel, index: int = 0) -> str:
    """把第 N 次模型调用收到的全部消息拼成一段文本（断言用）。

    Args:
        model: 假模型
        index: 第几次调用

    Returns:
        拼接后的文本
    """
    return "\n".join(str(message.content) for message in model.calls[index])


# ---------- 只删不增 ----------


def test_approve_all_keeps_everything() -> None:
    """测试：空 removals = 全部保留。"""
    comments = [make_comment("第一条"), make_comment("第二条")]
    model = ScriptedChatModel(responses=[verdict()])

    outcome = run(model, comments)

    assert [item.content for item in outcome.kept] == ["第一条", "第二条"]
    assert outcome.removed == []
    assert outcome.failure is None


def test_removes_only_the_named_comment() -> None:
    """测试：只去掉被点名的那些，其余原样保留且顺序不变。"""
    comments = [make_comment("第一条"), make_comment("第二条"), make_comment("第三条")]
    model = ScriptedChatModel(responses=[verdict(removal("C-02", "A", "diff 里没有它说的那段"))])

    outcome = run(model, comments)

    assert [item.content for item in outcome.kept] == ["第一条", "第三条"]


def test_removal_records_ground_path_and_reasoning() -> None:
    """测试：删除依据、**路径**与指认都要留档。

    路径由我们的候选表补齐（`a.go`），不是模型给的 —— 它只报了编号。
    """
    model = ScriptedChatModel(responses=[verdict(removal("C-01", "B", "第 8 行就是那个检查"))])

    outcome = run(model, [make_comment("x")])

    assert len(outcome.removed) == 1
    removed = outcome.removed[0]
    assert (removed.comment_id, removed.ground, removed.path) == ("C-01", "B", "a.go")
    assert removed.reasoning == "第 8 行就是那个检查"


def test_duplicate_ids_collapse() -> None:
    """测试：同一次调用里同一编号报两次 → 合成一条（dict 的键决定了这件事）。"""
    model = ScriptedChatModel(responses=[verdict(removal("C-01"), removal("C-01", "A", "另说"))])

    outcome = run(model, [make_comment("第一条"), make_comment("第二条")])

    assert [item.comment_id for item in outcome.removed] == ["C-01"]
    assert [item.content for item in outcome.kept] == ["第二条"]


def test_does_not_mutate_the_input_list() -> None:
    """测试：入参列表不被修改。"""
    comments = [make_comment("第一条"), make_comment("第二条")]

    run(ScriptedChatModel(responses=[verdict(removal("C-01"))]), comments)

    assert len(comments) == 2


def test_only_one_model_call_when_accepted() -> None:
    """测试：受理之后**不再多调一次模型**。

    `create_agent` 的默认退出是「模型不再调用工具」，不管它的话会再问一次
    「还有吗」—— 结论已经在手里了，那是一次纯浪费。`ReflectionCompletion`
    在 `before_model` 里直接结束，所以模型只被调了一次。
    """
    model = ScriptedChatModel(responses=[verdict(removal("C-01"))])

    run(model, [make_comment("x")])

    assert len(model.calls) == 1


# ---------- 纠错反馈：实体工具换来的能力 ----------


def test_unknown_id_is_rejected_and_fed_back() -> None:
    """测试：编号不存在 → 退回 → 模型改对后重新提交。

    **这是把工具做成实体之后拿到的东西**：问题作为工具结果回到模型手里，
    它在同一轮对话里改一版。只当输出格式的版本只能事后 fail-open。
    """
    model = ScriptedChatModel(
        responses=[verdict(removal("C-99")), verdict(removal("C-01", "B", "第 8 行"))]
    )

    outcome = run(model, [make_comment("x")])

    assert [item.comment_id for item in outcome.removed] == ["C-01"]
    # 第二次请求里能看到退回原因 —— 模型就是靠它改对的
    second = request_text(model, 1)
    assert "C-99" in second
    assert "不在本次候选清单里" in second
    # 第一次的坏结论**没有被写进状态**（整份退回）
    assert len(model.calls) == 2


def test_blank_reasoning_is_rejected_and_fed_back() -> None:
    """测试：理由为空 → 退回，并告诉它为什么。

    「指不出具体哪一处 diff 就保留」这条规则，在实体工具里成了可执行的门槛。
    """
    model = ScriptedChatModel(
        responses=[verdict(removal("C-01", "B", "   ")), verdict(removal("C-01", "B", "第 8 行"))]
    )

    outcome = run(model, [make_comment("x")])

    assert [item.comment_id for item in outcome.removed] == ["C-01"]
    assert "没给出理由" in request_text(model, 1)


def test_rejected_then_plain_text_fails_open() -> None:
    """测试：退回之后再没交结论 → 全部保留（不留半截）。

    第一次报的坏结论没有被采纳，所以「失败即全留」这条语义是干净的。
    """
    model = ScriptedChatModel(
        responses=[verdict(removal("C-99")), AIMessage(content="算了，都保留吧")]
    )

    outcome = run(model, [make_comment("第一条"), make_comment("第二条")])

    assert len(outcome.kept) == 2
    assert outcome.removed == []
    assert outcome.failure is not None


def test_miss_with_unknown_hunk_is_dropped_but_call_accepted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """测试：漏报的 hunk 编号不存在 → **丢掉那一条，其余照收**。

    漏报是提示不是评论 —— 不能因为一条 hunk 编号写错，把整份删除结论也按住。

    丢弃说明写在**工具回执**里。这一次是受理（不是退回），模型拿到结论就收工了，
    所以它看不到那段说明 —— 能证明「没被静默吞掉」的只有日志（回执正文由
    `ReviewTrace` 的 `wrap_tool_call` 打出来）。
    """
    model = ScriptedChatModel(
        responses=[
            verdict(
                removal("C-01"),
                misses=[
                    {"hunk_id": "H-99", "description": "不知哪来的"},
                    {"hunk_id": "H-01", "description": "这处改动没人提"},
                ],
            )
        ]
    )

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER):
        outcome = run(model, [make_comment("第一条"), make_comment("第二条")])

    assert [item.comment_id for item in outcome.removed] == ["C-01"]
    assert [item.hunk_id for item in outcome.misses] == ["H-01"]
    assert outcome.misses[0].path == "a.go", "路径由我们的 hunk 表补齐"
    assert "H-99" in caplog.text
    assert len(model.calls) == 1, "受理即收工，不再多调一次"


# ---------- 失败 fail-open ----------


def test_plain_text_response_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    """测试：模型压根没调工具（回了纯文本）→ 全部保留 + 告警。"""
    model = ScriptedChatModel(responses=[AIMessage(content="我觉得都还行")])
    comments = [make_comment("第一条"), make_comment("第二条")]

    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(model, comments)

    assert len(outcome.kept) == 2
    assert outcome.failure is not None
    assert "已保留全部 2 条" in caplog.text


def test_repeated_rejections_fail_open() -> None:
    """测试：一直交不出可受理的结论 → 撞上调用上限 → 全部保留。

    上限是 `REFLECTION_MAX_CALLS`（3）：首次 + 一次纠错 + 一次余量。
    """
    model = ScriptedChatModel(responses=[verdict(removal("C-99")) for _ in range(3)])

    outcome = run(model, [make_comment("第一条"), make_comment("第二条")])

    assert len(outcome.kept) == 2
    assert outcome.removed == []
    assert outcome.failure is not None
    assert len(model.calls) == 3, "不该超过调用上限"


def test_model_error_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    """测试：调用报错（超时这类）→ 全部保留 + 告警。"""
    model = ScriptedChatModel(responses=[verdict()], error="connection reset")

    with caplog.at_level(logging.WARNING, logger=REFLECT_LOGGER):
        outcome = run(model, [make_comment("第一条")])

    assert [item.content for item in outcome.kept] == ["第一条"]
    assert outcome.failure is not None and "connection reset" in outcome.failure
    assert "反思 · 失败" in caplog.text


def test_no_comments_means_no_call() -> None:
    """测试：没有评论时不调模型。"""
    model = ScriptedChatModel(responses=[verdict()])

    outcome = reflect(model, make_group({"H-01": "a.go"}), [])

    assert outcome.kept == []
    assert model.calls == []


# ---------- 送给模型的材料 ----------


def test_prompt_carries_diff_and_locate_metadata() -> None:
    """测试：提示词里同时有 diff、评论正文与定位元数据。

    定位元数据是「先定位后反思」换来的那部分信号，漏掉它等于白做定位。
    """
    model = ScriptedChatModel(responses=[verdict()])
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

    first = request_text(model)
    assert "<diff>" in first and 'id="H-01"' in first
    assert 'id="C-01"' in first and 'id="C-02"' in first
    assert 'locate="hunk" side="LEFT" lines="7"' in first
    assert 'locate="failed:no_match"' in first
    # 未定位的评论不给位置属性 —— 宁可少一个属性，也不给一个假坐标
    assert 'locate="failed:no_match" side=' not in first
    # 两侧职责相反，system 必须是复核那份，不是审查那份
    assert "复核者" in first


def test_comments_are_numbered_in_order() -> None:
    """测试：评论按传入顺序编号 C-01、C-02、C-03。"""
    model = ScriptedChatModel(responses=[verdict()])

    reflect(
        model,
        make_group({"H-01": "a.go"}),
        [make_comment("一"), make_comment("二"), make_comment("三")],
    )

    first = request_text(model)
    assert first.index('id="C-01"') < first.index('id="C-02"') < first.index('id="C-03"')


def test_the_real_tool_is_bound() -> None:
    """测试：绑的是那个**实体工具**（有执行体、有 `Args:` 参数说明）。

    与「只当输出格式的 schema」的区别就在这里：那个是 `convert_to_openai_tool`
    出来的字典，没有 `.name` / 不能 `invoke`。
    """
    model = ScriptedChatModel(responses=[verdict()])

    run(model, [make_comment("x")])

    bound = model.tools_seen[0][0]
    assert isinstance(bound, BaseTool)
    assert bound.name == TOOL_NAME


def test_tool_schema_shape() -> None:
    """测试：工具的参数形状固定 —— removals 与 misses 两个数组，嵌套字段带说明。

    参数说明走 Pydantic 的 `Field`（嵌套结构只能这么表达），而**不是**
    `args_schema=` —— 那个会让 `runtime` 注入失效（见 CLAUDE.md 框架缺口表）。
    """
    schema = convert_to_openai_tool(report_incorrect_comments)["function"]

    assert schema["name"] == TOOL_NAME
    properties = schema["parameters"]["properties"]
    assert set(properties) == {"removals", "misses"}
    removal_fields = properties["removals"]["items"]["properties"]
    assert set(removal_fields) == {"comment_id", "ground", "reasoning"}
    assert removal_fields["ground"]["enum"] == ["A", "B"]


# ---------- 日志 ----------


def test_tool_definition_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """测试：**工具定义进日志**（含完整参数 schema）。

    这是走 `ReviewTrace` 中间件换来的 —— 裸 `bind_tools(...).invoke(...)` 那条
    路径上，`bind_tools` 只是把 schema 发给 API，本地一个字都不留。
    """
    model = ScriptedChatModel(responses=[verdict()])

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER):
        run(model, [make_comment("x")])

    assert "工具定义 · 1 个" in caplog.text
    assert f"name: {TOOL_NAME}" in caplog.text
    assert '"removals"' in caplog.text


def test_tool_call_is_logged_with_args(caplog: pytest.LogCaptureFixture) -> None:
    """测试：每次工具调用的**参数**与耗时都进日志（格式与审查流程一致）。"""
    model = ScriptedChatModel(responses=[verdict(removal("C-01", "B", "第 8 行"))])

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER):
        run(model, [make_comment("x")])

    assert TOOL_NAME in caplog.text
    assert "执行" in caplog.text and "ms" in caplog.text
    assert "'comment_id': 'C-01'" in caplog.text, "工具参数要看得见"


def test_completion_is_logged_with_counts_and_grounds(caplog: pytest.LogCaptureFixture) -> None:
    """测试：完成日志带删除数、保留数、删除率，以及删除依据（审计用）。"""
    comments = [make_comment("第一条"), make_comment("第二条"), make_comment("第三条")]
    model = ScriptedChatModel(responses=[verdict(removal("C-02", "A", "diff 里没有那段"))])

    with caplog.at_level(logging.INFO, logger=REFLECT_LOGGER):
        run(model, comments)

    assert "反思 · 开始 · group-001 · 候选 3 条" in caplog.text
    assert "反思 · 完成" in caplog.text
    assert "删除 1 / 保留 2（删除率 33%）" in caplog.text
    assert "Ground A：diff 里没有那段" in caplog.text


def test_request_is_logged_in_full(caplog: pytest.LogCaptureFixture) -> None:
    """测试：请求（system + user）全文进日志。

    复核不再自己打这两条 —— 交给 `ReviewTrace`，于是轮次号、token 用量、
    finish_reason 这些与审查流程是同一套。
    """
    model = ScriptedChatModel(responses=[verdict()])

    with caplog.at_level(logging.INFO, logger=TRACE_LOGGER):
        run(model, [make_comment("被复核的评论正文")])

    assert "模型配置" in caplog.text
    assert "被复核的评论正文" in caplog.text
    assert "轮次=1" in caplog.text
