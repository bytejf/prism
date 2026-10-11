"""`action.yml` 与示例 workflow 的契约测试。

**为什么要这样一条**（照着 OCR 那边的做法抄的）：action 里的错**本地跑不出来** ——
引用一个不存在的输入、声明一个 Prism 根本不读的环境变量、把报告写到 `hashFiles`
看不见的地方，都要等真推一个 PR 才暴露，而那时人已经在看别的东西了。

这里钉住的都是**具体踩过的**（第一版里就有两处）：

- `PRISM_LLM_MODEL` / `PRISM_LLM_BASE_URL` —— 我按常理以为模型和 base_url 能走
  环境变量，实际 Prism **只从 `DEEPSEEK_API_KEY` 读密钥**，模型与地址在
  `config/default.toml` 里（`settings.py` 只 `getenv` 了那一个）
- `hashFiles('/tmp/...')` —— `hashFiles` 只看工作区，看不到 `/tmp`，于是「发评论」
  那一步的门条件**永远是假**
"""

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML 没有类型桩，也不值当为它加一个 dev 依赖

REPO = Path(__file__).resolve().parents[2]
ACTION = REPO / "action.yml"
WORKFLOW = REPO / ".github" / "workflows" / "prism-review.yml"

# Prism **真的会读**的环境变量。加一个之前先去 `config/settings.py` 与
# `observability/` 里确认它存在 —— 这条清单就是「别声明一个不存在的旋钮」的守卫
KNOWN_ENV = frozenset(
    {
        "DEEPSEEK_API_KEY",  # settings.py 的 API_KEY_ENV
        "PRISM_LOG_LEVEL",
        "PRISM_TRACE_FILE",
        "PRISM_TRACE_DIR",
        "GITHUB_TOKEN",  # 不是 Prism 的，是 gh-post 读的
        "GITHUB_OUTPUT",
        "GITHUB_REPOSITORY",
        "RUNNER_TEMP",
    }
)

# **不用 `importorskip`**：跳过等于这条守卫不跑，而它守的正是「本地跑不出来」的
# 那类错。PyYAML 是 langchain（已声明的依赖）拉进来的，真的没有它说明依赖变过 ——
# 那时应该红给我看，而不是安静地跳过


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    """解析 `action.yml`。"""
    return cast("dict[str, Any]", yaml.safe_load(ACTION.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """解析示例 workflow。"""
    return cast("dict[str, Any]", yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def test_action_is_a_composite_with_the_expected_steps(action: dict[str, Any]) -> None:
    """形态：composite action，五个步骤，顺序是「装 → 审 → 投递 → 传产物」。"""
    assert action["runs"]["using"] == "composite"
    names = [step.get("name", "") for step in action["runs"]["steps"]]
    assert names == [
        "Set up uv",
        "Fetch Prism source",
        "Install Prism",
        "Run review",
        "Post review comments",
        "Upload report",
    ]


def test_every_referenced_input_exists(action: dict[str, Any]) -> None:
    """`${{ inputs.X }}` 里的每个 X 都得在 `inputs:` 里 —— 打错一个字母就是空值。

    composite action 引用不存在的输入**不报错**，只是渲染成空串。后果可能是
    「密钥传了个空 → 审查跑不起来」，而那看起来像配置问题，不像打错了字。
    """
    text = ACTION.read_text(encoding="utf-8")
    referenced = set(re.findall(r"inputs\.([a-z_][a-z0-9_]*)", text))
    assert referenced, "一个输入都没用？"
    missing = referenced - set(action["inputs"])
    assert not missing, f"引用了不存在的输入：{sorted(missing)}"


def test_action_reads_no_env_var_prism_does_not_have(action: dict[str, Any]) -> None:
    """action 里出现的 `PRISM_*` / `*_API_KEY` 都必须是 Prism 真读的变量。

    这是**跨文件**的一致性：action 声明了一个不存在的旋钮，会比「没这个功能」
    更糟 —— 使用者以为自己配上了，而 Prism 一个字都没读到。
    """
    text = ACTION.read_text(encoding="utf-8")
    candidates = re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", text)
    used = {name for name in candidates if name.startswith(("PRISM_", "DEEPSEEK_"))}
    assert used <= KNOWN_ENV, f"声明了 Prism 不读的变量：{used - KNOWN_ENV}"


def test_api_key_is_passed_through_env_only(action: dict[str, Any]) -> None:
    """密钥只经 `env:` 传，**绝不出现在命令行或 `echo` 里**。

    命令行会进日志（`ps` 也能看到），echo 更是直接把密钥写进 job log。
    """
    text = ACTION.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY: ${{ inputs.api_key }}" in text
    assert 'echo "${{ inputs.api_key' not in text
    assert "--api-key" not in text, "别把密钥做成命令行参数"


def test_posting_is_gated_on_a_successful_review(action: dict[str, Any]) -> None:
    """投递那一步要挂在「审查真的出报告了」上 —— 否则会去投一份不存在的报告。"""
    steps = {step.get("id"): step for step in action["runs"]["steps"]}
    assert steps["post"]["if"] == "${{ steps.review.outcome == 'success' }}"
    assert steps["upload"]["if"] == "${{ always() && steps.review.outputs.report != '' }}", (
        "上传要在失败时也跑（那是排查的唯一线索），但报告路径必须来自 review 那一步"
    )


def test_report_path_lives_in_runner_temp(action: dict[str, Any]) -> None:
    """报告写在 `runner.temp`，并且路径由 review 那一步输出。

    第一版写的是固定路径 `/tmp/prism-report.json` + `hashFiles(...)` 当门条件 ——
    `hashFiles` 只在工作区里找，永远返回空，于是投递那一步**永远不执行**。
    这类错不会报错，只是什么都不发生。
    """
    text = ACTION.read_text(encoding="utf-8")
    assert "runner.temp" in text
    assert "hashFiles" not in text, "hashFiles 看不到 runner.temp 这种工作区外的路径"
    assert 'echo "report=$report"' in text


def test_prism_is_installed_from_a_pinned_ref(action: dict[str, Any]) -> None:
    """装的是**固定 ref**（默认 = 本 action 的 ref），不从被审的树里装。

    **必须是「检出到子目录 + 本地装」而不是匿名 clone**：本仓库是私有的，
    匿名 `git+https://github.com/...` 会以 `could not read Username` 失败
    （实测）。带 token 检出公开私有都能用，且拿到的仍是我们指定的那个 ref。
    """
    assert action["inputs"]["prism_ref"]["default"] == "${{ github.action_ref || 'main' }}"
    steps = {step.get("name"): step for step in action["runs"]["steps"]}
    fetch = steps["Fetch Prism source"]
    assert fetch["uses"].startswith("actions/checkout@")
    assert fetch["with"]["ref"] == "${{ inputs.prism_ref }}"
    assert fetch["with"]["path"] == ".prism-source"
    # **读源码与发评论是**两个** token**：运行自带的 token 只能读它自己那个仓库，
    # 所以「从别的私有仓库取源码」要单独传一个只读 token —— 发评论那份保持最小权限
    assert fetch["with"]["token"] == "${{ inputs.source_token }}", "私有仓库要认证才能检出"
    assert fetch["with"]["token"] != "${{ inputs.github_token }}", "两个用途别共用一个 token"
    install = steps["Install Prism"]["run"]
    assert "uv tool install --force" in install and ".prism-source" in install


# ---------- 示例 workflow ----------


def test_workflow_triggers_on_every_push(workflow: dict[str, Any]) -> None:
    """每次 push 都重审（用户定的语义），并且**取消同 PR 的旧运行**。

    取消是省钱的一半：连推两次时，旧的那次跑到一半被取消，而新的一次会先删掉
    我们上次发的评论 —— 所以取消不会在 PR 上留下半份结论。
    """
    # YAML 1.1 把 `on:` 解析成布尔键 True（`yaml.safe_load` 至今如此）——
    # 两种形态都认，免得因为解析器的这个老坑让用例时红时绿
    trigger = (workflow.get("on") or workflow[True])["pull_request"]  # type: ignore[index]
    assert set(trigger["types"]) == {"opened", "synchronize", "reopened"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert workflow["concurrency"]["group"] == "prism-${{ github.event.pull_request.number }}"


def test_workflow_asks_for_the_permissions_it_needs(workflow: dict[str, Any]) -> None:
    """`pull-requests: write` 是发评论必需的；`contents: read` 是够用的最小集。"""
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_workflow_only_reviews_same_repo_prs(workflow: dict[str, Any]) -> None:
    """只审同仓库 PR，且 fork 那条路**要说出来**（静默跳过会让人以为没问题）。

    这是本项目最大的一处安全取舍：Prism 是**解释执行的 Python 包**，装它就会
    执行 `pyproject.toml` 里的构建配置。用 `pull_request_target` 让 fork 也拿到
    密钥，等于「在持有密钥与写权限的进程里执行任意外部 PR 提供的构建配置」。
    """
    jobs = workflow["jobs"]
    same_repo = "github.event.pull_request.head.repo.full_name == github.repository"
    assert jobs["review"]["if"] == same_repo
    assert jobs["skipped-fork"]["if"] == (
        "github.event.pull_request.head.repo.full_name != github.repository"
    )
    assert "::notice::" in jobs["skipped-fork"]["steps"][0]["run"]


def test_workflow_installs_prism_from_the_base_branch(workflow: dict[str, Any]) -> None:
    """`prism_ref` 取 **base 分支的提交**，不能用 `github.sha`。

    `pull_request` 事件里的 `github.sha` 是 GitHub 造的**合并提交**，只存在于
    runner 上 —— `uv tool install git+...@<那个 SHA>` 拉不到它。
    """
    step = workflow["jobs"]["review"]["steps"][-1]
    assert step["with"]["prism_ref"] == "${{ github.event.pull_request.base.sha }}"
    assert "github.sha" not in str(step["with"])


def test_workflow_checks_out_full_history(workflow: dict[str, Any]) -> None:
    """`fetch-depth: 0`：Prism 要算 merge-base，两个提交都得在本地。"""
    checkout = workflow["jobs"]["review"]["steps"][0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0


# composite action 的步骤**只允许这些键**（GitHub runner 的清单校验是硬的：
# 出现别的键时它**在加载阶段就拒掉整个 action**，日志里只有一句
# `Unexpected value 'xxx'`）。第一版我把 `timeout-minutes` 写在步骤上，就是这么炸的
COMPOSITE_STEP_KEYS = frozenset(
    {
        "name",
        "id",
        "if",
        "uses",
        "run",
        "shell",
        "env",
        "with",
        "working-directory",
        "continue-on-error",
    }
)


def test_composite_steps_only_use_supported_keys(action: dict[str, Any]) -> None:
    """composite 的每个步骤只能用 GitHub 支持的键。

    **这条用例是有来历的**：第一版在「Run review」那步写了 `timeout-minutes`，
    本地 YAML 解析得好好的、`actionlint` 也没跑，直到真推一个 PR 才报

        action.yml (Line: 101, Col: 7): Unexpected value 'timeout-minutes'

    —— 而且是在**加载 action 的阶段**就失败，连第一步都没跑。这类错本地跑不出来，
    只能靠把 runner 的规则写成用例。
    """
    for step in action["runs"]["steps"]:
        name = step.get("name")
        unknown = set(step) - COMPOSITE_STEP_KEYS
        assert not unknown, f"步骤 {name!r} 用了 composite 不支持的键：{sorted(unknown)}"
        assert "shell" in step or "uses" in step, f"步骤 {name!r} 既没有 shell 也没有 uses"


def test_job_timeout_lives_in_the_calling_workflow(workflow: dict[str, Any]) -> None:
    """卡死由**调用方的 job** 兜（`timeout-minutes` 在 job 上是支持的）。

    Prism 自己有整条 20 分钟的预算；这里是第二道 —— 防的是「job 卡住不动」。
    """
    assert workflow["jobs"]["review"]["timeout-minutes"] == 30
