"""静态分析场景校准：静态发现真的进到提示词里了吗、模型会不会引用它。

**这个脚本回答三个问题**：

1. 现场跑一次 golangci-lint，发现**能不能关联到 hunk**（关联不上就只是一段
   游离的文本，hunk 属性 `static_hint` 也不会出现）
2. 注入到提示词里的那一版长什么样（表格、`static_hint`、排序）
3. 模型确认了某条发现时，会不会在评论正文里带上 `[S-01]` —— 后置校验靠它
   判断「这条发现有人管」

用的是一个**新建文件**：新文件的 hunk 覆盖全部行，所以静态发现必然落在 hunk
区间里，关联这一步不靠运气。文件里埋了 gosec（变量 URL、按变量读文件）与
errcheck（没检查的返回值）。

跑法（真调 LLM 一次，1 组、几轮、约 20 秒）：

    PRISM_LOG_LEVEL=INFO uv run pytest tests/manual/check_static.py -s

**期望**：第 2 步打印的表格里有 4~6 条发现、`关联 hunk` 列不是占位符、
hunk 属性上有 `static_hint`；第 4 步打印的评论里至少有条件地带着 `[S-xx]`。
一条都没被引用不算失败（模型这一次恰好自己也想起来了，分不出来），
但**一条发现都没进提示词**就是失败 —— 那说明关联或预算这两步有一处断了。

改过 `core/static/` 的任一文件、或改过提示词的静态那两段之后要跑这个。
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from prism.config.settings import settings
from prism.core.agent.agent import run_review_agent
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import fixed_prompt_tokens, group_by_package, group_tokens
from prism.core.guidance.builder import build_diff_block, build_static_block
from prism.core.llm.client import LLMClient
from prism.core.output.locate import locate_group
from prism.core.static.analyzer import analyze, format_finding, select_for_group
from prism.core.static.verify import verify_group
from prism.models.grouping import Group
from prism.models.static import StaticFinding

BASE = "main"
HEAD = "HEAD"

_GO_MOD = "module staticfixture\n\ngo 1.22\n"

_MAIN_BEFORE = """\
package main

import "fmt"

// main 启动服务。
func main() {
\tfmt.Println("up")
}
"""

# 这次改动的全部内容：新增一个包，里面集中几条静态工具抓得住的问题
_API_NEW = """\
package api

import (
\t"net/http"
\t"os"
)

// Serve 按名字读一个文件并回给调用方。
func Serve(w http.ResponseWriter, r *http.Request) {
\tquery := r.URL.Query()

\tresp, err := http.Get(query.Get("url"))
\tif err != nil {
\t\thttp.Error(w, "bad gateway", http.StatusBadGateway)
\t\treturn
\t}
\tdefer resp.Body.Close()

\tname := query.Get("name")
\tdata, err := os.ReadFile(name)
\tif err != nil {
\t\thttp.Error(w, "not found", http.StatusNotFound)
\t\treturn
\t}

\tf, err := os.Create(name)
\tif err != nil {
\t\treturn
\t}
\tf.WriteString(string(data))
\tw.Write(data)
}
"""


def _git(repo: Path, *args: str) -> None:
    """在仓库里跑一条 git 命令。

    Args:
        repo: 仓库路径
        args: git 参数
    """
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _write(repo: Path, path: str, text: str) -> None:
    """写入仓库内的一个文件。

    Args:
        repo: 仓库路径
        path: 仓库内相对路径
        text: 文件内容
    """
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def static_repo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """搭一个「PR 新增了一个问题不少的文件」的临时仓库。

    Args:
        tmp_path_factory: pytest 的临时目录工厂

    Yields:
        仓库路径，工作区停在 feature 分支
    """
    repo = tmp_path_factory.mktemp("static")
    _git(repo, "init", "-q", "-b", BASE)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")

    _write(repo, "go.mod", _GO_MOD)
    _write(repo, "main.go", _MAIN_BEFORE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "api/serve.go", _API_NEW)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat(api): 新增文件下载接口")
    yield repo


@pytest.fixture(scope="module")
def injected(static_repo: Path) -> tuple[Group, list[StaticFinding]]:
    """跑一次真实静态分析，打印发现清单与注入提示词的那一版。

    Args:
        static_repo: 临时仓库

    Returns:
        (审查单元, 注入本组的静态发现)
    """
    reviewable = filter_files(load_diffs(static_repo, BASE, HEAD)).reviewable
    mapped = map_hunks_to_symbols(static_repo, BASE, HEAD, reviewable)
    report = analyze(static_repo, HEAD, sorted({item.file_path for item in mapped}), mapped)

    print(f"\n=== 静态分析（跳过原因={report.skip_reason}，耗时 {report.elapsed_seconds:.1f}s）===")
    for item in report.findings:
        print(format_finding(item))

    group = group_by_package(mapped)[0]
    selected = select_for_group(group.files, report.findings)
    total = fixed_prompt_tokens() + group_tokens(group.hunks, report.findings)
    print(
        f"\n=== 注入 {group.group_id} 的 {len(selected)} 条"
        f"（本组约 {total}/{settings.review.max_group_tokens} token）==="
    )
    print(build_static_block(selected) or "（空：这一组没有静态发现）")

    print("\n=== 带 static_hint 的 diff 块 ===")
    print(build_diff_block(group, selected))
    return group, selected


@pytest.fixture(scope="module")
def reviewed(static_repo: Path, injected: tuple[Group, list[StaticFinding]]) -> Iterator[list[str]]:
    """跑一次真实审查，打印评论与后置校验的结果。

    Args:
        static_repo: 临时仓库
        injected: (审查单元, 注入本组的静态发现)

    Yields:
        每条评论的正文（供用例做粗断言）
    """
    group, selected = injected
    model: BaseChatModel = LLMClient(settings.llm).model
    result = run_review_agent(group, model, str(static_repo), HEAD, static_findings=selected)
    located = locate_group(group, result.hunk_comments)
    verdict = verify_group(selected, located)

    print(f"\n=== 定位后的评论（{len(located)} 条）===")
    for comment in located:
        print(f"[{comment.severity} · {comment.category}] {comment.path}:{comment.start_line}")
        print(comment.content)

    print(
        f"\n=== 后置校验：引用 {list(verdict.referenced)}，"
        f"无效 {list(verdict.invalid)}，高优先级未处理 {verdict.unhandled_count} 条 ==="
    )
    for item in verdict.unhandled_high:
        print(f"  未处理：{item.finding_id} {item.path}:{item.line} {item.text}")
    yield [comment.content for comment in located]


def test_findings_land_on_hunks(injected: tuple[Group, list[StaticFinding]]) -> None:
    """静态发现必须关联到 hunk，否则 `static_hint` 与排序都不会发生。

    Args:
        injected: (审查单元, 注入本组的静态发现)
    """
    _group, selected = injected
    assert selected, "这一组必须有静态发现 —— 一条都没有说明分析或过滤断了"
    unassociated = [item for item in selected if item.hunk_id is None]
    assert not unassociated, f"这些发现没关联到 hunk：{[i.finding_id for i in unassociated]}"


def test_prompt_carries_static_hint(injected: tuple[Group, list[StaticFinding]]) -> None:
    """`static_hint` 必须出现在 diff 块里（模型靠它把 hunk 与发现对上）。

    Args:
        injected: (审查单元, 注入本组的静态发现)
    """
    group, selected = injected
    block = build_diff_block(group, selected)
    for item in selected:
        assert f'static_hint="{item.finding_id}' in block or item.finding_id in block
    assert "### 静态分析提示" in build_static_block(selected)
    assert "（以下按静态分析的建议顺序排列，仅供参考）" in block


def test_model_may_reference_findings(reviewed: list[str]) -> None:
    """占位用例：真正的判定在打印出来的评论里（见模块 docstring）。

    **不断言**模型一定引用了编号 —— 那是提示词措辞在起作用还是模型这一次恰好
    自己也想起来了，单次运行分不出来（同 `check_cross_file.py` 的态度）。

    Args:
        reviewed: 评论正文列表
    """
    assert reviewed is not None
