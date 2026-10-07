"""用 AACR-Bench 的**真实标注**标定反思（复核）阶段。

**这个脚本回答一个问题**：给一批**真实 PR 上的真实评论**（其中一部分被专家标为
低质 / 错误），我们的复核阶段会不会把它们拦下来？会不会误删正确的那些？

数据集：`Alibaba-Aone/aacr-bench`（HuggingFace，单文件 `dataset.json`）。
每行是一条**评论**，不是一个 PR：

    project_main_language / pr_url / pr_source_commit / pr_target_commit
    path / side / from_line / to_line / note / category / label
    label: 1 = 专家验证正确；0 = 低质 / 错误

**必须先设定期望**：数据集问的是「这条评论值不值得提」（宽口径，含风格、价值判断），
而我们的复核**只删被 diff 直接推翻的**（窄口径，`REFLECTION_SYSTEM` 里写着「拿不准就
放过」+ 保护性否决）。所以：

- **误删率必须接近 0**（删掉 label=1 = 把对的评论弄没了，代价最高）
- **拦截率会偏低，这是设计使然**，不是失败：数据集里大量 label=0 是「说得太浅 / 风格
  问题 / 价值不高」，那些我们**故意不删**

跑法（真调 LLM，只跑 Go 子集时约 22 次调用 / 十几秒 / 几分钱）：

    PRISM_BENCH_DATASET=/tmp/aacr.json \
    PRISM_LOG_LEVEL=INFO uv run pytest tests/manual/calibrate_reflection_bench.py -s

    # 只跑前 3 个 PR（先验证流程）
    PRISM_BENCH_LIMIT=3 uv run pytest tests/manual/calibrate_reflection_bench.py -s

数据集下载（2 MB）：

    curl -sL https://huggingface.co/datasets/Alibaba-Aone/aacr-bench/resolve/main/dataset.json \
        -o /tmp/aacr.json

看什么：

    每个 PR 一行：候选 / 正确 / 低质 / 误删 / 拦截
    末尾总计：**误删率**（越低越好）与**拦截率**（偏低是预期的）

**改 `guidance/templates.py` 的 `REFLECTION_SYSTEM` / `tools.py` 之后要跑这个** ——
它比 `calibrate_reflection.py`（手搓假评论）真实得多。
"""

import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from prism.config.settings import settings
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.core.reflection.reflect import reflect
from prism.models.comment import (
    LOCATE_HUNK,
    SIDE_LEFT,
    SIDE_RIGHT,
    Comment,
)
from prism.models.grouping import Group

# 数据集里 label=1 是「专家验证正确」，label=0 是「低质 / 错误」
LABEL_CORRECT = 1

# side 取值：数据集用小写，我们内部照抄 GitHub 用大写
SIDE_MAP = {"right": SIDE_RIGHT, "left": SIDE_LEFT}


def read_options() -> dict[str, Any]:
    """从环境变量取参数（与 `test_locate_against_git.py` 同一套做法）。

    **不用 pytest 的命令行参数**：pytest 会把不认识的选项直接报错，为几个可选参数
    去改 `conftest.py` 加 `pytest_addoption`，收益不抵那份共享改动。

        PRISM_BENCH_DATASET     数据集 json（默认 /tmp/aacr.json）
        PRISM_BENCH_REPO_CACHE  仓库克隆缓存（默认复用 prism-bench 的那份）
        PRISM_BENCH_LANGUAGE    只看哪个语言（默认 Go）
        PRISM_BENCH_LIMIT       只跑前 N 个 PR（默认 0 = 全跑）

    Returns:
        参数字典
    """
    import os

    return {
        "dataset": os.getenv("PRISM_BENCH_DATASET", "/tmp/aacr.json"),
        "repo_cache": os.getenv(
            "PRISM_BENCH_REPO_CACHE",
            str(Path.home() / "Projects/python/prism-bench/evaluation/repo"),
        ),
        "language": os.getenv("PRISM_BENCH_LANGUAGE", "Go"),
        "limit": int(os.getenv("PRISM_BENCH_LIMIT", "0")),
    }


def load_rows(path: Path, language: str) -> list[dict[str, Any]]:
    """读数据集，筛出目标语言的评论行。

    Args:
        path: `dataset.json` 路径
        language: `project_main_language` 的取值

    Returns:
        评论行列表
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    picked = [row for row in rows if row.get("project_main_language") == language]
    print(f"\n数据集 {path.name}：{len(rows)} 行，其中 {language} {len(picked)} 行")
    return picked


def repo_slug(pr_url: str) -> str:
    """从 PR URL 里取 `owner/name`。

    Args:
        pr_url: 形如 ``https://github.com/owner/name/pull/123``

    Returns:
        ``owner/name``
    """
    parts = pr_url.rstrip("/").split("/")
    return f"{parts[3]}/{parts[4]}"


def ensure_repo(cache: Path, slug: str) -> Path:
    """确保仓库在缓存里（沿用 harness 的 `<owner>__<name>` 命名，两边共用一份）。

    Args:
        cache: 缓存目录
        slug: ``owner/name``

    Returns:
        本地仓库路径

    Raises:
        RuntimeError: clone 失败
    """
    target = cache / slug.replace("/", "__")
    if (target / ".git").exists():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    print(f"  克隆 {slug} → {target}（只拉一次，后续复用）")
    result = subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{slug}.git", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"clone 失败：{result.stderr.strip()}"
        raise RuntimeError(msg)
    return target


def ensure_commit(repo: Path, sha: str) -> None:
    """确保某个提交在本地仓库里（不在就按 SHA 直接 fetch）。

    **这一步不能省**：PR 的头提交常常来自 fork，既不是上游的任何分支，也不在
    `git clone` 拉下来的历史里 —— 实测报错是 `无法解析版本 'dc0b3e63…'`。
    按 SHA fetch 是 GitHub 支持的做法（harness 的 `ensure_commit_present` 同此）。

    Args:
        repo: 仓库路径
        sha: 提交 SHA

    Raises:
        RuntimeError: 本地没有、又 fetch 不到
    """
    check = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if check.returncode == 0:
        return
    print(f"  提交 {sha[:10]} 不在本地，按 SHA fetch")
    fetch = subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", "origin", sha],
        capture_output=True,
        text=True,
        check=False,
    )
    if fetch.returncode != 0:
        msg = f"fetch {sha[:10]} 失败：{fetch.stderr.strip()}"
        raise RuntimeError(msg)


def head_lines(repo: Path, rev: str, path: str, start: int, end: int) -> str:
    """取某个版本里文件的一段内容（用来当评论的「引用代码」）。

    数据集只给行号，不给评论引用的代码 —— 而复核提示词需要它。这里按**该侧的坐标系**
    取：`side=right` 取被审查版本、`side=left` 取基准版本。

    Args:
        repo: 仓库路径
        rev: 版本（分支 / SHA）
        path: 文件相对路径
        start: 起始行（1-based）
        end: 结束行（1-based，闭区间）

    Returns:
        代码文本；读不到时返回空串
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    lines = result.stdout.splitlines()
    return "\n".join(lines[max(start - 1, 0) : end])


def rows_to_comments(
    repo: Path, rows: Sequence[dict[str, Any]], group: Group, *, source: str, target: str
) -> tuple[list[tuple[Comment, int]], Counter[str]]:
    """把数据集行转成我们内部的评论（只留能挂进这一组的）。

    一个文件只属于一个包，所以「这一行落在哪一组」是确定的 —— 调用方按组分批，
    与生产一致（反思本来就是**每组一次**）。

    **标签跟着评论一起带出来**（`(评论, label)` 成对）—— 后面对着两个列表按位置
    找回标签，一轮 `del`/重建下来很容易错位（第一版就是这么写的）。

    Args:
        repo: 仓库路径
        rows: 该 PR 的评论行
        group: 复审用的审查单元（提供 diff 与 hunk 归属）
        source: 基准版本
        target: 被审查版本

    Returns:
        ([(评论, label)], 丢弃原因计数)
    """
    hunk_by_path = {item.file_path: item.hunk_id for item in group.hunks}
    comments: list[tuple[Comment, int]] = []
    dropped: Counter[str] = Counter()
    for row in rows:
        path = str(row.get("path") or "")
        hunk_id = hunk_by_path.get(path)
        if hunk_id is None:
            # 文件不在这一组里（被过滤掉，或没改动）→ 复核提示词里没有它的 diff，
            # 拿它去问「是否被 diff 推翻」是问不出来的
            dropped["文件不在审查单元里"] += 1
            continue
        start = int(row.get("from_line") or 0)
        end = int(row.get("to_line") or start)
        side = SIDE_MAP.get(str(row.get("side") or "").lower())
        rev = source if side == SIDE_LEFT else target
        comments.append(
            (
                Comment(
                    path=path,
                    severity="medium",  # 数据集没有严重度；复核用不到它，给个中性值
                    category=str(row.get("category") or ""),
                    content=str(row.get("note") or ""),
                    existing_code=head_lines(repo, rev, path, start, end),
                    hunk_id=hunk_id,
                    locate_level=LOCATE_HUNK,
                    side=side,
                    start_line=start,
                    end_line=end,
                ),
                int(row.get("label") or 0),
            )
        )
    return comments, dropped


def build_groups(repo: Path, source: str, target: str) -> list[Group]:
    """按我们的流水线跑一遍，得到这个 PR 的全部审查单元。

    Args:
        repo: 仓库路径
        source: 基准版本
        target: 被审查版本

    Returns:
        审查单元列表；这个 PR 没有可审 hunk 时返回空列表
    """
    ensure_commit(repo, source)
    ensure_commit(repo, target)
    reviewable = filter_files(load_diffs(repo, source, target)).reviewable
    mapped = map_hunks_to_symbols(repo, source, target, reviewable)
    return group_by_package(mapped)


@pytest.fixture(scope="module")
def report() -> Iterator[dict[str, Counter[str]]]:
    """跑完整轮标定，打印逐 PR 与总计。

    Yields:
        统计（供用例做粗断言）
    """
    args = read_options()
    rows = load_rows(Path(args["dataset"]), str(args["language"]))
    if args["limit"]:
        prs = sorted({r["pr_url"] for r in rows})[: int(args["limit"])]
        rows = [r for r in rows if r["pr_url"] in prs]

    by_pr: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pr[(row["pr_url"], row["pr_source_commit"], row["pr_target_commit"])].append(row)

    model: BaseChatModel = LLMClient(settings.llm).model
    cache = Path(args["repo_cache"])
    total: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    print(f"\n=== 逐 PR（{len(by_pr)} 个 PR）===")
    print(f"{'PR':<14}{'候选':>5}{'正确':>5}{'低质':>5}{'误删':>5}{'拦截':>5}   备注")
    for (pr_url, source, target), items in sorted(by_pr.items()):
        name = pr_url.split("/")[-1]
        repo = ensure_repo(cache, repo_slug(pr_url))
        groups = build_groups(repo, source, target)
        if not groups:
            dropped["这个 PR 没有可审 hunk（改动全被过滤）"] += len(items)
            print(f"{name:<14}{len(items):>5}{'':>5}{'':>5}{'':>5}{'':>5}   跳过：无可审 hunk")
            continue

        # 一个文件只属于一个包 → 每条评论能唯一落到某一组；复审**按组跑**
        # （与生产一致：反思本来就是每组一次）
        group_of_path = {path: group for group in groups for path in group.files}
        batches: dict[int, list[tuple[Comment, int]]] = defaultdict(list)
        for row in items:
            group = group_of_path.get(str(row.get("path") or ""))
            if group is None:
                dropped["文件不在审查单元里（被过滤 / 没改动）"] += 1
                continue
            got, _ = rows_to_comments(repo, [row], group, source=source, target=target)
            batches[id(group)].extend(got)

        kept_ids: set[int] = set()
        removals: list[Any] = []
        for group in groups:
            batch = batches.get(id(group))
            if not batch:
                continue
            outcome = reflect(model, group, [comment for comment, _ in batch])
            kept_ids |= {id(comment) for comment in outcome.kept}
            removals.extend(outcome.removed)

        reviewed = [pair for batch in batches.values() for pair in batch]
        correct = sum(1 for _, label in reviewed if label == LABEL_CORRECT)
        low = len(reviewed) - correct
        deleted = [(comment, label) for comment, label in reviewed if id(comment) not in kept_ids]
        removed_correct = sum(1 for _, label in deleted if label == LABEL_CORRECT)
        removed_low = sum(1 for _, label in deleted if label != LABEL_CORRECT)

        total["评论"] += len(reviewed)
        total["正确"] += correct
        total["低质"] += low
        total["误删"] += removed_correct
        total["拦截"] += removed_low
        print(
            f"{name:<14}{len(reviewed):>5}{correct:>5}{low:>5}"
            f"{removed_correct:>5}{removed_low:>5}   {len(groups)} 组"
        )
        # **删掉的逐条打出来**：删评论是这条流水线上唯一有破坏性的动作，必须可审计。
        # 误删（专家标为正确却被删）尤其要看 —— 是复核真抓到了 diff 矛盾，还是判断错了
        for comment, label in deleted:
            flag = "⚠️ 误删" if label == LABEL_CORRECT else "拦截"
            print(f"    · {flag} {comment.path}:{comment.start_line} | {comment.content[:70]}")
        for item in removals:
            print(f"      依据[{item.ground}] {item.reasoning[:110]}")
    print("\n=== 总计 ===")
    print(f"  进了复核的评论：{total['评论']}（正确 {total['正确']} / 低质 {total['低质']}）")
    print(
        f"  误删（删掉了标注为正确的）：{total['误删']}"
        f"（{total['误删'] / max(total['正确'], 1) * 100:.1f}%）  ← 越低越好"
    )
    print(
        f"  拦截（删掉了标注为低质的）：{total['拦截']}"
        f"（{total['拦截'] / max(total['低质'], 1) * 100:.1f}%）  ← 偏低是预期的："
        "我们只删「被 diff 直接推翻」的，数据集问的是「值不值得提」"
    )
    if dropped:
        print(f"  没进复核：{dict(dropped)}")
    yield {"total": total, "dropped": dropped}


def test_no_correct_comment_is_lost(report: dict[str, Counter[str]]) -> None:
    """**误删率要接近 0** —— 这是复核阶段最重要的断言。

    删掉一条标注为「专家验证正确」的评论是最高代价的错误（真实发现静默消失），
    提示词里为此写了「拿不准就放过」+ 保护性否决。这条用例不设阈值（样本量还在攒），
    但把它打印出来盯住趋势。
    """
    total = report["total"]
    assert total["评论"] > 0, "数据集要真的读到东西 —— 路径或语言参数是不是错了？"
    assert total["误删"] == 0, f"误删了 {total['误删']} 条标注为正确的评论，逐条看上面那张表"


def test_low_quality_interception_is_reported(report: dict[str, Counter[str]]) -> None:
    """拦截率不设下限（偏低是设计使然），但必须算得出来、且样本足够。

    这一条的用途是**防止「数字看起来是 0 因为根本没跑到」**：确认低质样本真的进了
    候选（而不是全被「文件不在审查单元里」筛掉）。
    """
    total = report["total"]
    assert total["低质"] > 0, "一条低质样本都没进来，统计没意义"
