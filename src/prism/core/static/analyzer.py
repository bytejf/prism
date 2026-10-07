"""静态分析：跑一次 golangci-lint，把发现解析成 `StaticFinding`。

设计文档：`acr-min/04-添加静态审查/18-静态分析集成设计.md`

四条硬约束，每条都有代价，都不省：

1. **跑的是被审查版本**。golangci-lint 只能分析「工作区」，读不了任意 rev，而
   我们的 `file_read` / `code_search` 读的都是 `--head`（见 CLAUDE.md）。所以先
   `git worktree add --detach` 把 head 检出到临时目录，在那里跑、跑完删掉 ——
   工作区停在别的分支时，报出来的行号才不会与待审代码错位。
2. **失败一律降级**。二进制不存在、超时、退出码非 0、JSON 解析失败、目标不是 Go
   模块 —— 全部只记 WARNING 后返回 `skip_reason`，审查照常继续（设计 §2.5）。
3. **只要可审查文件的发现**。golangci-lint 报的是整个仓库，我们按 reviewable 集合
   过滤 —— 没改动的文件不在审查范围内，它们的发现进来只会挤占 token 并诱导模型
   去评论没改过的代码。
4. **不做 `--new-from-*`**。那套 flag 是给 CI 用的（GitHub Action 的
   `only-new-issues` 就是它）：只留「改动的那几行上」的问题。我们的口径更宽也更好
   解释 —— 改过的**文件**里的发现全要，因为 hunk 关联与 token 截断已经各自兜了
   一道；再加一层 git blame 语义，只会多一个看不见的失效点。

跑法与解析都是实测定的（golangci-lint v2.14.0）：`--output.json.path=stdout` 仍会把
统计摘要和文本报告混进 stdout，必须同时给 `--output.text.path=stderr` 与
`--show-stats=false`，否则拿到的不是纯 JSON（报错是 `Extra data: line 2`）。
"""

import json
import logging
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from prism.config.settings import settings
from prism.models.ast import HunkWithSymbol
from prism.models.static import (
    NO_HUNK_MARK,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_RANK,
    StaticFinding,
    StaticReport,
)

logger = logging.getLogger(__name__)

GOLANGCI_LINT_BIN = "golangci-lint"

# 保留的 linter（设计 §2.2）。**v2 起 `gosimple` / `stylecheck` 已并入
# `staticcheck`**，所以设计文档那张表里的 `gosimple` 在这里没有对应项 ——
# 原样抄过来会让 golangci-lint 报 unknown linter 后整次分析失败。
LINTERS: tuple[str, ...] = (
    "gosec",
    "staticcheck",
    "errcheck",
    "govet",
    "ineffassign",
    "unused",
    "bodyclose",
    "sqlclosecheck",
)

# linter → 工具标记（设计 §2.2）。**这是契约不是旋钮**：它决定 hunk 的展示顺序与
# 「高优先级未处理」告警的判定，配错了只会静默改变行为，所以写死在代码里。
PRIORITY_BY_LINTER: dict[str, str] = {
    "gosec": PRIORITY_HIGH,
    "staticcheck": PRIORITY_HIGH,
    # typecheck 关不掉（`--default=none` 也照样启用），它报的是「代码根本编不过」：
    # 编译错误是能找到的最严重的问题，按 high 报
    "typecheck": PRIORITY_HIGH,
    "errcheck": PRIORITY_MEDIUM,
    "govet": PRIORITY_MEDIUM,
    "ineffassign": PRIORITY_MEDIUM,
    "bodyclose": PRIORITY_MEDIUM,
    "sqlclosecheck": PRIORITY_MEDIUM,
    "unused": PRIORITY_LOW,
}

# 没在表里的 linter 的兜底档位。按我们的跑法不该出现（只启用了上表 + typecheck），
# 真出现时按 medium 处理并记一笔 DEBUG：宁可多报一条也不要静默丢掉
UNKNOWN_LINTER_PRIORITY = PRIORITY_MEDIUM

# 编号格式 S-01：全局连续分配，与 hunk 编号同样的道理（跨组引用要能对上）
FINDING_ID_PREFIX = "S-"
FINDING_ID_WIDTH = 2

# 描述在提示词里的最大长度。golangci-lint 偶尔甩一长串（typecheck 是整段编译
# 错误），压成一行再截断 —— 表格的单元格里放不下换行
MAX_TEXT_LENGTH = 200


def analyze(
    root: Path,
    head_rev: str,
    reviewable: Sequence[str],
    hunks: Sequence[HunkWithSymbol],
) -> StaticReport:
    """跑一次静态分析，返回可审查文件上的发现。

    Args:
        root: 被审查仓库的根目录
        head_rev: 被审查的版本。为空时只能分析工作区
        reviewable: 可审查文件的路径（过滤阶段的产出）
        hunks: 带归属符号的 hunk 列表，用于把发现关联到 hunk

    Returns:
        StaticReport。任何一步失败都返回带 skip_reason 的空报告，不抛异常
    """
    if not reviewable:
        logger.debug("静态分析跳过：没有可审查文件")
        return StaticReport()

    try:
        with head_tree(root, head_rev) as tree:
            payload, elapsed = _run_linter(tree)
            findings, raw_count = _parse(payload, tree, reviewable)
    except _AnalysisError as exc:
        logger.warning("静态分析跳过：%s", exc)
        return StaticReport(skip_reason=str(exc))

    ordered = _order_and_number(_associate(findings, hunks))
    counts = _count_by_linter(ordered)
    # 两个数都要报：只看「可用发现」的话，「全被过滤掉了」与「工具没发现问题」
    # 长得一模一样
    logger.info(
        "静态分析完成：%.1f 秒，可用发现 %d 条（原始 %d 条）%s",
        elapsed,
        len(ordered),
        raw_count,
        f"，{counts}" if counts else "",
    )
    return StaticReport(findings=ordered, elapsed_seconds=elapsed, raw_count=raw_count)


def select_for_group(
    group_files: Sequence[str], findings: Sequence[StaticFinding]
) -> list[StaticFinding]:
    """挑出属于这些文件的发现（按文件分桶，设计 §2.3）。

    Args:
        group_files: 某个审查单元的文件列表
        findings: 全量静态发现

    Returns:
        该组文件上的发现，保持全量列表的顺序（已按优先级降序）
    """
    files = set(group_files)
    return [item for item in findings if item.path in files]


def format_finding(item: StaticFinding) -> str:
    """把一条发现渲染成一行（给人看的干跑入口用）。

    Args:
        item: 静态发现

    Returns:
        形如 ``S-01 [high] gosec pkg/a/iface.go:15 -> H-03``
    """
    hunk = item.hunk_id if item.hunk_id is not None else NO_HUNK_MARK
    return f"{item.finding_id} [{item.priority}] {item.linter} {item.path}:{item.line} -> {hunk}"


class _AnalysisError(Exception):
    """内部信号：这次静态分析没跑成（是降级，不是程序错误）。"""


@contextmanager
def head_tree(root: Path, head_rev: str) -> Iterator[Path]:
    """把被审查版本检出到临时目录，退出时删掉。

    没给 head_rev 时直接用工作区 —— 这时「被审查版本」就是工作区，两者的差别
    不存在，为它建一个 worktree 只是白花钱。

    Args:
        root: 仓库根目录
        head_rev: 被审查的版本

    Yields:
        可以分析的目录

    Raises:
        _AnalysisError: 检出失败（版本不存在、仓库损坏等）
    """
    if not head_rev:
        logger.debug("静态分析未指定 head，直接分析工作区")
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="prism-static-") as tmp:
        tree = Path(tmp) / "tree"
        _git(root, ["worktree", "add", "--detach", str(tree), head_rev])
        try:
            logger.debug("静态分析检出 %s → %s", head_rev, tree)
            yield tree
        finally:
            # --force：分析过程可能留下缓存文件，不让它挡住清理。
            # 清理失败只记日志：临时目录本身会被 TemporaryDirectory 删掉，
            # 留下的只是 git 的工作树登记项
            try:
                _git(root, ["worktree", "remove", "--force", str(tree)])
            except _AnalysisError as exc:
                logger.warning("清理临时 worktree 失败：%s", exc)


def _git(root: Path, args: list[str]) -> str:
    """跑一条 git 命令，失败时抛 `_AnalysisError`。

    Args:
        root: 仓库根目录（作为 cwd）
        args: git 子命令与参数

    Returns:
        stdout 文本

    Raises:
        _AnalysisError: git 不存在、超时、或退出码非 0
    """
    timeout = settings.static.timeout_seconds
    command = ["git", "-C", str(root), *args]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"git {' '.join(args)} 超时（{timeout} 秒）"
        raise _AnalysisError(msg) from exc
    except OSError as exc:
        msg = f"无法执行 git：{exc}"
        raise _AnalysisError(msg) from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        msg = f"git {' '.join(args)} 失败（rc={completed.returncode}）：{detail}"
        raise _AnalysisError(msg)
    return completed.stdout


def _run_linter(tree: Path) -> tuple[dict[str, Any], float]:
    """在给定目录里跑 golangci-lint，返回解析后的 JSON 与耗时。

    Args:
        tree: 要分析的目录

    Returns:
        (JSON 对象, 耗时秒数)

    Raises:
        _AnalysisError: 二进制不存在、超时、退出码非 0、输出不是合法 JSON
    """
    command = [
        GOLANGCI_LINT_BIN,
        "run",
        # 不用目标仓库自己的配置：我们的结论要可复现，不能被审的仓库改一行
        # .golangci.yml 就换一套 linter
        "--no-config",
        "--default=none",
        "-E",
        ",".join(LINTERS),
        # 发现问题不算失败 —— 让退出码只表示「跑没跑成」，一种信号一个含义
        "--issues-exit-code=0",
        "--max-issues-per-linter=0",
        "--max-same-issues=0",
        # stdout 只留 JSON：文本报告与统计摘要默认也走 stdout，混在一起就不是
        # 合法 JSON 了（实测报错：Extra data: line 2）
        "--output.json.path=stdout",
        "--output.text.path=stderr",
        "--show-stats=false",
        # 绝对路径：相对路径的基准是 golangci-lint 内部推的，在 worktree 里会变成
        # `../../../tmp/xxx` 这种。相对化我们自己按分析目录做，那个是确定的
        "--path-mode=abs",
    ]
    timeout = settings.static.timeout_seconds
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=tree, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"golangci-lint 超时（{timeout} 秒）"
        raise _AnalysisError(msg) from exc
    except FileNotFoundError as exc:
        msg = f"找不到 {GOLANGCI_LINT_BIN}（未安装或不在 PATH 上）"
        raise _AnalysisError(msg) from exc
    except OSError as exc:
        msg = f"无法执行 {GOLANGCI_LINT_BIN}：{exc}"
        raise _AnalysisError(msg) from exc
    elapsed = time.monotonic() - started

    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines()
        msg = f"golangci-lint 退出码 {completed.returncode}：{lines[-1] if lines else ''}"
        raise _AnalysisError(msg)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        msg = f"golangci-lint 输出不是合法 JSON：{exc}"
        raise _AnalysisError(msg) from exc
    if not isinstance(payload, dict):
        msg = "golangci-lint 输出不是 JSON 对象"
        raise _AnalysisError(msg)
    return payload, elapsed


def _parse(
    payload: dict[str, Any], tree: Path, reviewable: Sequence[str]
) -> tuple[list[StaticFinding], int]:
    """解析 JSON，只留可审查文件上的发现。

    Args:
        payload: golangci-lint 的 JSON 输出
        tree: 分析目录（用来把绝对路径相对化）
        reviewable: 可审查文件的路径

    Returns:
        (发现列表, 原始条数)。发现未排序、未编号、未关联 hunk；
        原始条数是工具报出的总数（含被过滤掉的），用于日志对账
    """
    allowed = set(reviewable)
    raw = payload.get("Issues") or []
    result: list[StaticFinding] = []
    for item in raw:
        path = _repo_relative(str((item.get("Pos") or {}).get("Filename") or ""), tree)
        if path is None:
            continue
        if path not in allowed:
            continue
        result.append(_to_finding(item, path))
    dropped = len(raw) - len(result)
    logger.debug(
        "静态发现解析：原始 %d 条，保留 %d 条%s",
        len(raw),
        len(result),
        f"，丢弃 {dropped} 条（不在可审查文件里，或不在分析目录内）" if dropped else "",
    )
    return result, len(raw)


def _repo_relative(raw: str, tree: Path) -> str | None:
    """把 golangci-lint 报的绝对路径转成仓库内相对路径。

    分析目录是临时 worktree，前缀每次都不同，所以**必须**按传进来的 tree 算，
    不能靠字符串截取。路径不在 tree 里时返回 None（那条发现会被丢掉并计数）——
    宁可少一条发现，也不要一个错的路径混进白名单比对。

    Args:
        raw: golangci-lint 报的路径
        tree: 分析目录

    Returns:
        相对路径（posix 形式）；不在分析目录内时返回 None
    """
    if not raw:
        return None
    try:
        return Path(raw).resolve().relative_to(tree.resolve()).as_posix()
    except ValueError:
        return None


def _to_finding(item: dict[str, Any], path: str) -> StaticFinding:
    """把一条原始 issue 转成 `StaticFinding`（编号此时还空着）。

    Args:
        item: golangci-lint 的一条 issue
        path: 已相对化的文件路径

    Returns:
        StaticFinding；`finding_id` 由 `_order_and_number` 填
    """
    linter = str(item.get("FromLinter") or "")
    priority = PRIORITY_BY_LINTER.get(linter)
    if priority is None:
        logger.debug("未在优先级表里的 linter：%s（%s）", linter, path)
        priority = UNKNOWN_LINTER_PRIORITY
    return StaticFinding(
        finding_id="",
        path=path,
        line=int((item.get("Pos") or {}).get("Line") or 0),
        linter=linter,
        priority=priority,
        text=_flatten(str(item.get("Text") or "")),
    )


def _flatten(text: str) -> str:
    """把描述压成一行并截断。

    表格的单元格里放不下换行；typecheck 那条更是整段编译错误。

    Args:
        text: 原始描述

    Returns:
        单行描述
    """
    flat = " ".join(text.split())
    if len(flat) > MAX_TEXT_LENGTH:
        return flat[:MAX_TEXT_LENGTH] + "…"
    return flat


def _associate(
    findings: Sequence[StaticFinding], hunks: Sequence[HunkWithSymbol]
) -> list[StaticFinding]:
    """把每条发现关联到 hunk（弱链接，关联不上就留空）。

    两级：先看行号是否落在某片 hunk 的**新侧区间**里（这条发现就长在这次改动上）；
    不中再看是否落在某片 hunk 归属符号的行区间里（符号级关联：「改的是这个函数，
    问题也报在这个函数里」）。都不中就是 None —— 跨 hunk 的发现、以及落在改动
    之外（比如改了这个文件里另一个函数）的发现都走这条路。

    Args:
        findings: 尚未关联的发现
        hunks: 带归属符号的 hunk 列表

    Returns:
        关联后的发现列表，顺序不变
    """
    by_path: dict[str, list[HunkWithSymbol]] = {}
    for item in hunks:
        by_path.setdefault(item.file_path, []).append(item)

    return [
        replace(finding, hunk_id=_hunk_for_line(finding.line, by_path.get(finding.path, [])))
        for finding in findings
    ]


def _hunk_for_line(line: int, candidates: Sequence[HunkWithSymbol]) -> str | None:
    """在一组候选 hunk 里找行号所属的那片。

    Args:
        line: 行号（被审查版本的坐标）
        candidates: 同一文件里的 hunk

    Returns:
        hunk 编号；不属于任何一片时返回 None
    """
    for item in candidates:
        part = item.part
        if part.new_lines > 0 and part.new_start <= line < part.new_start + part.new_lines:
            return item.hunk_id
    for item in candidates:
        if item.head_lines is not None and item.head_lines[0] <= line <= item.head_lines[1]:
            return item.hunk_id
    return None


def _order_and_number(findings: Sequence[StaticFinding]) -> list[StaticFinding]:
    """排序并编号。

    排序：**优先级降序 → 路径 → 行号 → linter**。优先级在前的另一个好处是编号
    最小的那批就是最要紧的那批 —— 按 token 预算截断时留下的正好是 S-01…S-K
    这一段连续编号，读者不会看到断号。

    Args:
        findings: 关联完毕的发现

    Returns:
        排序并填好编号的新列表
    """
    ordered = sorted(
        findings,
        key=lambda item: (
            PRIORITY_RANK.get(item.priority, len(PRIORITY_RANK)),
            item.path,
            item.line,
            item.linter,
        ),
    )
    return [
        replace(item, finding_id=f"{FINDING_ID_PREFIX}{index:0{FINDING_ID_WIDTH}d}")
        for index, item in enumerate(ordered, start=1)
    ]


def _count_by_linter(findings: Sequence[StaticFinding]) -> str:
    """统计各 linter 的条数，拼成日志用的一句话。

    Args:
        findings: 发现列表

    Returns:
        形如 ``gosec 3 / errcheck 2``；空列表时返回空串
    """
    counts: dict[str, int] = {}
    for item in findings:
        counts[item.linter] = counts.get(item.linter, 0) + 1
    return " / ".join(f"{name} {count}" for name, count in sorted(counts.items()))
