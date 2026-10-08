"""流水线对账：把「hunk 不重不漏 / 预算账 / 提示词自洽 / 定位同源」逐条量出来。

**这不是断言套件，是花钱跑评测之前的对账**：拿一个真实仓库，把四项逐条量出来，
任何一项不对就先别跑 LLM（白烧 token 之前先发现）。

    跑法（不花钱、秒级）：
        uv run python tests/manual/audit_pipeline.py ~/Projects/python/prism-tests

    看什么：
        1) hunk 不重不漏：跨组 hunk 数 == 映射结果数，且无重复
        2) 预算账 vs 真实渲染：偏差应在取整误差内（账是装箱的依据，漂了预算就是空话）
        3) 提示词自洽：每个 hunk 一个标签、静态表格行数 == 注入条数、
           static_hint 不引用没注入的编号、没有发现时不出现静态段
        4) 定位同源：定位后的评论 path 必须与它 hunk_id 指向的文件一致

    脚本里把 `max_group_tokens` 压到 3000 —— 否则走查仓库只会有一组，
    装箱那条路径根本没被走过。**这四项都不会花钱**（不构造 LLM 客户端）。
"""

import dataclasses
import re
import sys
from pathlib import Path

from prism.config.settings import settings
from prism.core.agent.agent import initial_state
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import fixed_prompt_tokens, group_by_package, group_tokens
from prism.core.guidance import templates as tpl
from prism.core.guidance.tokens import estimate_tokens
from prism.core.output.locate import locate_group
from prism.models.comment import Comment  # noqa: F401  （仅供类型参考）
from prism.models.tool import CommentInput

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Projects/python/prism-tests").expanduser()
BUDGET = 3000  # 故意压小：逼出多组，才能验装箱
BASE, HEAD = "main", "HEAD"


def main() -> int:
    """跑完全部对账。

    Returns:
        退出码：0 全过，1 有对账失败
    """
    # 把预算压到 3000 逼出多组（否则走查仓库只有一组，装箱那条路径没被走到）。
    # 直接替换模块上的 settings 名字 —— 分组在每个判断处现读它
    import prism.core.grouping.packages as pkg

    pkg.settings = dataclasses.replace(  # type: ignore[attr-defined]
        settings, review=dataclasses.replace(settings.review, max_group_tokens=BUDGET)
    )

    reviewable = filter_files(load_diffs(ROOT, BASE, HEAD)).reviewable
    mapped = map_hunks_to_symbols(ROOT, BASE, HEAD, reviewable)
    groups = group_by_package(mapped, [])
    print(
        f"仓库 {ROOT.name}：{len(reviewable)} 个可审查文件 / {len(mapped)} 个 hunk → {len(groups)} 组\n"
    )

    failures: list[str] = []

    # ---------- 1) hunk 不重不漏 ----------
    seen = [item.hunk_id for group in groups for item in group.hunks]
    print("1) hunk 不重不漏")
    print(f"   跨组 hunk 数 {len(seen)} vs 映射结果 {len(mapped)}；去重后 {len(set(seen))}")
    if len(seen) != len(mapped) or len(set(seen)) != len(seen):
        failures.append("hunk 有重复或丢失")
    for group in groups:
        paths = {item.file_path for item in group.hunks}
        if set(group.files) != paths:
            failures.append(
                f"{group.group_id} 的 files 与 hunks 不一致：{group.files} vs {sorted(paths)}"
            )
        if [item.file_path for item in group.hunks] != sorted(
            item.file_path for item in group.hunks
        ):
            failures.append(f"{group.group_id} 组内 hunk 没按文件排序")
    print(f"   组内 files 与 hunks 一致、按文件排序：{'✓' if not failures else '✗'}\n")

    # ---------- 2) 预算账 vs 真实渲染 ----------
    print("2) 预算账（fixed + group_tokens）vs 真实 user 消息")
    print(f"   {'组':<12}{'账上':>7}{'真实':>7}{'差':>7}{'差%':>7}")
    worst_gap = 0.0
    for group in groups:
        estimated = fixed_prompt_tokens() + group_tokens(group.hunks, [])  # = SYSTEM+要求+diff
        # **可比口径**：账上 = SYSTEM + 要求块 + diff；真实整条请求 = SYSTEM + user 消息，
        # 而 user 消息里已经含要求块。所以这一侧只能加 SYSTEM，**不能再加 fixed**
        # （第一版加了 fixed、第二版忘了加 SYSTEM —— 两次都是我把口径搞错了，
        # 差的常数恰好分别等于要求块与 system 段，很容易误判成代码有 bug）
        real = estimate_tokens(tpl.SYSTEM) + estimate_tokens(
            str(initial_state(group)["messages"][0].content)
        )
        gap = real - estimated
        ratio = gap / max(estimated, 1) * 100
        worst_gap = max(worst_gap, abs(ratio))
        print(f"   {group.group_id:<12}{estimated:>7}{real:>7}{gap:>7}{ratio:>6.1f}%")
    print(f"   最大偏差 {worst_gap:.1f}%（账 < 真实 = 会超预算）")
    if worst_gap > 10:
        failures.append(f"预算账与真实渲染偏差 {worst_gap:.1f}%")
    print()

    # ---------- 3) 提示词自洽 ----------
    print("3) 提示词自洽（每个 hunk 出现一次；静态表格行数 = 注入条数）")
    findings = [
        __import__("prism.models.static", fromlist=["StaticFinding"]).StaticFinding(
            finding_id=f"S-{index:02d}",
            path=group.files[0],
            line=1,
            linter="gosec",
            priority="high",
            text="G107: 假发现",
            hunk_id=group.hunks[0].hunk_id,
        )
        for index, group in enumerate(groups, start=1)
    ]
    prompt_failures: list[str] = []
    for group in groups:
        related = [item for item in findings if item.path in set(group.files)]
        text = str(initial_state(group, related)["messages"][0].content)
        ids = re.findall(r'<hunk id="(H-\d+)"', text)
        if len(ids) != len(group.hunks) or len(set(ids)) != len(ids):
            prompt_failures.append(
                f"{group.group_id} 的 hunk 标签数 {len(ids)} != {len(group.hunks)}（或重复）"
            )
        rows = [line for line in text.splitlines() if line.startswith("| S-")]
        if len(rows) != len(related):
            prompt_failures.append(
                f"{group.group_id} 静态表格 {len(rows)} 行 != 注入 {len(related)} 条"
            )
        hints = re.findall(r'static_hint="([^"]+)"', text)
        for hint in hints:
            for one in hint.split():
                if one not in {item.finding_id for item in related}:
                    prompt_failures.append(
                        f"{group.group_id} 的 static_hint 引用了没注入的编号 {one}"
                    )
        if related and "### 静态分析提示" not in text:
            prompt_failures.append(f"{group.group_id} 有静态发现却没有静态段")
        if not related and "### 静态分析提示" in text:
            prompt_failures.append(f"{group.group_id} 没有静态发现却出现了静态段")
    failures.extend(prompt_failures)
    print(
        f"   {'✓' if not prompt_failures else '✗ ' + '；'.join(prompt_failures)}"
        f"（{len(groups)} 组逐组核对）\n"
    )

    # ---------- 4) 定位同源：评论的 path 必须与它 hunk_id 的 path 一致 ----------
    print("4) 定位同源（评论 path 与 hunk_id 指向同一文件）")
    path_of_hunk = {item.hunk_id: item.file_path for group in groups for item in group.hunks}
    checked = 0
    locate_failures: list[str] = []
    for group in groups:
        # 故意报一个「本组的 hunk 但填了别的文件的 path」的评论 —— 定位应把它挂回自己文件
        comments = {
            group.hunks[0].hunk_id: [
                CommentInput(
                    severity="high",
                    category="bug",
                    content="核对用",
                    existing_code=_first_added_line(group.hunks[0].part.content),
                )
            ]
        }
        for located in locate_group(group, comments):
            checked += 1
            expected = path_of_hunk.get(located.hunk_id)
            if expected and located.path != expected:
                locate_failures.append(
                    f"定位后的评论 path={located.path} 与 hunk {located.hunk_id} 的 {expected} 不一致"
                )
    failures.extend(locate_failures)
    print(f"   核了 {checked} 条定位结果：{'✓' if not locate_failures else '✗'}\n")

    # ---------- 收尾 ----------
    if failures:
        print("对账失败：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("全部对账通过 ✓")
    return 0


def _first_added_line(content: str) -> str:
    """取 hunk 正文里第一行新增行（去掉 `+`）。

    Args:
        content: hunk 正文

    Returns:
        新增行文本；没有新增行时返回空串
    """
    for line in content.splitlines():
        if line.startswith("+"):
            return line[1:].strip()
    return ""


if __name__ == "__main__":
    sys.exit(main())
