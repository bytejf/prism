"""反思的定标脚本：量「该删的删没删、不该删的动没动」。

**手动跑，不进 `pytest` 收集**（文件名没有 `test_` 前缀）。理由有两个：

1. **它要花钱**（真实 API 调用），不该混进任何一次顺手跑的测试里
2. **它的期望值是构造出来的**，不是稳定的断言 —— 模型的输出有随机性，
   拿它当回归测试会偶发红灯

## 为什么要这样一个脚本

反思是全流程里**唯一会销毁东西**的一步：删掉一条对的评论，真实发现就静默
消失了。单测能钉住「编号对不上不许删」这类代码里的闸门，钉不住**提示词写得
对不对** —— 而提示词是删错评论的主要来源。

所以这里拿**真实仓库的真实 diff**，混进几条构造好的评论，看模型删的是不是
恰好该删的那几条：

    F1  「新增了 recover() 但吞了堆栈」      → 该删（Ground A：diff 里没有 recover）
    F2  「Trim 仍然会去掉首尾空白」          → 该删（Ground B：diff 里是 `+ return s`）
    F3  「parts 已经不再使用」               → 该删（Ground B：diff 里 `+ return parts[...]`）
    P1  「Label 里新加的这个 mutex 会死锁」  → **该留**（主题是并发，保护性否决）
    R1  「parts[len(parts)] 越界」           → 该留（真问题）

F1~F3 量**召回**，P1/R1 量**精度**。P1 尤其关键：它被 diff 直接推翻，但主题
落在保护性否决里 —— 删了就说明那条规则没生效。

用法::

    uv run python tests/manual/calibrate_reflection.py
"""

import os
import sys
from pathlib import Path

from prism.config.settings import settings
from prism.core.ast.mapper import map_hunks_to_symbols
from prism.core.diff.loader import load_diffs
from prism.core.filter.rules import filter_files
from prism.core.grouping.packages import group_by_package
from prism.core.llm.client import LLMClient
from prism.core.reflection.reflect import reflect
from prism.models.comment import LOCATE_HUNK, SIDE_RIGHT, Comment

REPO = Path(os.getenv("PRISM_MIN_TEST_DIR", Path.home() / "Projects/python/prism-min-test"))


def build(hunk_id: str, content: str, existing_code: str, category: str = "逻辑错误") -> Comment:
    """按某个真实 hunk 的位置造一条评论。

    位置、路径都取自真实 hunk，只有正文是构造的 —— 这样反思看到的材料与
    真实运行时一模一样，变的只有评论内容本身。

    Args:
        hunk_id: 挂在哪个 hunk 上
        content: 评论正文
        existing_code: 引用的代码片段
        category: 问题类别

    Returns:
        Comment 对象
    """
    item = next(h for h in HUNKS if h.hunk_id == hunk_id)
    return Comment(
        path=item.file_path,
        severity="high",
        category=category,
        content=content,
        existing_code=existing_code,
        hunk_id=hunk_id,
        locate_level=LOCATE_HUNK,
        side=SIDE_RIGHT,
        start_line=item.part.new_start,
        end_line=item.part.new_start,
    )


def main() -> int:
    """跑一次定标并打印结果。

    Returns:
        0 表示删的恰好是该删的那几条
    """
    group = GROUPS[0]
    cases = [
        build(
            "H-01",
            "这个 hunk 新增了 recover() 兜住 panic，但 recover 把堆栈吞了，出问题查不到现场。",
            "defer func() { recover() }()",
        ),
        build(
            "H-04",
            "Trim 会去掉首尾空白后再返回，保持原有契约不变。",
            "+	return s",
        ),
        build(
            "H-02",
            "parts 这个变量在这段里已经不再使用了，可以删掉。",
            "+	return parts[len(parts)]",
        ),
        build(
            "H-03",
            "Label 里新加的这个 mutex 会跟调用方的锁形成死锁。",
            "+	s.mu.Lock()",
            category="并发安全",
        ),
        build(
            "H-02",
            "parts[len(parts)] 越界：Split 至少返回 1 个元素，但下标最大是 len-1，会 panic。",
            "+	return parts[len(parts)]",
        ),
    ]
    labels = ["F1", "F2", "F3", "P1（保护性主题）", "R1（真问题）"]

    outcome = reflect(LLMClient(settings.llm).model, group, cases)

    deleted = {item.comment_id for item in outcome.removed}
    print(f"失败：{outcome.failure}")
    for index, (label, item) in enumerate(zip(labels, cases, strict=True), start=1):
        comment_id = f"C-{index:02d}"
        verdict = "删" if comment_id in deleted else "留"
        ground = next((r.ground for r in outcome.removed if r.comment_id == comment_id), "-")
        print(f"  {verdict}（{ground}）{label}：{item.content[:38]}")
    for miss in outcome.misses:
        print(f"  漏报 {miss.hunk_id} {miss.path}：{miss.description[:40]}")

    expected = {"C-01", "C-02", "C-03"}
    ok = deleted == expected
    print(f"\n应当删除 {sorted(expected)}，实际 {sorted(deleted)} → {'✓' if ok else '✗'}")
    return 0 if ok else 1


DIFFS = filter_files(load_diffs(REPO, "main", "HEAD")).reviewable
HUNKS = map_hunks_to_symbols(REPO, "main", "HEAD", DIFFS)
GROUPS = group_by_package(HUNKS)

if __name__ == "__main__":
    sys.exit(main())
