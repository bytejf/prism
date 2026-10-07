"""机器可读结果（JSON 报告）的单元测试。"""

import json
from pathlib import Path
from typing import Any

from prism.core.output.report import build_report, write_report
from prism.models.comment import LOCATE_FAILED, SIDE_LEFT, SIDE_RIGHT, Comment, ReviewStats
from prism.models.cost import CostConfig, UsageTotals
from prism.models.reflection import PotentialMiss
from prism.models.static import PRIORITY_HIGH, StaticFinding

PRICES = CostConfig(cache_hit=0.5, cache_miss=4.0, output=12.0)


def make_comment(**overrides: object) -> Comment:
    """构造测试用评论。

    Args:
        **overrides: 要覆盖的字段

    Returns:
        Comment
    """
    base: dict[str, Any] = {
        "path": "internal/ls/diagnostics.go",
        "severity": "high",
        "category": "边界条件",
        "content": "下标越界",
        "existing_code": "-return parts[len(parts)]",
        "hunk_id": "H-02",
        "locate_level": "hunk",
        "side": SIDE_RIGHT,
        "start_line": 44,
        "end_line": 44,
    }
    base.update(overrides)
    return Comment(**base)


def build(
    comments: list[Comment] | None = None,
    *,
    usage: UsageTotals | None = None,
) -> dict[str, Any]:
    """按默认参数组装一份报告。

    Args:
        comments: 评论列表
        usage: 累计用量

    Returns:
        报告字典
    """
    return build_report(
        base="main",
        head="HEAD",
        project_root="/tmp/repo",
        stats=ReviewStats(
            base_branch="main",
            head_sha="HEAD",
            files_count=3,
            comments_count=len(comments or []),
            duration_seconds=12.5,
            skipped_groups=1,
        ),
        usage=usage or UsageTotals(),
        prices=PRICES,
        comments=comments or [],
        misses=[PotentialMiss(hunk_id="H-07", path="a.go", description="没人审这块")],
        static_gaps=[
            StaticFinding(
                finding_id="S-01",
                path="a.go",
                line=12,
                linter="gosec",
                priority=PRIORITY_HIGH,
                text="G107",
                hunk_id="H-01",
            )
        ],
    )


def test_report_carries_the_fields_the_bench_needs() -> None:
    """报告里要有 bench 侧要用的东西：评论的行号与路径、用量、三类计数。"""
    report = build([make_comment()])

    assert report["base"] == "main"
    assert report["head"] == "HEAD"
    # 评论字段与内部模型同名（不在这一层改名 —— 目标格式的适配在 bench 那一侧）
    comment = report["comments"][0]
    assert comment["path"] == "internal/ls/diagnostics.go"
    assert comment["side"] == SIDE_RIGHT
    assert comment["start_line"] == 44
    assert comment["end_line"] == 44
    assert comment["locate_level"] == "hunk"
    # 缺口信息一个都不能少：少审了几个组与「那些组没问题」在数值上长得一样
    stats = report["stats"]
    assert stats == {
        "files_count": 3,
        "comments_count": 1,
        "duration_seconds": 12.5,
        "skipped_groups": 1,
        "failed_groups": 0,
        "pending_hunks": 0,
    }
    assert report["static_gaps"][0]["finding_id"] == "S-01"
    assert report["misses"][0]["description"] == "没人审这块"


def test_report_keeps_unlocated_comments_without_lines() -> None:
    """未定位的评论也要进报告（行号为 0、带失败原因）。

    bench 那边的匹配对 `null` 行号会跳过行号阶段、只走语义匹配 —— 但「这条评论
    存在」这件事必须能传到评测器。
    """
    report = build(
        [
            make_comment(
                locate_level=LOCATE_FAILED, locate_reason="no_match", start_line=0, end_line=0
            )
        ]
    )

    comment = report["comments"][0]
    assert comment["locate_level"] == LOCATE_FAILED
    assert comment["locate_reason"] == "no_match"
    assert comment["start_line"] == 0


def test_report_keeps_left_side_comments() -> None:
    """删除行的评论（side=LEFT）原样带出去 —— 评测那边要靠它区分两侧。"""
    report = build([make_comment(side=SIDE_LEFT, start_line=25, end_line=27)])

    assert report["comments"][0]["side"] == SIDE_LEFT


def test_usage_and_cost_are_reported() -> None:
    """用量与折算后的花费都要落进报告。"""
    usage = UsageTotals(calls=5, cache_hit=800, cache_miss=200, output=100)

    report = build(usage=usage)
    reported = report["usage"]

    assert reported["calls"] == 5
    assert reported["input_tokens"] == 1000
    assert reported["cache_hit_tokens"] == 800
    assert reported["cache_miss_tokens"] == 200
    assert reported["cost_yuan"] == 0.0024
    assert reported["missing_usage_calls"] == 0


def test_missing_usage_is_reported_not_hidden() -> None:
    """拿不到用量的调用数要报出来 —— 非零意味着那些 token 只是下界。"""
    report = build(usage=UsageTotals(calls=1, missing=2))

    assert report["usage"]["missing_usage_calls"] == 2


def test_write_report_creates_dirs_and_keeps_chinese(tmp_path: Path) -> None:
    """写文件：父目录自动创建，中文不转义（人能直接看）。"""
    target = tmp_path / "outputs" / "aacr-bench-go" / "x-y-1.json"

    write_report(target, build([make_comment()]))

    text = target.read_text(encoding="utf-8")
    assert "下标越界" in text, "ensure_ascii=False —— 中文原样写，不转成 \\uXXXX"
    assert json.loads(text)["comments"][0]["path"] == "internal/ls/diagnostics.go"
