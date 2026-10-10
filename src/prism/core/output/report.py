"""机器可读的审查结果（JSON）。

终端渲染是**给人看**的，这份是**给程序读**的：bench 驱动（`prism-bench` 里接官方
AACR-Bench harness 的那个）要靠它把评论交给评测器算准确率/召回率。两者都从同样的
数据出发，只是受众不同 —— 所以这里不复用 `TerminalRenderer` 的字符串，而是导出
结构化字段。

设计：`acr-min/06-可观测性/21-可观测性设计.md` §4.2（`outputs/<repo>-<pr>.json`）
"""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from prism.core.agent.budget import cost_yuan
from prism.models.comment import Comment, ReviewStats
from prism.models.cost import CostConfig, UsageTotals
from prism.models.reflection import PotentialMiss
from prism.models.static import StaticFinding

logger = logging.getLogger(__name__)


def build_report(
    *,
    base: str,
    head: str,
    project_root: str,
    stats: ReviewStats,
    usage: UsageTotals,
    prices: CostConfig,
    comments: list[Comment],
    misses: list[PotentialMiss],
    static_gaps: list[StaticFinding],
    invalid_rule_refs: list[str] | None = None,
) -> dict[str, Any]:
    """把一次审查的产出组装成 JSON 可序列化的字典。

    **字段名与内部模型一致**（`locate_level` / `side` / `start_line`…）：这是一份
    忠实转储，不是又一层契约。评测器需要的 `note` / `from_line` 之类的改名放在
    bench 那一侧的适配层做（那里才知道目标格式长什么样）。

    Args:
        base: base 分支 / 版本
        head: head 分支 / 版本
        project_root: 被审查仓库路径
        stats: 审查统计
        usage: 累计用量（含反思）
        prices: 价格表（折算花费）
        comments: 保留的评论（含未定位的）
        misses: 疑似漏报信号
        static_gaps: 没被引用的高优先级静态发现
        invalid_rule_refs: 评论里引用了、但规则库里没有的规则代号

    Returns:
        可直接 `json.dump` 的字典
    """
    return {
        "prism_version": _version(),
        "base": base,
        "head": head,
        "project_root": project_root,
        "stats": {
            "files_count": stats.files_count,
            "comments_count": stats.comments_count,
            "duration_seconds": stats.duration_seconds,
            "skipped_groups": stats.skipped_groups,
            "failed_groups": stats.failed_groups,
            "pending_hunks": stats.pending_hunks,
        },
        "usage": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "cache_hit_tokens": usage.cache_hit,
            "cache_miss_tokens": usage.cache_miss,
            "output_tokens": usage.output,
            # 拿不到用量的调用次数。**非零意味着上面的 token 只是下界**，
            # 与终端输出里那句「⚠️ N 次调用拿不到用量」是同一件事
            "missing_usage_calls": usage.missing,
            "cost_yuan": round(cost_yuan(usage, prices), 4),
        },
        "comments": [_comment(item) for item in comments],
        "misses": [
            {"hunk_id": item.hunk_id, "path": item.path, "description": item.description}
            for item in misses
        ],
        "invalid_rule_refs": list(invalid_rule_refs or ()),
        "static_gaps": [
            {
                "finding_id": item.finding_id,
                "path": item.path,
                "line": item.line,
                "linter": item.linter,
                "priority": item.priority,
                "hunk_id": item.hunk_id,
                "text": item.text,
            }
            for item in static_gaps
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """把报告写成 JSON 文件（父目录不存在时创建）。

    Args:
        path: 目标文件路径
        report: `build_report` 的产出
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON 报告已写入：%s（%d 条评论）", path, len(report.get("comments", [])))


def _comment(item: Comment) -> dict[str, Any]:
    """一条评论的 JSON 形态。

    Args:
        item: 评论

    Returns:
        字典
    """
    return {
        "path": item.path,
        "severity": item.severity,
        "category": item.category,
        "content": item.content,
        "existing_code": item.existing_code,
        "hunk_id": item.hunk_id,
        "locate_level": item.locate_level,
        "side": item.side,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "locate_reason": item.locate_reason,
        "suggestion_code": item.suggestion_code,
        # 两个引用字段**分开**出现，不要合并成一个字符串：消费方（评测、
        # 面板）要能分别按规则 / 静态发现统计
        "rule_ref": item.rule_ref,
        "finding_ref": item.finding_ref,
    }


def _version() -> str:
    """取包版本（拿不到时返回空串）。

    Returns:
        版本号
    """
    try:
        return version("prism")
    except PackageNotFoundError:  # pragma: no cover - 只在未安装时发生
        return ""
