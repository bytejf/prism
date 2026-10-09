"""按包路径分组，并按 token 预算切成审查单元。

**算法只有一条**：把所有 hunk 排成一列（**同一个包的挨在一起**），从上往下一路
合并进当前组，装不下就封口、开新组接着合并。

    [pkg/a 的 hunk...][pkg/b 的 hunk...][pkg/c 的 hunk...]
    └──── group-001（装满预算）────┘└── group-002 ──┘

所以：

- **包只决定顺序，不决定边界**。一个包可能被拆到几组里（装不下时），几个小包也
  可能落在同一组里（合起来都没装满时）—— 前者是「切」，后者是「并」，同一条规则
  的两个方向。**包挨在一起**这一点是刻意保住的：它让每一组拿到的是连续一段 hunk，
  相邻的改动仍然在一块儿
- **`Group.package_path` 因此只是「首片所在的包」**，给人看的完整形态用
  `package_label`（跨包时显示成 `pkg/a → pkg/b`）

三个取舍：

- **切分粒度是文件，不是 hunk**：一个文件的 hunk 默认整块进同一组。只有当**这个
  文件自己**就超预算（改了几百行的大文件）时，才退化成按 hunk 切并记 WARNING ——
  到那一层已经没有更细的边界可用了
- **成本用真实渲染量出来**，不另写一套「每个 hunk 多少 token」的估算模型：
  试算时直接调 `build_diff_block` / `build_static_block` 渲染候选内容再数字符。
  代价是装箱过程是 O(k²) 次渲染（k = 一组的 hunk 数，实测最大的组 95 个），
  换来的是**不会出现「成本模型算得下、真渲染超了」这种漂移**
- **预算同时管住了两件事的反面**：一组塞得越多，固定开销（1111 token）摊得越薄、
  反思调用越少；但一组的 hunk 越多，那一次会话也越长。预算只约束**起手提示词**，
  不约束会话长度 —— 长度由 hunk 数与探索轮次决定，由钱 / 时间预算兜住
"""

import logging
from collections.abc import Sequence
from pathlib import PurePosixPath

from prism.config.settings import settings
from prism.core.guidance.builder import build_diff_block, build_static_block, build_system
from prism.core.guidance.tokens import estimate_tokens
from prism.models.ast import HunkWithSymbol
from prism.models.grouping import Group
from prism.models.static import StaticFinding

logger = logging.getLogger(__name__)

# 分组 ID 前缀与补零宽度
GROUP_ID_PREFIX = "group-"
GROUP_ID_WIDTH = 3

# 根目录下文件的包路径
ROOT_PACKAGE = "."

# 跨包组的显示连接符（`pkg/a → pkg/b`）
PACKAGE_ARROW = " → "


def group_by_package(
    hunks: list[HunkWithSymbol], findings: Sequence[StaticFinding] = ()
) -> list[Group]:
    """把 hunk 按包分组，再按 token 预算装箱。

    组按包路径升序，组内 hunk 保持「文件路径 + 行号」升序；从上往下一路合并，
    装不下就封口开下一组。于是同一组的包路径必然是连续的一段（`package_path`
    记首片所在的包，完整形态见 `package_label`）。

    Args:
        hunks: 带归属符号的 hunk 列表
        findings: 全量静态发现。**参与预算**：某个 hunk 关联的发现要占额度，
            所以静态发现多的组会更早被切开

    Returns:
        Group 列表，按顺序编号；无输入时返回空列表
    """
    ordered = sorted(hunks, key=_hunk_sort_key)
    budget = settings.review.max_group_tokens
    pieces = _pieces(ordered, findings, budget)
    groups = _pack(pieces, findings, budget)

    logger.info(
        "分组完成：%d 个包，%d 个 hunk → %d 组（预算 %d token/组）",
        len({package_path(item.file_path) for item in ordered}),
        len(hunks),
        len(groups),
        budget,
    )
    for group in groups:
        tokens = group_tokens(group.hunks, findings)
        logger.debug(
            "%s %s：%d 个文件，%d 个 hunk，约 %d token（额度 %d）",
            group.group_id,
            package_label(group),
            len(group.files),
            len(group.hunks),
            tokens,
            budget,
        )
    return groups


def package_path(file_path: str) -> str:
    """从文件路径推导包路径。

    Args:
        file_path: 仓库内相对路径，如 ``internal/agent/agent.go``

    Returns:
        包路径，如 ``internal/agent``；根目录下的文件返回 ``.``
    """
    return str(PurePosixPath(file_path).parent)


def package_label(group: Group) -> str:
    """给一组一个可读的包标签。

    跨包组（几个小包合进同一组、或者一个包的尾巴与下一个包的头挨在一起）显示成
    ``pkg/a → pkg/b`` —— 只显示首片那个包会让人以为这一组都在一个包里。

    组按包排序、装箱保序，所以同一组的包路径必然是**连续区间**，首尾一拼就是全部。

    Args:
        group: 审查单元

    Returns:
        包路径；跨包时是 ``首 → 尾``；没有文件时返回 ``ROOT_PACKAGE``
    """
    packages = sorted({package_path(path) for path in group.files})
    if not packages:
        return ROOT_PACKAGE
    if len(packages) == 1:
        return packages[0]
    return f"{packages[0]}{PACKAGE_ARROW}{packages[-1]}"


def group_tokens(hunks: Sequence[HunkWithSymbol], findings: Sequence[StaticFinding]) -> int:
    """量一组提示词（不含 system 与要求块）的 token 数。

    直接渲染再数字符 —— 与真正发出去的提示词是同一份代码路径。

    Args:
        hunks: 这一组的 hunk
        findings: 全量静态发现（内部按文件挑出属于这些 hunk 的）

    Returns:
        估算 token 数
    """
    files = sorted({item.file_path for item in hunks})
    if not files:
        return 0
    related = _findings_for_files(files, findings)
    return estimate_tokens(build_diff_block(_probe(hunks), related)) + estimate_tokens(
        build_static_block(related)
    )


def group_cost(hunks: Sequence[HunkWithSymbol], findings: Sequence[StaticFinding]) -> int:
    """一组的**总**成本：固定开销 + 内容。

    **装箱的每一处判断都必须用它**，不要拿 `group_tokens` 单独跟预算比 ——
    漏掉固定开销的写法会让「文件自己放得下」与「装进组里放不下」给出两个答案，
    结果就是一个超预算的组带着 ⚠️ 出现在干跑输出里（实测踩到过）。

    Args:
        hunks: 这一组的 hunk
        findings: 全量静态发现

    Returns:
        估算 token 数
    """
    return fixed_prompt_tokens() + group_tokens(hunks, findings)


def fixed_prompt_tokens() -> int:
    """量一组的固定开销：system 段（角色 + 主任务 + 工作方式 + 硬约束 + 收尾）。

    它每轮都要重发，所以是**每一轮的地板价** —— 组数越多，这笔钱付得越多。
    通用约束整段搬进 system 之后，这里就只剩一项（原来还要加 user 里的要求块）。

    Returns:
        估算 token 数
    """
    return estimate_tokens(build_system())


def _findings_for_files(
    files: Sequence[str], findings: Sequence[StaticFinding]
) -> list[StaticFinding]:
    """挑出属于这些文件的静态发现（与注入那一侧同一个判据）。

    Args:
        files: 文件路径
        findings: 全量静态发现

    Returns:
        相关发现，保持全量列表顺序（已按优先级降序）
    """
    wanted = set(files)
    return [item for item in findings if item.path in wanted]


def _probe(hunks: Sequence[HunkWithSymbol]) -> Group:
    """构造一个只为试算用的临时组。

    Args:
        hunks: 候选 hunk

    Returns:
        临时 Group（编号与包路径都是占位，不会外泄）
    """
    return Group(
        group_id="probe",
        package_path="probe",
        files=sorted({item.file_path for item in hunks}),
        hunks=list(hunks),
    )


def _pieces(
    hunks: Sequence[HunkWithSymbol], findings: Sequence[StaticFinding], budget: int
) -> list[list[HunkWithSymbol]]:
    """把有序 hunk 切成「不可再切的片」。

    一个文件一片（它的 hunk 必须待在一起）；**只有这个文件自己就超预算时**才退化成
    每个 hunk 一片。切完的片仍然保持原顺序。

    Args:
        hunks: 已排序的 hunk
        findings: 全量静态发现
        budget: 单组预算

    Returns:
        片的列表
    """
    by_file: dict[str, list[HunkWithSymbol]] = {}
    for item in hunks:
        by_file.setdefault(item.file_path, []).append(item)

    pieces: list[list[HunkWithSymbol]] = []
    for path, items in by_file.items():
        cost = group_cost(items, findings)
        if cost <= budget:
            pieces.append(items)
            continue
        logger.warning(
            "文件 %s 自己就超过单组预算（约 %d > %d token）：按 hunk 切成 %d 片，它的 hunk 会跨组",
            path,
            cost,
            budget,
            len(items),
        )
        pieces.extend([item] for item in items)
    return pieces


def _pack(
    pieces: Sequence[Sequence[HunkWithSymbol]],
    findings: Sequence[StaticFinding],
    budget: int,
) -> list[Group]:
    """把片按顺序装进组：装不下就封口，下一片开新组。

    Args:
        pieces: 有序的片
        findings: 全量静态发现（参与预算）
        budget: 单组预算

    Returns:
        组装好的审查单元列表
    """
    packed: list[list[HunkWithSymbol]] = []
    current: list[HunkWithSymbol] = []
    for piece in pieces:
        # 唯一的封口条件：这一片装不下了。包只决定顺序，不决定边界
        if current and group_cost([*current, *piece], findings) > budget:
            packed.append(current)
            current = []
        current.extend(piece)
        if group_cost(current, findings) > budget:
            # 一片（单文件或单 hunk）自己就超：它单独成组，只提醒不改预算 ——
            # 到这一层已经没有更细的边界，硬拆只会把上下文切碎
            logger.warning(
                "单组预算 %d 装不下这一片（约 %d token，%d 个 hunk）：它单独成组",
                budget,
                group_cost(current, findings),
                len(current),
            )
    if current:
        packed.append(current)

    return [
        Group(
            group_id=f"{GROUP_ID_PREFIX}{index:0{GROUP_ID_WIDTH}d}",
            package_path=package_path(items[0].file_path),
            files=sorted({item.file_path for item in items}),
            hunks=list(items),
        )
        for index, items in enumerate(packed, start=1)
    ]


def _hunk_sort_key(item: HunkWithSymbol) -> tuple[str, str, int, int]:
    """全量排序键：包 → 文件路径 → 新文件起始行 → 旧文件起始行。

    先按包排是为了让**同一个包的 hunk 连续**（装箱才不会把包打散）；纯删除的 hunk
    新文件行号为 0，用旧文件起始行兜底保证顺序稳定。

    Args:
        item: 带符号的 hunk

    Returns:
        (包路径, 文件路径, 新文件起始行, 旧文件起始行)
    """
    return (
        package_path(item.file_path),
        item.file_path,
        item.part.new_start,
        item.part.old_start,
    )
