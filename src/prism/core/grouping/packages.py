"""按包路径分组。

V1 只用目录路径作为分组依据：不调调用图、不做装箱算法、不做 token 预算。
同一个包的文件倾向于一起看，分组结果也是后续并发审查的最小单位。
"""

import logging
from collections import defaultdict
from pathlib import PurePosixPath

from prism.models.ast import HunkWithSymbol
from prism.models.grouping import Group

logger = logging.getLogger(__name__)

# 分组 ID 前缀与补零宽度
GROUP_ID_PREFIX = "group-"
GROUP_ID_WIDTH = 3

# 根目录下文件的包路径
ROOT_PACKAGE = "."


def group_by_package(hunks: list[HunkWithSymbol]) -> list[Group]:
    """把 hunk 按所在包（目录）分组。

    组按包路径升序排列；组内文件按路径升序，hunk 按文件路径 + 行号升序。
    同一个包无论多少文件都不再拆分（V1 不做装箱）。

    Args:
        hunks: 带归属符号的 hunk 列表

    Returns:
        Group 列表，按包路径升序；无输入时返回空列表
    """
    by_package: dict[str, list[HunkWithSymbol]] = defaultdict(list)
    for item in hunks:
        by_package[package_path(item.file_path)].append(item)

    groups: list[Group] = []
    for index, package in enumerate(sorted(by_package), start=1):
        ordered = sorted(by_package[package], key=_hunk_sort_key)
        groups.append(
            Group(
                group_id=f"{GROUP_ID_PREFIX}{index:0{GROUP_ID_WIDTH}d}",
                package_path=package,
                files=sorted({item.file_path for item in ordered}),
                hunks=ordered,
            )
        )

    logger.info("分组完成：%d 个包，%d 个 hunk", len(groups), len(hunks))
    for group in groups:
        logger.debug(
            "%s %s：%d 个文件，%d 个 hunk",
            group.group_id,
            group.package_path,
            len(group.files),
            len(group.hunks),
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


def _hunk_sort_key(item: HunkWithSymbol) -> tuple[str, int, int]:
    """组内 hunk 的排序键。

    纯删除的 hunk 新文件行号为 0，用旧文件起始行兜底保证顺序稳定。

    Args:
        item: 带符号的 hunk

    Returns:
        (文件路径, 新文件起始行, 旧文件起始行)
    """
    return (item.file_path, item.hunk.new_start, item.hunk.old_start)
