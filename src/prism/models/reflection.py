"""反思（复核）相关的数据模型。"""

from dataclasses import dataclass


@dataclass
class PotentialMiss:
    """疑似漏报 —— 反思时观察到「diff 里有改动没被任何评论覆盖」。

    **它不是评论**：不进 `comments`、不进 warnings。反思「只删不增」这条约束
    不因为漏报信号而松动 —— 它只是一句提示，供人判断要不要人工补审。

    Attributes:
        hunk_id: 相关 hunk 编号
        path: 该 hunk 的文件路径。**由 hunk_id 查出，不让模型自己写** —— 编码
            坐标的字段只留一个来源，与评论的 `path` 同源规则一致
        description: 疑似漏报的描述
    """

    hunk_id: str
    path: str
    description: str
