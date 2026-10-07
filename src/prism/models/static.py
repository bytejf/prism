"""静态分析（golangci-lint）相关的数据模型。

静态发现**不是评论**：它不进最终输出、不进定位、不参与排序，只是提示词里的一句
线索（见设计文档 18 号 §2.1）。所以这里的模型只描述「工具说了什么」，
不带任何「我们认为它成立」的字段。
"""

from dataclasses import dataclass, field

# 工具标记（给人看的「优先」档位），借用了 comment 的 severity 取值。
#
# 它是**我们按 linter 名字映射出来的**，不是 golangci-lint 自己的 severity ——
# 实测那个字段大面积为空（errcheck 全是 ""，gosec 全是 "medium"，与规则严重性
# 无关），直接用它等于把优先级交给工具的内部实现。
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

# 优先级降序用的档位。两个消费方各取所需：分析阶段按它给发现编号，提示词组装
# 按它给 hunk 排展示顺序 —— 放这里是为了让两处不可能各写一份
PRIORITY_RANK: dict[str, int] = {PRIORITY_HIGH: 0, PRIORITY_MEDIUM: 1, PRIORITY_LOW: 2}

# 发现关联不上 hunk 时的占位符（提示词表格与 CLI 干跑输出共用）
NO_HUNK_MARK = "—"


@dataclass(frozen=True)
class StaticFinding:
    """一条静态发现。

    Attributes:
        finding_id: 编号（如 "S-01"），全局连续分配，模型在评论正文里引用它
        path: 文件路径（仓库内相对路径，被审查版本的坐标）
        line: 行号。**是被审查版本的坐标** —— 分析跑在 head 的检出上，
            与 file_read 读到的是同一套；纯删除的文件不会有发现（它不在检出里）
        linter: 产生这条发现的 linter 名
        priority: 工具标记（high / medium / low），由 linter 名映射而来
        text: 工具给的描述原文（可能是多行，注入前会被压成一行）
        hunk_id: 关联的 hunk 编号；落在改动行之外（跨 hunk 或改动行附近之外）
            时为 None。**只是弱链接**，不保证存在
    """

    finding_id: str
    path: str
    line: int
    linter: str
    priority: str
    text: str
    hunk_id: str | None = None


@dataclass
class StaticReport:
    """一次静态分析的结果。

    区分「跑了但没发现」与「没跑成」是**必须的**：两者都不注入提示词，但对人的
    含义完全相反 —— 前者是干净的，后者是「这次审查没有静态辅助」。合成一个空
    列表就等于把后者静默成前者（同 `file_read_diff` 索引为空要报错）。

    Attributes:
        findings: 发现列表（只含可审查文件），按「优先级降序 + 路径 + 行号」排序
        skip_reason: 非空表示这次没跑成，值是人看的说明；跑到一半的发现一律丢弃
            （半份结果比没有结果更坏：模型会以为没报的就是干净的）
        elapsed_seconds: 子进程耗时（秒）。跳过时为 0
        raw_count: golangci-lint 报出的总条数（含被我们过滤掉的），
            用于日志对账 —— 只看 findings 数会把「全被过滤」误读成「工具没发现问题」
    """

    findings: list[StaticFinding] = field(default_factory=list)
    skip_reason: str | None = None
    elapsed_seconds: float = 0.0
    raw_count: int = 0

    @property
    def skipped(self) -> bool:
        """这次静态分析是否被跳过。"""
        return self.skip_reason is not None
