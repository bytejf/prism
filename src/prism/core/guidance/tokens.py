"""提示词 token 估算。

**一个估算函数，两个消费方**：分组拿它决定「这个 hunk 还装不装得下」（设计 18 号
§2.4 的前提就是分组要把静态发现的 token 一起算），提示词的干跑入口拿它给人看
「这一组大概多大」。两处共用一份，是为了不让「成本模型」和「实际渲染」漂开 ——
比精确更重要的是**只有一处**。

估算口径：字符数 ÷ 2.5（设计 §2.4 的取舍）。中文约 1.5 字/token、英文约
4 字符/token，取中间值；误差 ±20%，由预算余量吸收。要精确到 DeepSeek 的分词器
就得引入 tiktoken，而它算的也不是同一个分词器。
"""

# 字符数 → token 的除数
CHARS_PER_TOKEN = 2.5


def estimate_tokens(text: str) -> int:
    """粗估一段文本的 token 数（字符数 ÷ 2.5，向上取整）。

    Args:
        text: 任意文本

    Returns:
        token 估算值；空串为 0
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1
