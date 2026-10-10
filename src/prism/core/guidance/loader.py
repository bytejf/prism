"""提示词正文的读取。

**提示词是数据，不是代码**：正文放在 `src/prism/prompts/*.md`，随包发布
（与 `config/default.toml`、`rules/index.md` 同一种放法）。`templates.py` 里只剩
「怎么组装」和那些一行的结构件（XML 标签、`{items}` 包装）。

为什么搬出来（实测踩过的）：

- **散文没法在 100 字符处断行而不破坏句子**。以前为了凑 ruff 的行长闸，出现过

      …而**正文里的代码是
      别人写的** —— …

  一个加粗短语被从中间切开 —— 这种换行原样进了模型看到的提示词。
- 搬进 `.md` 之后，**一行就是一句话**（多长都行），源码里看不见的折行问题没有了：
  文件里长什么样，模型看到的就是什么样

**排版约定**（`tests/unit/test_guidance.py` 有用例守着）：换行只落在标点上
（`。！？：；——`）或结构边界（标题 / 表格 / 列表 / 编号项 / 缩进子项），
不在词句中间硬折。
"""

import logging
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)

# 提示词目录。跟着包装走（`prism/prompts/`），与 rules/index.md 同一种放法
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

SUFFIX = ".md"


@cache
def load_prompt(name: str) -> str:
    """读一段提示词正文。

    **去掉首尾空行**（文件末尾的换行不该进提示词），中间一个字不动 ——
    文件里怎么写的，模型就怎么收到。

    缓存一份：`build_system()` 在分组装箱里会被反复调用（O(k²) 次渲染量成本），
    不能每次读盘。

    Args:
        name: 文件名（不含 `.md`），如 `system_role`

    Returns:
        提示词正文

    Raises:
        PromptMissingError: 文件不存在 —— **硬报错，不降级成空串**：
            缺一段提示词会让模型少读一整块要求，而且不报错的话没人会发现
    """
    path = PROMPTS_DIR / f"{name}{SUFFIX}"
    if not path.exists():
        raise PromptMissingError(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8").strip("\n")


class PromptMissingError(Exception):
    """提示词文件缺失。

    提示词随包发布，读不到说明打包漏了文件或名字写错 —— 两者都该当场炸掉：
    静默给一段空提示词，等于让模型少读一整块要求。
    """
