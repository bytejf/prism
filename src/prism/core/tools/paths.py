"""工具参数的路径校验：**一条不变量，一处实现**。

工具入参是模型给的 —— 这是我们唯一的信任边界。三个工具里有两个会把参数交给
**外面**（`git` 子进程 / `pygit2`），而它们对越界与非法字符的反应各不相同：

| 输入 | git / pygit2 的反应 |
|---|---|
| `../*`、`/etc/*` 这类 pathspec | git 自己拒：`fatal: outside repository` |
| 含 NUL 的字符串 | `subprocess` 抛 `ValueError`（**没人接，整组审查崩**） |
| 被审查版本里没有的路径 | pygit2 给 `KeyError`，我们当成「文件不存在」 |

三种失败长得完全不一样：有的是我们的中文提示，有的把 git 的英文 stderr 连同
**本机绝对路径**一起原样回给模型（那段文字是要发给外部 API 的），有的直接抛。
**校验放在我们这一侧**，失败就长一个样：

- 越界（绝对路径 / `..`）→ 明确的中文错误，且**不把本机路径写进去**
- NUL → 同样的中文错误（它在下游会让 `subprocess` 抛未捕获异常）

`file_read_diff` 不走这里：它只做字典查询，越界路径的结局就是「查不到」，
不碰文件系统也就没有要防的东西。
"""

from pathlib import PurePosixPath

# NUL 在 POSIX 路径里不合法，且会在 subprocess 与 pygit2 里以不同方式炸掉
_ILLEGAL = "\x00"


def check_no_nul(value: str, *, field: str) -> None:
    """拒绝含 NUL 的字符串。

    NUL 是唯一**能让子进程直接抛异常**的字符（`subprocess.run` 的 argv 不接受它），
    而那个异常会一路穿过工具与框架（`_default_handle_tool_errors` 只转
    `ToolInvocationError`，其余原样 re-raise），把整组审查带走。所以它必须在
    参数刚进来时就拦住。

    Args:
        value: 待校验的字符串（路径、pathspec、正则都可能）
        field: 参数名，用于错误信息

    Raises:
        ValueError: 含 NUL
    """
    if _ILLEGAL in value:
        msg = f"{field} 含非法字符（NUL）：{value!r}"
        raise ValueError(msg)


def check_relative_path(value: str, *, field: str = "path") -> None:
    """要求参数是**仓库内的相对路径**。

    纯字符串判断，不碰文件系统 —— 因此对「读 git 对象」那条路同样有效
    （那条路不经过 `Path.resolve()`，先读后校验会让 `..` 绕过去）。

    这两条挡不住的只剩**符号链接**：仓库里一个指向外面的软链接，字面上是合法
    相对路径。那个由读取侧再挡一道（`file_read._resolve_inside_project` 的
    `resolve()` + `is_relative_to`；走 git 时读的是 blob 内容，链接本身就是
    一段文本，天然不跟随）。

    Args:
        value: 待校验的参数
        field: 参数名，用于错误信息

    Raises:
        ValueError: 空、含 NUL、绝对路径，或含 ``..`` 段
    """
    check_no_nul(value, field=field)
    if not value.strip():
        msg = f"{field} 不能为空"
        raise ValueError(msg)

    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        msg = f"{field} 必须是仓库内相对路径：{value}"
        raise ValueError(msg)
    if ".." in candidate.parts:
        msg = f"{field} 不得包含 ..：{value}"
        raise ValueError(msg)
