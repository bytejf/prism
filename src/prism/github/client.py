"""GitHub REST API 的最小客户端。

**用标准库 `urllib`，不引 httpx**：我们要的只有五个 JSON 端点（列评论 / 删评论 /
建 review / 读写 issue 评论），标准库够用；而 httpx 现在只是 langchain 拉进来的
**传递依赖**，没写进 `pyproject.toml` —— 靠传递依赖等于把「哪天上游不拉了」变成
我们的故障（这条判据见 `rules` 那边对 PyYAML 的同类判断）。

**只管 HTTP，不管业务**：发哪些评论、删哪些、摘要写什么，都在 `post.py`。
这里只做三件事：拼请求、认错误、分页。
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
# 一页最多 100，GitHub 的上限就是它（写小了只是多跑几趟）
PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30.0


class GitHubError(Exception):
    """GitHub API 调用失败。

    **带状态码与响应体片段**：422 的响应体会指明「哪个字段、哪一行不合法」，
    只报「422」等于把唯一有用的信息丢掉（我们区分「行号锚不上」和「别的 422」
    正是靠它）。
    """

    def __init__(self, message: str, *, status: int, body: str = "") -> None:
        """初始化。

        Args:
            message: 人看的说明
            status: HTTP 状态码（网络层失败时为 0）
            body: 响应体片段（截断，避免日志被一个 HTML 错误页刷屏）
        """
        super().__init__(message)
        self.status = status
        self.body = body


class GitHubClient:
    """按仓库与令牌绑定的 REST 客户端。

    Attributes:
        repo: `owner/name`
        token: 访问令牌（**只在内存里**，不落盘、不进日志）
    """

    def __init__(
        self,
        token: str,
        repo: str,
        *,
        api_root: str = API_ROOT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """初始化。

        Args:
            token: 访问令牌
            repo: `owner/name`
            api_root: API 根地址（企业版 GitHub 可换）
            timeout: 单次请求超时（秒）
        """
        self.repo = repo
        self.token = token
        self._api_root = api_root.rstrip("/")
        self._timeout = timeout

    # ---------- 底层 ----------

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """发一个请求并解析 JSON。

        Args:
            method: HTTP 方法
            path: 相对路径（如 `/repos/o/r/pulls/1/comments`）
            payload: 请求体（JSON）
            params: query 参数

        Returns:
            解析后的 JSON（204 这类空响应返回 None）

        Raises:
            GitHubError: 网络层失败或状态码非 2xx
        """
        url = f"{self._api_root}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", API_VERSION)
        if data is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:800]
            raise GitHubError(
                f"GitHub API {method} {path} 失败：HTTP {exc.code}",
                status=exc.code,
                body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise GitHubError(f"GitHub API {method} {path} 连不上：{exc.reason}", status=0) from exc

        logger.debug("GitHub API %s %s → %d 字节", method, path, len(raw))
        return json.loads(raw) if raw.strip() else None

    def _paginate(self, path: str) -> list[dict[str, Any]]:
        """把一列资源读全（按 `per_page=100` 翻页）。

        发评论之前要扫一遍现有评论找我们自己的 —— 只读第一页会漏掉第 101 条
        之后的，于是「删旧评论」只删掉一半，而且**不报错**。

        Args:
            path: 相对路径

        Returns:
            全部条目
        """
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self._request("GET", path, params={"per_page": PAGE_SIZE, "page": page})
            if not isinstance(batch, list):
                raise GitHubError(f"读 {path} 期望列表，拿到 {type(batch).__name__}", status=0)
            items.extend(batch)
            if len(batch) < PAGE_SIZE:
                return items
            page += 1

    # ---------- 用到的端点 ----------

    def list_review_comments(self, pr: int) -> list[dict[str, Any]]:
        """PR 的全部行内评论（含别人的）。

        Args:
            pr: PR 编号

        Returns:
            评论列表
        """
        return self._paginate(f"/repos/{self.repo}/pulls/{pr}/comments")

    def delete_review_comment(self, comment_id: int) -> None:
        """删一条行内评论。

        **只能删自己发的**（GitHub 会拒别人的）—— 我们只删带自己标记的，所以
        天然落在权限内。

        Args:
            comment_id: 评论 id
        """
        self._request("DELETE", f"/repos/{self.repo}/pulls/comments/{comment_id}")

    def create_review(
        self, pr: int, *, commit_id: str, comments: list[dict[str, Any]], body: str = ""
    ) -> dict[str, Any]:
        """建一次 review，把若干条行内评论一次性带上。

        **为什么是批量而不是逐条**：逐条 `createReviewComment` 会在 PR 时间线上
        刷出 N 条独立记录（OCR 那边记着这是它们 issue #624 的 churn 问题）。
        批量还有个好性质：**一次请求的失败是原子的**，不会留下「发了一半」这种
        状态要人去猜。

        Args:
            pr: PR 编号
            commit_id: 评论锚在哪个提交上（PR 的 head SHA）
            comments: `path` / `line` / `side` / 可选 `start_line`·`start_side` / `body`
            body: review 本身的正文（留空；结论都在摘要那条里）

        Returns:
            GitHub 返回的 review 对象
        """
        payload = {"commit_id": commit_id, "comments": comments, "event": "COMMENT"}
        if body:
            payload["body"] = body
        result = self._request("POST", f"/repos/{self.repo}/pulls/{pr}/reviews", payload=payload)
        return result if isinstance(result, dict) else {}

    def list_issue_comments(self, pr: int) -> list[dict[str, Any]]:
        """PR 的普通评论（sticky 摘要就是其中一条）。

        Args:
            pr: PR 编号

        Returns:
            评论列表
        """
        return self._paginate(f"/repos/{self.repo}/issues/{pr}/comments")

    def create_issue_comment(self, pr: int, body: str) -> dict[str, Any]:
        """发一条普通评论。

        Args:
            pr: PR 编号
            body: 正文

        Returns:
            GitHub 返回的评论对象
        """
        result = self._request(
            "POST", f"/repos/{self.repo}/issues/{pr}/comments", payload={"body": body}
        )
        return result if isinstance(result, dict) else {}

    def update_issue_comment(self, comment_id: int, body: str) -> dict[str, Any]:
        """就地更新一条普通评论（sticky 摘要走这里）。

        Args:
            comment_id: 评论 id
            body: 新正文

        Returns:
            GitHub 返回的评论对象
        """
        result = self._request(
            "PATCH", f"/repos/{self.repo}/issues/comments/{comment_id}", payload={"body": body}
        )
        return result if isinstance(result, dict) else {}
