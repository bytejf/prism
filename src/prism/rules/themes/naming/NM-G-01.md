---
rule_id: NM-G-01
theme: naming
default_severity: low
keywords: [Url, Id, Http, URL, ID]
source: google
---

# NM-G-01：缩略词大小写要一致

**默认严重程度**：low

## 规则
Go 的惯例是缩略词**整体大写或整体小写**，不要首字母大写：

| 对 | 错 |
|---|---|
| `URL` / `url` | `Url` |
| `ID` / `id` | `Id` |
| `HTTPClient` / `httpClient` | `HttpClient` |
| `userID` | `userId` |

导出的名字里全大写（`URL`），未导出的开头小写（`url`），中间出现时跟着前后词的大小写（`userID`）。

## 正例
```go
func FetchURL(ctx context.Context, userID string) (*Response, error)
```

## 反例
```go
func FetchUrl(ctx context.Context, userId string) (*Response, error)
type HttpResponse struct{ RequestId string }
```

## 边界情况
- 已经是既有 API 的一部分、改了会破坏兼容的（导出的字段名、序列化的 JSON key）不动，但**新加**的名字要按这条来
