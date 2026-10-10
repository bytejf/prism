---
rule_id: SC-G-03
theme: security
default_severity: high
keywords: [exec, filepath, sql, input]
source: google
---

# SC-G-03：外部输入进命令、路径、查询前必须校验

**默认严重程度**：high

## 规则
来自请求、配置、文件内容、另一个服务的字符串都算**外部输入**。它们要进下面这些地方时，
必须经过校验或转义，不能直接拼：

| 去处 | 要求 |
|---|---|
| 命令 | 用 `exec.CommandContext(ctx, name, args...)` 的**参数形式**，不要把输入拼进命令字符串 |
| 文件路径 | 先 `filepath.Clean`，再确认结果仍在允许的目录内（`strings.HasPrefix`/`filepath.Rel` 判 `..`） |
| SQL | 一律参数化（`db.Query("... WHERE id = ?", id)`），**不要** `fmt.Sprintf` 拼条件 |
| 模板 / 正则 | 输入进正则前 `regexp.QuoteMeta`，防止正则注入 |

## 正例
```go
cmd := exec.CommandContext(ctx, "git", "log", "--oneline", "-n", strconv.Itoa(n))
abs, err := filepath.Abs(filepath.Join(root, name))
if err != nil || !strings.HasPrefix(abs, root+string(os.PathSeparator)) {
    return "", ErrInvalidPath
}
```

## 反例
```go
exec.Command("sh", "-c", "git log "+branch)            // branch 里的 `;` 就是一条命令
rows, _ := db.Query("SELECT * FROM t WHERE k = '" + key + "'")
```

## 边界情况
- 输入已经在**上一层**严格校验过（白名单枚举、已解析过的 ID）时，本层不必重复校验，但要能在代码里指出那一层
- 转义库的用法要按它文档说的方式调用（比如 HTML 模板用 `html/template` 而不是 `text/template`）
