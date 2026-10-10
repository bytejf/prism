---
rule_id: EH-G-02
theme: error-handling
default_severity: high
keywords: [errors.Is, errors.As, error]
source: google
---

# EH-G-02：判定错误用 `errors.Is` / `errors.As`，不要比字符串或 `==`

**默认严重程度**：high

## 规则
- 判「是不是某个已知错误」用 `errors.Is(err, ErrNotFound)`
- 取错误里的具体类型用 `errors.As(err, &target)`
- 不要写 `err.Error() == "not found"`，也不要对**包装过**的错误用 `==` 直接比较

## 正例
```go
if errors.Is(err, sql.ErrNoRows) {
    return nil, ErrNotFound
}
```

## 反例
```go
// ① 字符串比较：错误消息一改就静默失效，而它通常没人当接口看
if err.Error() == "sql: no rows in result set" {
    return nil, ErrNotFound
}
// ② ==：只要中间有人用 %w 包了一层，这里就判不出来
if err == sql.ErrNoRows {
}
```

## 边界情况
- 判定**哨兵错误本身**（自己刚返回的那个、没有被包装过）时 `==` 也能用，但要能说清这一路没有包装
- 自定义错误类型如果实现了 `Is` 方法，仍然走 `errors.Is`
