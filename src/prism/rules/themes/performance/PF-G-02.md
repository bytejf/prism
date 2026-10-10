---
rule_id: PF-G-02
theme: performance
default_severity: medium
keywords: ["strings.Builder", concat, "[]byte", string]
source: google
---

# PF-G-02：循环里别反复拼接与转换

**默认严重程度**：medium

## 规则
- 循环里用 `+=` 拼字符串是 O(n²)（每次都要新建一个字符串、复制一遍）；用 `strings.Builder`
- `[]byte(s)` / `string(b)` 每次都**复制一份**：在热路径或循环里反复做的，要看看能不能只转一次
- 需要按行写入时用 `bufio.Writer`，别每行一次 `Write` 系统调用

## 正例
```go
var b strings.Builder
for _, item := range items {
    b.WriteString(item.Name)
    b.WriteByte('\n')
}
return b.String()
```

## 反例
```go
var out string
for _, item := range items {
    out += item.Name + "\n"     // 每轮复制一次，items 大时是平方级
}
```

## 边界情况
- 循环次数确定且很小（2~3 次）时不值得改，可读性更重要
- 顺序敏感的字符串拼接要注意 `Builder` 与直接相加的**求值顺序**差异
