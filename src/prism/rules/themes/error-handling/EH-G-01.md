---
rule_id: EH-G-01
theme: error-handling
default_severity: medium
keywords: [error, "%w", fmt.Errorf]
source: google
---

# EH-G-01：错误要带上上下文，并且用 `%w` 包装

**默认严重程度**：medium

## 规则
- 往上返回错误时补一层**上下文**（哪个文件、哪个键、哪次操作），但不要复述调用方已知的信息
- 需要调用方用 `errors.Is` / `errors.As` 判定时，用 `%w` 包装；用 `%v` / `%s` 会把链路断掉
- 错误消息小写开头、不以标点结尾（Go 的错误会被层层拼接，大写与句号在中途变成噪音）

## 正例
```go
f, err := os.Open(path)
if err != nil {
    return nil, fmt.Errorf("open %s: %w", path, err)
}
```

## 反例
```go
// ① 没上下文：调用方只知道「有错」② 用 %v 断掉了 errors.Is
data, err := os.ReadFile(path)
if err != nil {
    return fmt.Errorf("Failed to read file: %v.", err)
}
```

## 边界情况
- 错误链条上**每一层都加**同样的上下文（`open a: open a: open a: ...`）比不加更糟，只在信息不重复时加
- 第三方库返回的错误原样往上传是对的：它已经带了上下文
