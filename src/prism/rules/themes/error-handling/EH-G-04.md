---
rule_id: EH-G-04
theme: error-handling
default_severity: medium
keywords: [panic, recover, Must]
source: google
---

# EH-G-04：`panic` 只用于不可恢复的错误

**默认严重程度**：medium

## 规则
- 可预期的失败（参数不合法、依赖不可用、解析失败）应该返回 `error`，不 `panic`
- 不要用 `panic` 做控制流，也不要用 `recover` 把 `panic` 吞成正常返回
- 只有「继续跑下去毫无意义」的初始化错误（如配置缺失导致进程不该启动）才适合 `panic`

## 正例
```go
func Parse(s string) (Config, error) {
    cfg, err := decode(s)
    if err != nil {
        return Config{}, fmt.Errorf("parse config: %w", err)
    }
    return cfg, nil
}
```

## 反例
```go
func Parse(s string) Config {
    cfg, err := decode(s)
    if err != nil {
        // 调用方没法处理，进程只能挂 —— 而且 panic 带的是原始 err，没有上下文
        panic(err)
    }
    return cfg
}
```

## 边界情况
- `Must*` 形式的辅助函数（`template.Must` 那种，只在包初始化里用）是惯例做法，不算问题
- `recover` 用在一处的**边界**（HTTP handler 顶层、goroutine 顶层）是把崩溃转成错误的正路；用在中途吞掉再继续跑就不是
