---
rule_id: TS-G-01
theme: testing
default_severity: low
keywords: [test, t.Run, table]
source: google
---

# TS-G-01：测试用表驱动，失败信息要能直接定位

**默认严重程度**：low

## 规则
- 同一段逻辑的多个用例写成**表驱动** + `t.Run(name, ...)`：加用例只加一行，失败时子测试名直接指出是哪个输入
- 失败信息要带上**输入、实际值、期望值**：`t.Errorf("Parse(%q) = %v, want %v", in, got, want)`
- 只说 `t.Error("failed")` 的断言，出问题的人得重新跑一遍才能知道发生了什么
- 抽了断言辅助函数的，在里面调 `t.Helper()`，否则报错行号指到辅助函数里

## 正例
```go
for _, tc := range []struct{ in, want string }{
    {"a=b", "a"},
    {"", ""},
} {
    t.Run(tc.in, func(t *testing.T) {
        if got := Split(tc.in); got != tc.want {
            t.Errorf("Split(%q) = %q, want %q", tc.in, got, tc.want)
        }
    })
}
```

## 反例
```go
func TestSplit(t *testing.T) {
    if Split("a=b") != "a" { t.Error("failed") }   // 哪个输入、期望什么都没说
    if Split("") != "" { t.Error("failed") }
}
```

## 边界情况
- 只有一个用例、或每个用例的前置构造差异很大时，不硬套表驱动
- 表格里的用例要覆盖**错误路径与边界输入**（空、零值、超长），只测 happy path 的表没有价值
