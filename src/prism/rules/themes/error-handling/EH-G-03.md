---
rule_id: EH-G-03
theme: error-handling
default_severity: high
keywords: [err, error, log, "_"]
source: google
---

# EH-G-03：错误不许被静默丢掉

**默认严重程度**：high

## 规则
以下三种都算「丢掉」，都要报：

- `if err != nil { }` —— 空分支，判断了但什么也没做
- 只 `log` 一句就继续往下走，而调用方拿不到「这一步失败了」这个事实
- 把错误抹成零值继续：`if err != nil { return "" }`（调用方分不清「没有」和「出错」）

真正要忽略时，必须**写清为什么**：`//nolint:errcheck // 关闭只读文件，失败无后果`

## 正例
```go
if err != nil {
    return fmt.Errorf("load config: %w", err)
}
```

## 反例
```go
value, err := s.Get(key)
if err != nil {
    // ① 静默吞掉：调用方拿到空串，以为「键不存在」
    return ""
}
if err != nil {
    log.Printf("get failed: %v", err)   // ② 记了日志但流程照continue，失败被当成成功
}
```

## 边界情况
- `_ = f.Close()` 这类**明确丢弃**是可接受的（但要能说出后果），`errcheck` 已能扫到的语法形态不必重复报
- 「记录并返回」不算丢掉；「记录并继续」要判它到底算不算处理（能自愈的算，剩下的不算）
