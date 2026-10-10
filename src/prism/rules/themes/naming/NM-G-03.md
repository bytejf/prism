---
rule_id: NM-G-03
theme: naming
default_severity: low
keywords: [name, scope, 命名]
source: google
---

# NM-G-03：名字长度与作用域匹配

**默认严重程度**：low

## 规则
- **作用域越小，名字越短**：三行的循环变量叫 `i`、`v` 就够；跨文件使用的名字要能自解释
- 不要用包装类型：`userSlice` → `users`，`userMap` → `byID`
- 不要用无从推断的缩写：`usrCfg`、`procReq` 这类读的人要先解码
- 同一个概念在不同地方用同一个词（不要一会儿 `item` 一会儿 `record` 一会儿 `entry`）

## 正例
```go
for i, user := range users {
    if err := store.Save(ctx, user); err != nil {
        return fmt.Errorf("save user %d: %w", i, err)
    }
}
```

## 反例
```go
func processUserConfigurationEntries(u []UserConfigurationEntry) error {
    for idx := range u {
        // 名字长到没有信息量：类型已经说了它是 UserConfigurationEntry 的切片
    }
    return nil
}
```

## 边界情况
- 约定俗成的短名不受此限：`ctx`、`err`、`ok`、`mu`、`t`（测试）
- 接口方法名是契约的一部分，改动代价高，优先按既有约定
