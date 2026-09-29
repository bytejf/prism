# go-sample 埋入的 bug（供人工核对，勿放进 base/ 或 buggy/）

本文件只给人和 CI 看。**不要**把下面的说明写进 `base/` 或 `buggy/` 的 Go 代码里——
LLM 会读到源码，若注释里直接点明 bug，smoke test 就失去意义了。

`base/` 与 `buggy/` 都是普通 Go 文件，**没有 `.git/`**；
测试时由 `tests/e2e/test_smoke.py` 用 `tmp_path` 程序化构造临时仓库
（main = base，feature = buggy），避免嵌套仓库无法被 git 跟踪的问题。

## bug 清单

| # | 文件 | 位置 | 问题 | 期望 category |
|---|------|------|------|--------------|
| 1 | `service.go` | `Process` | `user, _ := s.db.FindUser(id)` 忽略 error，查询失败也会往下走 | error_handling |
| 2 | `service.go` | `Run` | `for { s.doWork() }` 无退出条件，`stop` 通道不再被消费 | concurrency |
| 3 | `handler.go` | `GetUser` | 去掉了 `id == ""` 校验，空 id 会直接打到数据库 | bug |
| 4 | `db.go` | `FindUser` | `"SELECT ... WHERE id = '" + id + "'"` 字符串拼接 | security |
| 5 | `handler.go` | `GetUser` | `json.NewEncoder(w).Encode(user)` 返回值未检查 | error_handling |

## 跨文件一致性（隐含的第 6 个观察点）

`main.go` 两个版本完全相同，仍然是 `svc.Run()` 紧接 `defer svc.Stop()`。
bug 2 让 `Run` 不再消费 `stop`，于是 `main.go` 里的 `Stop()` 成了空转，
`defer` 永不执行 —— 这是「跨文件一致性」维度应当发现的点。

## V1 不做的断言

V1 没有定位、去重、反思，评论质量本身不稳定。
`tests/e2e/test_smoke.py` 只断言「命令跑通 + 有输出」，不断言命中上述 bug。
阶段 3 再加强断言（见《5-测试指导文档》§4）。
