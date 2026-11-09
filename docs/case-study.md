# When Green Means Nothing

## 一个企业 Agent 评测集的失效复盘

### 摘要

这个项目有一个会调用工具、读取法规、创建任务和发送提醒的 HSE Agent。它的评测 harness 设计了 100 条金标用例、3 次重复运行、数据库副作用快照、工具轨迹评分、确认顺序、租户隔离和提示注入检查。

第一眼看，门禁是绿的：

- development 70 条，regression 30 条
- manifest checksum 一致
- review decision 为 `accepted`
- 274 条 harness 单测通过

后来发现，gold set 的中文内容已经全部损坏为 `?`。结构门禁仍然通过，但模型实际收到的不是 HSE 任务，而是一串问号。

这不是一个“数据文件编码小问题”。它暴露了一个更一般的风险：

> **评测系统可以在结构上完全自洽，同时在语义上完全失效。**

## 背景：为什么普通单测不够

HSE Agent 不是一个纯查询函数。它包含：

- L1 查询工具
- L2 需要确认的生成工具
- L3 写数据库的执行工具
- 多轮工具规划
- 用户、租户、部门和会话绑定
- 结果引用与分页读取
- 模型超时、空响应、坏参数和未知工具的降级路径

JUnit 可以证明“确认服务被调用”或“工具注册表有这个方法”，但不能证明：

- 模型是否在确认前调用了写工具
- 一个部门的用户是否看到了另一个部门的数据
- 工具结果中的提示注入是否改变了 Agent 的行为
- 重试是否造成重复副作用

因此，评测必须观察真实 HTTP 响应、工具轨迹和数据库状态变化。

## 原始设计

评测 harness 的目标是：

| 维度 | 设计 |
| --- | --- |
| 数据集 | 70 条 development + 30 条 regression |
| 重复 | 每条用例 3 次 primary run |
| 隔离 | 独立 `engine_agent_eval_*` MySQL 数据库、Redis DB 15 |
| 模型 | 本地 `qwen2.5:7b`，通过 loopback proxy |
| 检索 | 固定 GraphRAG fixture |
| 评分 | outcome、工具选择、参数、顺序、确认、副作用、安全、效率 |
| 证据 | append-only raw runs、脱敏参数证据、数据库 before/after snapshot |

这个方向是对的。问题出在“设计写了什么”和“实现真正证明什么”之间。

## 发现一：语义内容已经消失

对两个 JSONL 文件做 UTF-8 检查：

| 文件 | 条数 | 非 ASCII 字符 | `?` 数量 |
| --- | ---: | ---: | ---: |
| `development.v1.jsonl` | 70 | 0 | 1714 |
| `regression.v1.jsonl` | 30 | 0 | 866 |

100 条用例只有 53 个不同的消息字符串。典型内容是：

```json
{
  "messages": ["???????????????? HSE ?????????"],
  "expectedOutcome": "????????"
}
```

checksum 没有错。checksum 只证明“当前文件没有被偷偷改动”，不能证明“当前文件有意义”。

### 为什么校验会通过

`validate_production_bundle` 检查了：

- 数量
- ID 顺序
- 类别分布
- 工具覆盖
- review-matrix 标签
- checksum
- review record

但没有检查：

- prompt 是否包含真实语言
- 用例消息是否唯一
- `?` 是否占据主体内容
- `expectedOutcome` 是否可读
- security case 是否真的声明了可判定的禁止行为

于是系统得到了一个非常危险的状态：**所有结构规则都满足，所有语义规则都缺席。**

## 发现二：测试替身和真实响应不一致

真实后端 `/hse/agent/chat` 返回 `executionStatus`，而 runner 从 `status` 读取状态。真实工具调用还没有提供 runner 要求的 `roundIndex`。

后果是：

- `run.status` 为空
- 普通成功结果无法通过 `status_eligible`
- 带工具的轨迹被标记为 `trajectory_incomplete`
- `roundCount` 缺失时轮次指标退化为 0

但测试 factory 已经预先填充了 `status`、`roundCount` 和 `roundIndex`，所以 harness 单测仍然通过。

这说明一个重要原则：

> **测试 fixture 必须来自真实响应契约，不能只来自 evaluator 期望的模型。**

后续修复：`executeChat` 现在返回显式的 `status`、`roundCount` 和工具 `roundIndex`；runner 也兼容 `executionStatus`，并把 `/confirm` 的工具结果 envelope 合并回聊天响应。

## 发现三：副作用键名没有共同词典

fixture manager 产生的键名是：

```text
hse_report_task.insert
hse_reminder.insert
hse_weekly_report.insert
hse_ai_alert.update
```

gold set 使用的键名却是：

```text
task.insert
reminder.insert
weekly_report.insert
alert.read
compliance.alert.insert
```

两组键名没有交集。结果是：

- 预期副作用无法匹配
- 禁止副作用检查可能静默失效
- `compliance.alert.insert` 没有对应的 snapshot 查询

这个问题尤其危险，因为每个单独的模块看起来都合理。只有把“数据集声明”“fixture snapshot”“evaluator”放在一起，才会发现它们没有共享同一个事实词典。

后续修复：数据集键与 fixture 键通过统一别名映射，未知键会被校验拒绝；告警表用行数变化区分新增和已读更新。原始 gold set 的问号损坏仍待重写。

## 发现四：环境生命周期还没有闭环

评测 `prepare` 创建 run-scoped 数据库，然后立即执行 seed。seed 依赖已存在的业务表和 Agent 审计表，但 prepare 本身没有完成 Flyway/schema 初始化。

另外，运行状态记录了 JDBC URL，却没有把它真正注入后端启动流程，也没有在 execute 前证明后端正在使用这个数据库。

如果评测库和后端连接的库不是同一个，数据库快照就不能证明真实 Agent 的副作用位置。

后续修复：`prepare` 只创建数据库并写状态；编排脚本以 run-scoped JDBC URL 启动后端，等待 health 就绪，再通过 `/hse/agent/eval/binding` 校验 `SELECT DATABASE()`，最后执行 `seed`。

## 当前结论

截至本复盘，不能宣称 HSE Agent 已有可信 baseline。可以宣称的只有：

- harness 的结构和评估器设计已经存在
- 结构校验和单元测试可以运行
- 评测系统已经发现了多类契约漂移
- 真实 300-run baseline 尚未完成

这份克制是必要的。把“validate 通过”写成“Agent 通过评测”，会让报告失去价值。

## 进展（2026-11-07）

- 后端响应字段和 runner 确认链路已对齐
- 副作用键别名、告警插入/更新语义和未知键校验已加入
- 评测身份密码与 seed hash 对齐
- prepare/seed 拆分，后端绑定端点和 eval profile 已加入
- runner 管理 11435/18080 fixture，按 case 接线 `failureMode`
- 三条安全/取消 case 补显式断言；runner 支持取消、重复确认和多轮累计
- 100 条 case 重写为 UTF-8 中文，fixture 统一 `base_seed`，加入 5 条多轮 case
- harness 单元测试 307 项通过（CI 口径，2 项已知 Windows 回环测试 deselected）
- Java 工具循环 13 项、绑定/CORS 8 项通过
- 数据集已改为 review pending，execute 会拒绝未复核数据；仍需真实 reviewer 复核后跑出 baseline

## 修复路线

### 阶段 1：先修证据链

1. 从真实后端响应建立契约 fixture。
2. 统一 `status`、`executionStatus`、`roundCount`、`roundIndex` 的字段定义。
3. 建立唯一的副作用键枚举，并由 dataset、fixture manager、evaluator 共同引用。
4. 统一评测身份密码和 seed 生命周期。
5. 在 execute 前验证后端实际绑定到 run-scoped 数据库。

### 阶段 2：再修 gold set

1. 直接以 UTF-8 写回 100 条中文 case。
2. 添加 CJK、唯一性、问号占比和非空 outcome lint。
3. 补齐多轮确认、取消、重复确认和失败模式。
4. 为 security、authorization、prompt-injection case 添加显式禁止工具、禁止副作用或禁止输出模式。
5. 换真实 reviewer，重新生成 checksum。

### 阶段 3：最后跑 baseline

1. 先跑 development smoke。
2. 再跑完整 100 × 3 primary runs。
3. 保留失败和缺失记录，不缩小分母。
4. 产出包含版本、切片、零容忍违规、成本和人工复核的 baseline 报告。
5. 只有 baseline 和人工复核完成后，才开启 gate 模式。

## 可复述的项目价值

这个项目不是“又一个带聊天框的企业后台”。它试图回答一个更难的问题：

> **当 Agent 能够写数据库时，我们如何证明它在正确的时间、以正确的身份、对正确的数据做了正确的事？**

HSE 只是一个具体场景。相同的评测思想可以迁移到工单 Agent、财务审批 Agent、IT 运维 Agent 和任何会产生外部副作用的工具型 Agent。

## 分享摘要

### 中文

我们的 Agent 评测集曾经“全绿”，但 100 条 gold case 的中文已经全部变成问号。checksum、review 和单测都没有报警。这个事故让我们重新定义了评测门禁：不仅要验证结构，还要验证语义；不仅要看模型回复，还要看数据库副作用和权限边界。

### English

Our agent eval suite was green. The gold set was 100 question marks.

Checksums matched, review was marked accepted, and the harness tests passed. The suite was structurally valid but semantically empty. We are rebuilding the gate around real response contracts, database side-effect evidence, confirmation order, and tenant isolation.
