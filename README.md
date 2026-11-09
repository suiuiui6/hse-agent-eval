# HSE Agent Evaluation Harness

面向会写数据库的 HSE Agent 的可复现评测与契约演示。

## 包含

- 100 条中文 gold case（70 development / 30 regression）
- 多轮、取消、重复确认、安全、越权、注入和数据库副作用断言
- 确定性 evaluator：结果、工具选择、参数、顺序、确认、副作用、预算
- 可恢复 runner、fixture 编排、报告生成
- 自包含 contract demo，不依赖 LLM、数据库或 Java 后端

## 不包含

- 企业平台源码、内部规划、UAT 和运维文档
- 企业内部法规库、Qdrant collection、法规 gold set、检索 query 和生产数据
- 公司名、人名、工号、内网地址和本地路径
- 任何测试环境凭据
- GraphRAG / OCR 服务实现：待 License 与漏洞修复后单独发布

## 快速开始

### Contract demo

```bash
docker compose -f demo/docker-compose.yml up --build --abort-on-container-exit --exit-code-from smoke
```

### Harness 单元测试

```bash
cd evals/hse-agent
python -m pytest tests -q
```

### 完整评测

需要兼容的 Agent 后端和自备 fixture；设计说明见 [docs/case-study.md](docs/case-study.md)。

## 发布状态

这是参考实现，不是生产就绪版本。上线前必须完成 [PRELAUNCH-CHECKLIST.md](PRELAUNCH-CHECKLIST.md) 中的依赖升级、License 复核和漏洞修复。