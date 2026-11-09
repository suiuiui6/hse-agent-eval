# 上线前检查清单

- 检查日期：2026-11-08
- 检查范围：`hse-agent-eval` 受控公开子集
- 结论：**未达上线条件**，存在中文注释、License 与依赖漏洞待办

## 1. 前端清理与字体

- 本仓库不包含前端构建产物，字体商用许可和无效代码检查不适用。
- 完整企业平台发布前仍需单独检查：无效代码、未使用字体、字体商用许可、包体积。

## 2. 后端测试路由

- 本仓库不包含后端服务。
- 撤下服务前已审计 `services/graphrag-provider/app/main.py` 和 `services/ocr-provider/app/main.py`：仅 `/api/v1/*` 业务端点，未发现 `/debug`、`/test`、`/admin` 或 Swagger 暴露。
- 服务代码待修复后单独发布。

## 3. 函数中文注释

- 当前仓库 AST 扫描：578 个函数/方法，523 个缺少 docstring。
- **未完成**。需要专项补齐中文注释，并由人工复核。
- 当前优先入口：`demo/`、`evals/hse-agent/run.py`、`evals/hse-agent/hse_agent_evals/`。

## 4. 开源组件 License

- 已扫描待发布 GraphRAG 与 OCR 服务的虚拟环境。
- **阻断项**：`PyMuPDF 1.26.3` 为 `AGPL-3.0 or commercial license`。因此 OCR 服务已从本仓库撤下；重新发布前必须取得商业授权、替换为宽松许可的 PDF 库，或按 AGPL 合规发布。
- 待人工确认：`fastapi`、`colorama`、`PyPDF2` 的元数据 License 为 `UNKNOWN`，需核对上游许可证。
- `certifi`、`tqdm` 含 `MPL-2.0`，需保留相应 notices。
- 其余主要为 MIT、BSD、Apache-2.0、PSF。
- 本仓库 harness 依赖仍需在 CI 中执行 License 清单扫描。

## 5. 漏洞扫描

使用 `pip-audit` 对待发布服务虚拟环境扫描：

- GraphRAG：20 条已知漏洞，涉及 5 个包：`certifi`、`pypdf2`、`pytest`、`setuptools`、`starlette`。
- OCR：至少 32 条已知漏洞，涉及 `pillow`、`pytest`、`python-multipart`、`starlette`。
- 两个服务已从本仓库撤下。重新发布前必须升级依赖、重新扫描并保留报告。
- 本仓库 harness 依赖仍需在 CI 中执行 `pip-audit`。

## 6. 隐私与内部文件

- 公司名/简称：0
- 个人姓名/拼音账号：0
- 员工工号：0
- 本地路径：0
- 法规正文/法规标题：0
- 乱码公司名：0
- 密钥/Token：0
- 企业内部法规库、Qdrant collection、法规 query、法规测试种子和企业内部文档均未包含。