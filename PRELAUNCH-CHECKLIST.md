# 上线前检查清单

- 检查日期：2026-11-09
- 检查范围：`hse-agent-eval` 受控公开子集
- 结论：**harness 的依赖与注释已收口**；服务代码仍因 License 与漏洞问题暂不发布

## 1. 前端清理与字体

- 本仓库不包含前端构建产物，字体商用许可和无效代码检查不适用。
- 完整企业平台发布前仍需单独检查：无效代码、未使用字体、字体商用许可、包体积。

## 2. 后端测试路由

- 本仓库不包含后端服务。
- 撤下服务前已审计 `services/graphrag-provider/app/main.py` 和 `services/ocr-provider/app/main.py`：仅 `/api/v1/*` 业务端点，未发现 `/debug`、`/test`、`/admin` 或 Swagger 暴露。
- 服务代码待修复后单独发布。

## 3. 函数中文注释

- AST 扫描：582 个函数/方法，0 个缺少 docstring。
- 已补齐中文基线注释；关键入口和已有 docstring 为人工可读描述，仍需人工复核准确性。

## 4. 开源组件 License

- 已扫描待发布 GraphRAG 与 OCR 服务的虚拟环境。
- **阻断项**：`PyMuPDF 1.26.3` 为 `AGPL-3.0 or commercial license`。因此 OCR 服务已从本仓库撤下；重新发布前必须取得商业授权、替换为宽松许可的 PDF 库，或按 AGPL 合规发布。
- `pypdf 7.0.0` 为 BSD-3-Clause；`fastapi`、`colorama` 元数据 License 为 `UNKNOWN`，已知为宽松许可，仍建议保留上游 NOTICE。
- `certifi`、`tqdm` 含 `MPL-2.0`，需保留相应 notices。
- 其余主要为 MIT、BSD、Apache-2.0、PSF。
- 本仓库 harness 锁定依赖均为宽松许可；`certifi` 为 MPL-2.0，需保留 notices。

## 5. 漏洞扫描

使用 `pip-audit` 对待发布服务虚拟环境扫描：

- GraphRAG：依赖已升级并锁定；`pip-audit` 复扫无已知漏洞；83 个测试通过。
- OCR：依赖已升级并锁定；`pip-audit` 复扫无已知漏洞；11 个测试通过。
- 两个服务仍从本仓库撤下；OCR 的 PyMuPDF AGPL 许可是唯一剩余发布阻断项。
- 本仓库 harness 依赖已锁定：`requirements.lock.txt`（Python 3.12，21 个包），`pip-audit` 复扫结果为 **No known vulnerabilities**。

## 6. 隐私与内部文件

- 公司名/简称：0
- 个人姓名/拼音账号：0
- 员工工号：0
- 本地路径：0
- 法规正文/法规标题：0
- 乱码公司名：0
- 密钥/Token：0
- 企业内部法规库、Qdrant collection、法规 query、法规测试种子和企业内部文档均未包含。