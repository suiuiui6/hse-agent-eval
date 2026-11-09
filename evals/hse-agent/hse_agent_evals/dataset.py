from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .contracts import KNOWN_SIDE_EFFECT_KEYS
from .models import EvalCase


EXPECTED_SPLIT_COUNTS = {"development": 70, "regression": 30}
EXPECTED_CATEGORY_COUNTS = {
    "no_tool": 10,
    "l1": 25,
    "l2": 15,
    "l3": 15,
    "multi_tool": 15,
    "boundary": 10,
    "security": 5,
    "failure": 5,
}
PRODUCTION_TOOL_NAMES = {
    "get_my_tasks",
    "get_kpi",
    "get_dept_ranking",
    "search_regulations",
    "read_tool_result_page",
    "get_alerts",
    "generate_weekly_report",
    "check_compliance",
    "generate_urge_suggestions",
    "send_reminder",
    "create_task",
    "mark_alert_read",
}
REVIEW_MATRIX_TAGS = {f"review-matrix-{index:02d}" for index in range(1, 46)}
REVIEW_DECISIONS = {"pending", "accepted", "rejected"}
SUPPORTED_FIXTURES = {"base_seed"}
SECURITY_TAGS = {
    "authorization",
    "prompt-injection",
    "cross-department",
    "security",
    "tenant-isolation",
    "resultref-isolation",
}
EXPECTED_PRODUCTION_SPLIT_CATEGORY_COUNTS = {
    "development": {
        "no_tool": 10,
        "l1": 25,
        "l2": 15,
        "l3": 15,
        "multi_tool": 5,
    },
    "regression": {
        "multi_tool": 10,
        "boundary": 10,
        "security": 5,
        "failure": 5,
    },
}


def load_cases(path: str | Path) -> list[EvalCase]:
    """处理 load cases 相关逻辑。"""
    source = Path(path)
    cases: list[EvalCase] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} 第 {line_number} 行不是有效 JSON") from exc
        try:
            cases.append(_parse_case(payload))
        except ValueError as exc:
            raise ValueError(f"{source} 第 {line_number} 行用例无效") from exc
    return cases


def validate_suite(
    development: Iterable[EvalCase | dict[str, Any]],
    regression: Iterable[EvalCase | dict[str, Any]],
    manifest: dict[str, Any] | None,
    *,
    allow_approved_manifest: bool = False,
) -> list[str]:
    """处理 validate suite 相关逻辑。"""
    errors: list[str] = []
    development_cases = _to_cases(development, "development", errors)
    regression_cases = _to_cases(regression, "regression", errors)
    all_cases = development_cases + regression_cases

    _validate_manifest(manifest, errors, allow_approved_manifest=allow_approved_manifest)
    _validate_split_cases("development", development_cases, errors)
    _validate_split_cases("regression", regression_cases, errors)

    ids = [case.id for case in all_cases]
    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"存在重复 ID: {', '.join(duplicate_ids)}")

    messages = [message for case in all_cases for message in case.messages]
    duplicate_messages = sorted(
        message for message, count in Counter(messages).items() if count > 1
    )
    if duplicate_messages:
        errors.append(
            f"存在重复用例消息: {duplicate_messages[0][:40]}"
        )

    actual_splits = {
        "development": len(development_cases),
        "regression": len(regression_cases),
    }
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        if actual_splits[split] != expected_count:
            errors.append(f"{split} 数量应为 {expected_count}，实际为 {actual_splits[split]}")

    category_counts = Counter(case.category for case in all_cases)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        actual_count = category_counts.get(category, 0)
        if actual_count != expected_count:
            errors.append(f"{category} 类别数量应为 {expected_count}，实际为 {actual_count}")

    return errors


def validate_production_bundle(
    development_path: str | Path,
    regression_path: str | Path,
    manifest_path: str | Path,
    review_path: str | Path | None = None,
) -> list[str]:
    """处理 validate production bundle 相关逻辑。"""
    development_source = Path(development_path)
    regression_source = Path(regression_path)
    manifest_source = Path(manifest_path)
    development_cases = load_cases(development_source)
    regression_cases = load_cases(regression_source)
    manifest = _load_json(manifest_source)
    review = _load_json(Path(review_path)) if review_path is not None else None

    errors = validate_suite(
        development_cases,
        regression_cases,
        manifest,
        allow_approved_manifest=review_path is not None,
    )
    all_cases = development_cases + regression_cases

    _validate_production_split_layout("development", development_cases, errors)
    _validate_production_split_layout("regression", regression_cases, errors)
    _validate_expected_case_order("development", development_cases, errors)
    _validate_expected_case_order("regression", regression_cases, errors)
    _validate_tool_coverage(all_cases, errors)
    _validate_review_matrix_tags(all_cases, errors)
    _validate_manifest_checksums_against_files(
        manifest,
        development_source,
        regression_source,
        errors,
    )
    if review_path is not None:
        _validate_review_record(review, manifest, all_cases, errors)
    return errors


def update_manifest_checksums(
    development_path: str | Path,
    regression_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """处理 update manifest checksums 相关逻辑。"""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    checksums = manifest.setdefault("checksums", {})
    checksums["development"] = _sha256(Path(development_path))
    checksums["regression"] = _sha256(Path(regression_path))
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _to_cases(
    values: Iterable[EvalCase | dict[str, Any]],
    split: str,
    errors: list[str],
) -> list[EvalCase]:
    """to cases 的内部辅助逻辑。"""
    cases: list[EvalCase] = []
    for index, value in enumerate(values, start=1):
        try:
            cases.append(value if isinstance(value, EvalCase) else _parse_case(value))
        except ValueError as exc:
            errors.append(f"{split} 第 {index} 条用例无效: {exc}")
    return cases


def _validate_manifest(
    manifest: dict[str, Any] | None,
    errors: list[str],
    *,
    allow_approved_manifest: bool = False,
) -> None:
    """validate manifest 的内部辅助逻辑。"""
    if not isinstance(manifest, dict):
        errors.append("manifest 必须为对象")
        return
    if manifest.get("version") != "v1":
        errors.append("manifest version 必须为 v1")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("manifest 缺少 dataset")
        return
    if dataset.get("splits") != EXPECTED_SPLIT_COUNTS:
        errors.append("manifest dataset.splits 不符合要求")
    if dataset.get("categoryCounts") != EXPECTED_CATEGORY_COUNTS:
        errors.append("manifest dataset.categoryCounts 不符合要求")
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(EXPECTED_SPLIT_COUNTS):
        errors.append("manifest checksums 必须包含 development 和 regression")
    approved = manifest.get("approved")
    if allow_approved_manifest:
        if not isinstance(approved, bool):
            errors.append("manifest approved 必须为布尔值")
    elif approved is not False:
        errors.append("manifest approved 必须为 false")
    if isinstance(checksums, dict):
        for split in EXPECTED_SPLIT_COUNTS:
            value = checksums.get(split)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
                errors.append(f"manifest checksums {split} 必须为 64 位十六进制")


def _validate_split_cases(split: str, cases: list[EvalCase], errors: list[str]) -> None:
    """validate split cases 的内部辅助逻辑。"""
    for case in cases:
        if case.dataset_split != split:
            errors.append(f"{split} 用例 {case.id} 的 datasetSplit 应为 {split}")
        if case.fixture not in SUPPORTED_FIXTURES:
            errors.append(f"{split} 用例 {case.id} 使用了不支持的 fixture: {case.fixture}")
        for message in case.messages:
            if not _has_cjk(message):
                errors.append(f"{split} 用例 {case.id} 的消息缺少中文内容")
            if "??" in message or message.count("?") > max(3, len(message) // 5):
                errors.append(f"{split} 用例 {case.id} 的消息疑似编码损坏")
        if not case.expected_outcome.strip() or not _has_cjk(case.expected_outcome):
            errors.append(f"{split} 用例 {case.id} 的 expectedOutcome 缺少可读中文")
        if case.confirmation_action == "cancel":
            if not case.confirmation_required:
                errors.append(f"{split} 用例 {case.id} 的取消用例必须要求确认")
            if case.expected_side_effects:
                errors.append(f"{split} 用例 {case.id} 的取消用例不能预期副作用")
        if case.confirmation_attempts > 1 and case.confirmation_action != "confirm":
            errors.append(f"{split} 用例 {case.id} 的重复确认必须使用 confirm 动作")
        security_tags = {tag.casefold() for tag in case.tags} & SECURITY_TAGS
        if (case.category == "security" or security_tags) and not (
            case.forbidden_tools
            or case.forbidden_side_effects
            or any(pattern.strip() for pattern in case.forbidden_output_patterns)
        ):
            errors.append(f"{split} 用例 {case.id} 缺少显式安全断言")
        for key in sorted(
            set(case.expected_side_effects) | set(case.forbidden_side_effects)
        ):
            if key not in KNOWN_SIDE_EFFECT_KEYS:
                errors.append(
                    f"{split} 用例 {case.id} 引用了未知副作用键: {key}"
                )


def _has_cjk(value: str) -> bool:
    """has cjk 的内部辅助逻辑。"""
    return any("\u4e00" <= character <= "\u9fff" for character in value)


def _parse_case(value: dict[str, Any]) -> EvalCase:
    """parse case 的内部辅助逻辑。"""
    return EvalCase.model_validate(value)


def _sha256(path: Path) -> str:
    """sha256 的内部辅助逻辑。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    """load json 的内部辅助逻辑。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validate_production_split_layout(
    split: str,
    cases: list[EvalCase],
    errors: list[str],
) -> None:
    """validate production split layout 的内部辅助逻辑。"""
    expected_counts = EXPECTED_PRODUCTION_SPLIT_CATEGORY_COUNTS[split]
    actual_counts = Counter(case.category for case in cases)
    if actual_counts != Counter(expected_counts):
        errors.append(f"{split} 类别分布应为 {expected_counts}，实际为 {dict(actual_counts)}")


def _validate_expected_case_order(
    split: str,
    cases: list[EvalCase],
    errors: list[str],
) -> None:
    """validate expected case order 的内部辅助逻辑。"""
    actual_ids = [case.id for case in cases]
    expected_ids = _expected_split_ids(split)
    if actual_ids != expected_ids:
        errors.append(f"{split} ID 顺序不符合 Task 12 分配规则")


def _validate_tool_coverage(cases: list[EvalCase], errors: list[str]) -> None:
    """validate tool coverage 的内部辅助逻辑。"""
    mentioned_counts: Counter[str] = Counter()
    for case in cases:
        for tool_name in _mentioned_tools(case):
            mentioned_counts[tool_name] += 1
    mentioned_tools = set(mentioned_counts)
    missing_tools = sorted(PRODUCTION_TOOL_NAMES - mentioned_tools)
    unexpected_tools = sorted(mentioned_tools - PRODUCTION_TOOL_NAMES)
    if missing_tools:
        errors.append(f"生产工具未覆盖: {', '.join(missing_tools)}")
    if unexpected_tools:
        errors.append(f"存在未知工具引用: {', '.join(unexpected_tools)}")
    underrepresented = sorted(
        tool_name
        for tool_name, count in mentioned_counts.items()
        if tool_name in PRODUCTION_TOOL_NAMES and count < 3
    )
    if underrepresented:
        errors.append(f"以下工具少于 3 个用例覆盖: {', '.join(underrepresented)}")


def _validate_review_matrix_tags(cases: list[EvalCase], errors: list[str]) -> None:
    """validate review matrix tags 的内部辅助逻辑。"""
    actual_tags = {
        tag
        for case in cases
        for tag in case.tags
        if tag.startswith("review-matrix-")
    }
    missing = sorted(REVIEW_MATRIX_TAGS - actual_tags)
    unexpected = sorted(actual_tags - REVIEW_MATRIX_TAGS)
    if missing:
        errors.append(f"缺少 review-matrix 标签: {', '.join(missing)}")
    if unexpected:
        errors.append(f"存在未定义 review-matrix 标签: {', '.join(unexpected)}")


def _validate_manifest_checksums_against_files(
    manifest: dict[str, Any],
    development_path: Path,
    regression_path: Path,
    errors: list[str],
) -> None:
    """validate manifest checksums against files 的内部辅助逻辑。"""
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        return
    expected = {
        "development": _sha256(development_path),
        "regression": _sha256(regression_path),
    }
    for split, checksum in expected.items():
        if checksums.get(split) != checksum:
            errors.append(f"manifest checksums {split} 与文件内容不一致")


def _validate_review_record(
    review: dict[str, Any] | None,
    manifest: dict[str, Any],
    cases: list[EvalCase],
    errors: list[str],
) -> None:
    """validate review record 的内部辅助逻辑。"""
    if not isinstance(review, dict):
        errors.append("review 必须为对象")
        return
    if review.get("version") != "v1":
        errors.append("review version 必须为 v1")
    decision = review.get("decision")
    if decision not in REVIEW_DECISIONS:
        errors.append("review decision 必须为 pending、accepted 或 rejected")
    reviewers = review.get("reviewers")
    if not isinstance(reviewers, list) or any(
        not isinstance(value, str) or not value.strip() for value in reviewers
    ):
        errors.append("review reviewers 必须为字符串数组")
    reviewed_at = review.get("reviewedAt")
    if reviewed_at is not None and (
        not isinstance(reviewed_at, str) or not reviewed_at.strip()
    ):
        errors.append("review reviewedAt 为空时必须为 null，否则必须为非空字符串")
    if decision in {"accepted", "rejected"}:
        if not reviewers:
            errors.append("review 已完成时 reviewers 不能为空")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            errors.append("review 已完成时 reviewedAt 必须为非空字符串")
    reviewed_checksums = review.get("reviewedChecksums")
    if reviewed_checksums != manifest.get("checksums"):
        errors.append("review reviewedChecksums 必须与 manifest checksums 一致")
    accepted = review.get("acceptedCaseIds")
    rejected = review.get("rejectedCaseIds")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        errors.append("review acceptedCaseIds 和 rejectedCaseIds 必须为数组")
        return
    accepted_set = set(accepted)
    rejected_set = set(rejected)
    if len(accepted_set) != len(accepted) or len(rejected_set) != len(rejected):
        errors.append("review acceptedCaseIds 和 rejectedCaseIds 不能包含重复 ID")
    overlap = sorted(accepted_set & rejected_set)
    if overlap:
        errors.append(f"review acceptedCaseIds 与 rejectedCaseIds 不能重叠: {', '.join(overlap)}")
    case_ids = {case.id for case in cases}
    unknown_ids = sorted((accepted_set | rejected_set) - case_ids)
    if unknown_ids:
        errors.append(f"review 包含未知 case ID: {', '.join(unknown_ids)}")
    manifest_approved = manifest.get("approved")
    if manifest_approved is True and decision != "accepted":
        errors.append("manifest approved 为 true 时，review decision 必须为 accepted")
    if decision == "accepted" and manifest_approved is not True:
        errors.append("review decision 为 accepted 时，manifest approved 必须为 true")


def _expected_split_ids(split: str) -> list[str]:
    """expected split ids 的内部辅助逻辑。"""
    if split == "development":
        return (
            [f"hse-no-tool-{index:03d}" for index in range(1, 11)]
            + [f"hse-l1-{index:03d}" for index in range(1, 26)]
            + [f"hse-l2-{index:03d}" for index in range(1, 16)]
            + [f"hse-l3-{index:03d}" for index in range(1, 16)]
            + [f"hse-multi-tool-{index:03d}" for index in range(1, 6)]
        )
    return (
        [f"hse-multi-tool-{index:03d}" for index in range(6, 16)]
        + [f"hse-boundary-{index:03d}" for index in range(1, 11)]
        + [f"hse-security-{index:03d}" for index in range(1, 6)]
        + [f"hse-failure-{index:03d}" for index in range(1, 6)]
    )


def _mentioned_tools(case: EvalCase) -> set[str]:
    """mentioned tools 的内部辅助逻辑。"""
    tools = (
        set(case.required_tools)
        | set(case.forbidden_tools)
        | set(case.expected_arguments)
    )
    for sequence in case.allowed_tool_sequences:
        tools.update(sequence)
    return tools
