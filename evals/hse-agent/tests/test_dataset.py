import hashlib
import json

import pytest

from hse_agent_evals.dataset import load_cases, update_manifest_checksums, validate_suite


CATEGORY_COUNTS = {
    "no_tool": 10,
    "l1": 25,
    "l2": 15,
    "l3": 15,
    "multi_tool": 15,
    "boundary": 10,
    "security": 5,
    "failure": 5,
}

DEVELOPMENT_COUNTS = {
    "no_tool": 7,
    "l1": 18,
    "l2": 10,
    "l3": 10,
    "multi_tool": 10,
    "boundary": 7,
    "security": 4,
    "failure": 4,
}

REGRESSION_COUNTS = {
    "no_tool": 3,
    "l1": 7,
    "l2": 5,
    "l3": 5,
    "multi_tool": 5,
    "boundary": 3,
    "security": 1,
    "failure": 1,
}


def make_case(case_number, split, category):
    """处理 make case 相关逻辑。"""
    return {
        "id": f"hse-{category.replace("_", "-")}-{case_number:03d}",
        "datasetSplit": split,
        "category": category,
        "riskLevel": "low",
        "role": "hse_user",
        "messages": [f"查询安全规范 {case_number}"],
        "fixture": "base_seed",
        "expectedOutcome": "返回结果",
        "requiredTools": [],
        "forbiddenTools": ["create_task"] if category == "security" else [],
        "allowedToolSequences": [[]],
        "expectedArguments": {},
        "confirmationRequired": False,
        "expectedSideEffects": [],
        "forbiddenSideEffects": ["task.insert"] if category == "security" else [],
        "maxRounds": 3,
        "tags": ["dataset"],
    }


def make_suite():
    """处理 make suite 相关逻辑。"""
    development = []
    regression = []
    case_number = 0
    for category, count in DEVELOPMENT_COUNTS.items():
        for _ in range(count):
            case_number += 1
            development.append(make_case(case_number, "development", category))
    for category, count in REGRESSION_COUNTS.items():
        for _ in range(count):
            case_number += 1
            regression.append(make_case(case_number, "regression", category))
    return development, regression


def make_manifest():
    """处理 make manifest 相关逻辑。"""
    return {
        "version": "v1",
        "dataset": {
            "splits": {"development": 70, "regression": 30},
            "categoryCounts": CATEGORY_COUNTS,
        },
        "checksums": {"development": "0" * 64, "regression": "0" * 64},
        "approved": False,
    }


def write_jsonl(path, cases):
    """处理 write jsonl 相关逻辑。"""
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def test_validate_suite_accepts_exactly_100_cases_with_required_splits_and_categories():
    """验证 validate suite accepts exactly 100 cases with required splits and categories 场景。"""
    development, regression = make_suite()

    assert len(development) == 70
    assert len(regression) == 30
    assert validate_suite(development, regression, make_manifest()) == []


def test_validate_suite_rejects_duplicate_ids_across_splits():
    """验证 validate suite rejects duplicate ids across splits 场景。"""
    development, regression = make_suite()
    regression[0]["id"] = development[0]["id"]

    errors = validate_suite(development, regression, make_manifest())

    assert any("重复 ID" in error for error in errors)


def test_validate_suite_rejects_unknown_side_effect_key():
    """验证 validate suite rejects unknown side effect key 场景。"""
    development, regression = make_suite()
    development[0]["forbiddenSideEffects"] = ["unknown.insert"]

    errors = validate_suite(development, regression, make_manifest())

    assert any("未知副作用键" in error for error in errors)


def test_validate_suite_rejects_missing_chinese_content():
    """验证 validate suite rejects missing chinese content 场景。"""
    development, regression = make_suite()
    development[0]["messages"] = ["hello"]

    errors = validate_suite(development, regression, make_manifest())

    assert any("缺少中文" in error for error in errors)


def test_validate_suite_rejects_duplicate_messages():
    """验证 validate suite rejects duplicate messages 场景。"""
    development, regression = make_suite()
    development[1]["messages"] = development[0]["messages"]

    errors = validate_suite(development, regression, make_manifest())

    assert any("重复用例消息" in error for error in errors)


def test_validate_suite_rejects_unknown_fixture():
    """验证 validate suite rejects unknown fixture 场景。"""
    development, regression = make_suite()
    development[0]["fixture"] = "regulations"

    errors = validate_suite(development, regression, make_manifest())

    assert any("不支持的 fixture" in error for error in errors)


def test_validate_suite_rejects_case_in_wrong_split():
    """验证 validate suite rejects case in wrong split 场景。"""
    development, regression = make_suite()
    regression[0]["datasetSplit"] = "development"

    errors = validate_suite(development, regression, make_manifest())

    assert any("regression" in error and "datasetSplit" in error for error in errors)


def test_validate_suite_rejects_wrong_category_count():
    """验证 validate suite rejects wrong category count 场景。"""
    development, regression = make_suite()
    regression[0]["category"] = "l1"

    errors = validate_suite(development, regression, make_manifest())

    assert any("no_tool" in error and "类别数量" in error for error in errors)


def test_validate_suite_rejects_wrong_failure_case_count():
    """验证 validate suite rejects wrong failure case count 场景。"""
    development, regression = make_suite()
    regression[0]["category"] = "failure"

    errors = validate_suite(development, regression, make_manifest())

    assert any("failure" in error and "类别数量" in error for error in errors)


def test_load_cases_reads_jsonl_into_eval_cases(tmp_path):
    """验证 load cases reads jsonl into eval cases 场景。"""
    development, _ = make_suite()
    path = tmp_path / "development.v1.jsonl"
    write_jsonl(path, development[:2])

    cases = load_cases(path)

    assert [case.id for case in cases] == ["hse-no-tool-001", "hse-no-tool-002"]
    assert cases[0].dataset_split == "development"


def test_update_manifest_checksums_hashes_case_files_without_approving_dataset(tmp_path):
    """验证 update manifest checksums hashes case files without approving dataset 场景。"""
    development, regression = make_suite()
    development_path = tmp_path / "development.v1.jsonl"
    regression_path = tmp_path / "regression.v1.jsonl"
    manifest_path = tmp_path / "manifest.v1.json"
    write_jsonl(development_path, development)
    write_jsonl(regression_path, regression)
    manifest_path.write_text(json.dumps(make_manifest()), encoding="utf-8")

    update_manifest_checksums(development_path, regression_path, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checksums"] == {
        "development": hashlib.sha256(development_path.read_bytes()).hexdigest(),
        "regression": hashlib.sha256(regression_path.read_bytes()).hexdigest(),
    }
    assert manifest["approved"] is False

def test_rejects_underscore_id_and_invalid_manifest_checksums():
    """验证 rejects underscore id and invalid manifest checksums 场景。"""
    development, regression = make_suite()
    development[0]["id"] = "hse-no_tool-001"
    manifest = make_manifest()
    manifest["checksums"] = {"development": "", "regression": "z" * 64}
    errors = validate_suite(development, regression, manifest)
    assert any("无效" in error for error in errors)
    assert any("checksums" in error for error in errors)


def test_load_cases_reports_invalid_json_and_fields(tmp_path):
    """验证 load cases reports invalid json and fields 场景。"""
    valid = make_suite()[0][0]
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_cases(path)


def test_load_cases_reports_invalid_case_fields(tmp_path):
    """验证 load cases reports invalid case fields 场景。"""
    path = tmp_path / "invalid-case.jsonl"
    path.write_text(json.dumps({"id": "bad"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 1 行用例无效"):
        load_cases(path)
def test_update_manifest_checksums_accepts_utf8_bom_manifest(tmp_path):
    """验证 update manifest checksums accepts utf8 bom manifest 场景。"""
    development, regression = make_suite()
    development_path = tmp_path / "development.v1.jsonl"
    regression_path = tmp_path / "regression.v1.jsonl"
    manifest_path = tmp_path / "manifest.v1.json"
    write_jsonl(development_path, development)
    write_jsonl(regression_path, regression)
    manifest_path.write_text(json.dumps(make_manifest()), encoding="utf-8-sig")

    update_manifest_checksums(development_path, regression_path, manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["checksums"]["development"]) == 64


def test_load_cases_accepts_utf8_bom_on_first_line(tmp_path):
    """验证 load cases accepts utf8 bom on first line 场景。"""
    path = tmp_path / "development.v1.jsonl"
    path.write_text(json.dumps(make_suite()[0][0]) + "\n", encoding="utf-8-sig")
    assert load_cases(path)[0].id == "hse-no-tool-001"
