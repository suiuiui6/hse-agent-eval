import json
from pathlib import Path

from hse_agent_evals.dataset import (
    PRODUCTION_TOOL_NAMES,
    REVIEW_MATRIX_TAGS,
    load_cases,
    validate_production_bundle,
)


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"


def _mentioned_tools(case):
    tools = set(case.required_tools) | set(case.forbidden_tools) | set(case.expected_arguments)
    for sequence in case.allowed_tool_sequences:
        tools.update(sequence)
    return tools


def test_validate_production_bundle_accepts_approved_dataset_after_review(tmp_path):
    development_path = CASES_DIR / "development.v1.jsonl"
    regression_path = CASES_DIR / "regression.v1.jsonl"
    manifest = json.loads((CASES_DIR / "manifest.v1.json").read_text(encoding="utf-8-sig"))
    cases = load_cases(development_path) + load_cases(regression_path)

    manifest["approved"] = True
    review = {
        "version": "v1",
        "decision": "accepted",
        "reviewers": ["/root"],
        "reviewedAt": "2026-07-22T00:00:00+08:00",
        "reviewedChecksums": manifest["checksums"],
        "acceptedCaseIds": [case.id for case in cases],
        "rejectedCaseIds": [],
    }

    tmp_development = tmp_path / "development.v1.jsonl"
    tmp_regression = tmp_path / "regression.v1.jsonl"
    tmp_manifest = tmp_path / "manifest.v1.json"
    tmp_review = tmp_path / "review.v1.json"
    tmp_development.write_bytes(development_path.read_bytes())
    tmp_regression.write_bytes(regression_path.read_bytes())
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_review.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert validate_production_bundle(
        tmp_development,
        tmp_regression,
        tmp_manifest,
        tmp_review,
    ) == []

def test_production_dataset_matches_task12_contract():
    development_path = CASES_DIR / "development.v1.jsonl"
    regression_path = CASES_DIR / "regression.v1.jsonl"
    manifest_path = CASES_DIR / "manifest.v1.json"
    review_path = CASES_DIR / "review.v1.json"

    development = load_cases(development_path)
    regression = load_cases(regression_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))

    assert validate_production_bundle(
        development_path,
        regression_path,
        manifest_path,
        review_path,
    ) == []

    assert len(development) == 70
    assert len(regression) == 30

    all_cases = development + regression
    ids = [case.id for case in all_cases]
    assert len(ids) == len(set(ids))

    mentioned_tools = set()
    review_tags = set()
    for case in all_cases:
        mentioned_tools.update(_mentioned_tools(case))
        review_tags.update(tag for tag in case.tags if tag.startswith("review-matrix-"))

    assert mentioned_tools == PRODUCTION_TOOL_NAMES
    assert review_tags == REVIEW_MATRIX_TAGS

    assert manifest["approved"] is False
    assert review["reviewedChecksums"] == manifest["checksums"]
    assert review["decision"] == "pending"
    assert review["reviewers"] == []
    assert review["reviewedAt"] is None
    assert review["acceptedCaseIds"] == []
    assert review["rejectedCaseIds"] == []


def test_modified_security_and_cancel_cases_have_explicit_assertions():
    cases = {
        case.id: case
        for case in load_cases(CASES_DIR / "development.v1.jsonl")
        + load_cases(CASES_DIR / "regression.v1.jsonl")
    }
    for case_id in ("hse-security-003", "hse-l3-005", "hse-l3-007"):
        case = cases[case_id]
        assert (
            case.forbidden_tools
            or case.forbidden_side_effects
            or case.forbidden_output_patterns
        )

    for case_id in ("hse-l3-005", "hse-l3-007"):
        case = cases[case_id]
        assert case.confirmation_required is True
        assert case.confirmation_action == "cancel"
        assert case.expected_side_effects == []
        assert case.forbidden_side_effects


