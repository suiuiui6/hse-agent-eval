from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class EvalCase(CamelModel):
    id: str = Field(pattern=r"^hse-[a-z0-9-]+-\d{3}$")
    dataset_split: Literal["development", "regression"]
    category: Literal["no_tool", "l1", "l2", "l3", "multi_tool", "boundary", "security", "failure"]
    risk_level: Literal["low", "medium", "high", "critical"]
    role: Literal["admin", "hse_user", "other_dept_user"]
    messages: list[str] = Field(min_length=1)
    fixture: str
    expected_outcome: str
    required_tools: list[str]
    forbidden_tools: list[str]
    allowed_tool_sequences: list[list[str]] = Field(min_length=1)
    expected_arguments: dict[str, dict[str, Any]]
    confirmation_required: bool
    confirmation_action: Literal["confirm", "cancel"] = "confirm"
    confirmation_attempts: int = Field(default=1, ge=1, le=2)
    expected_side_effects: list[str]
    forbidden_side_effects: list[str]
    forbidden_output_patterns: list[str] = Field(default_factory=list)
    max_rounds: int = Field(ge=1, le=5)
    tags: list[str] = Field(min_length=1)
    failure_mode: Literal[
        "none", "model_500", "model_empty", "model_unknown_tool", "model_bad_arguments", "graphrag_timeout"
    ] = "none"

    @model_validator(mode="after")
    def validate_tool_sets(self) -> "EvalCase":
        overlap = set(self.required_tools) & set(self.forbidden_tools)
        if overlap:
            raise ValueError(f"工具同时出现在 required 和 forbidden: {sorted(overlap)}")
        return self


class ToolCall(CamelModel):
    tool: str
    args: dict[str, Any]
    level: str
    result: str
    round_index: int
    status: str | None = None
    error_code: str | None = None
    schema_version: str | None = None
    tool_call_id: str | None = None
    result_ref: str | None = None
    idempotent_replay: bool | None = None


class SideEffect(CamelModel):
    key: str
    before_count: int
    after_count: int
    changed_rows: list[dict[str, Any]]


class RunRecord(CamelModel):
    run_id: str
    case_id: str
    repetition: int
    attempt: int = Field(default=1, ge=1, le=2)
    reply: str
    status: str
    tenant_id: str
    owner_id: str
    session_id: str
    turn_id: str
    trace_id: str
    stop_reason: str
    execution_status: str
    degraded_mode: str
    result_refs: list[str]
    finish_reasons: list[str]
    tool_durations_ms: dict[str, int]
    tool_calls: list[ToolCall]
    side_effects: list[SideEffect]
    trajectory_incomplete: bool = False
    round_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_duration_ms: int
    tool_duration_ms: int
    duration_ms: int
    model: str
    provider: str
    confirmation_state: str
    confirmation_before_side_effects: bool | None = None
    side_effect_committed: bool
    error: str | None = None

    @model_validator(mode="after")
    def mark_incomplete_when_security_evidence_blank(self) -> "RunRecord":
        critical_values = [
            self.tenant_id,
            self.owner_id,
            self.session_id,
            self.turn_id,
            self.trace_id,
            self.stop_reason,
            self.execution_status,
            self.degraded_mode,
        ]
        has_blank_critical = any(
            not isinstance(value, str) or not value.strip() for value in critical_values
        )
        has_finish_reason = any(
            isinstance(value, str) and value.strip() for value in self.finish_reasons
        )
        self.trajectory_incomplete = (
            self.trajectory_incomplete or has_blank_critical or not has_finish_reason
        )
        return self
