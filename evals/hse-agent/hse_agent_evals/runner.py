from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hse_agent_evals import environment
from hse_agent_evals.adapters.engine_client import EngineClient
from hse_agent_evals.adapters.fixture_manager import FixtureManager
from hse_agent_evals.adapters.trajectory import map_tool_calls
from hse_agent_evals.fixtures import GraphRAGFixtureServer, ModelProxyServer
from hse_agent_evals.guards import BackendBindingGuard, MySqlIntegrityGuard
from hse_agent_evals.models import EvalCase, RunRecord, SideEffect
from hse_agent_evals.settings import EvalSettings

logger = logging.getLogger(__name__)


def run_key(case_id: str, repetition: int, attempt: int) -> str:
    """去重键：一次评测尝试的唯一标识。"""
    return f"{case_id}|{repetition}|{attempt}"


def _text(value: Any) -> str:
    """text 的内部辅助逻辑。"""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _int(value: Any, fallback: int = 0) -> int:
    """int 的内部辅助逻辑。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return int(value)


MODEL_FAILURE_MODES = frozenset(
    {
        "model_500",
        "model_empty",
        "model_unknown_tool",
        "model_bad_arguments",
    }
)


def _url_port(url: str) -> int:
    """url port 的内部辅助逻辑。"""
    port = urlparse(url).port
    if port is None:
        raise ValueError(f"评测服务地址必须显式包含端口: {url}")
    return port


@contextmanager
def _managed_fixtures(settings: EvalSettings):
    """managed fixtures 的内部辅助逻辑。"""
    model_proxy = ModelProxyServer(
        port=_url_port(settings.model_proxy_url),
        upstream_url=settings.ollama_base_url.rstrip("/") + "/chat/completions",
    )
    graphrag_fixture = GraphRAGFixtureServer(
        port=_url_port(settings.graphrag_fixture_url),
    )
    try:
        model_proxy.start()
        graphrag_fixture.start()
        yield model_proxy, graphrag_fixture
    except OSError as exc:
        raise RuntimeError(
            "无法启动评测 fixture（"
            f"{settings.model_proxy_url} / {settings.graphrag_fixture_url}）: {exc}"
        ) from None
    finally:
        graphrag_fixture.stop()
        model_proxy.stop()


def _apply_case_failure_mode(
    case: EvalCase,
    model_proxy: ModelProxyServer | None,
    graphrag_fixture: GraphRAGFixtureServer | None,
) -> None:
    """apply case failure mode 的内部辅助逻辑。"""
    mode = case.failure_mode
    if mode == "none":
        if model_proxy is not None:
            model_proxy.set_mode("none")
        if graphrag_fixture is not None:
            graphrag_fixture.set_mode("none")
        return
    if mode in MODEL_FAILURE_MODES:
        if model_proxy is None:
            raise RuntimeError(f"failureMode={mode} 需要 runner 管理的模型代理")
        model_proxy.set_mode(mode)
        if graphrag_fixture is not None:
            graphrag_fixture.set_mode("none")
        return
    if mode == "graphrag_timeout":
        if graphrag_fixture is None:
            raise RuntimeError("failureMode=graphrag_timeout 需要 runner 管理的 GraphRAG fixture")
        graphrag_fixture.set_mode("graphrag_timeout")
        if model_proxy is not None:
            model_proxy.set_mode("none")
        return
    raise ValueError(f"未知 failureMode: {mode}")


def _reset_fixture_modes(
    model_proxy: ModelProxyServer | None,
    graphrag_fixture: GraphRAGFixtureServer | None,
) -> None:
    """reset fixture modes 的内部辅助逻辑。"""
    if model_proxy is not None:
        model_proxy.set_mode("none")
    if graphrag_fixture is not None:
        graphrag_fixture.set_mode("none")


def synthesize_finish_reasons(response: Mapping[str, Any]) -> list[str]:
    """从后端真实发来的字段推导 finish_reasons，不伪造。

    这是**回退**口径：后端补发 finishReasons（HseAgentServiceImpl 现已发出）
    时优先用显式值，只有缺失时才走这里，由 stopReason / executionStatus /
    confirmationToken 确定性推导。
    """
    reasons: list[str] = []
    stop_reason = _text(response.get("stopReason"))
    if stop_reason:
        reasons.append(stop_reason)
    execution_status = _text(response.get("executionStatus"))
    if execution_status and execution_status != "success":
        reasons.append("error")
    if _text(response.get("confirmationToken")):
        reasons.append("confirmation")
    return reasons


def build_run_record(
    *,
    run_id: str,
    case: EvalCase,
    repetition: int,
    attempt: int,
    response: Mapping[str, Any],
    side_effects: Sequence[SideEffect],
    session_id: str,
    owner_id: str,
    tenant_id: str,
    model: str,
    provider: str,
    duration_ms: int,
    confirmation_before_side_effects: bool | None,
    side_effect_committed: bool,
    error: str | None = None,
) -> RunRecord:
    """响应 → RunRecord 的唯一映射点。

    宁可留空并让 RunRecord 校验器置 trajectory_incomplete=True，也不编造遥测。

    owner_id / tenant_id 由评测侧提供：后端响应不携带身份，从 response 取值只会得到
    空串，令每条记录都被判为证据不完整。二者仅作为校验器的证据完整性信号存在。
    """
    mapping = map_tool_calls(None, dict(response))
    total_tokens = _int(response.get("tokens"))
    if total_tokens == 0:
        total_tokens = _int(response.get("promptTokens")) + _int(response.get("completionTokens"))

    duration = _int(response.get("durationMs"), fallback=duration_ms)
    if duration == 0:
        duration = duration_ms

    # 后端已补发分阶段耗时（modelDurationMs/toolDurationMs），优先采用；
    # 缺失时退回「整轮计入模型、工具记 0」，绝不编造拆分。
    explicit_model = _int(response.get("modelDurationMs"), fallback=-1)
    explicit_tool = _int(response.get("toolDurationMs"), fallback=-1)
    if explicit_model >= 0 and explicit_tool >= 0:
        model_duration = explicit_model
        tool_duration = explicit_tool
    else:
        model_duration = duration
        tool_duration = 0
    tool_durations: dict[str, int] = {}

    explicit_reasons = response.get("finishReasons")
    if isinstance(explicit_reasons, list) and explicit_reasons:
        finish_reasons = [str(item) for item in explicit_reasons]
    else:
        finish_reasons = synthesize_finish_reasons(response)

    status = _text(response.get("status")) or _text(response.get("executionStatus"))
    if status == "pending" and _text(response.get("confirmationToken")):
        status = "pending_confirmation"
    execution_status = _text(response.get("executionStatus")) or status
    round_count = _int(response.get("roundCount"), fallback=-1)
    if round_count < 0:
        round_count = max(
            (call.round_index for call in mapping.tool_calls),
            default=1 if status else 0,
        )
    confirmation_state = _text(response.get("confirmationState"))
    if not confirmation_state:
        confirmation_state = "pending" if response.get("confirmationToken") else "not_required"

    return RunRecord(
        run_id=run_id,
        case_id=case.id,
        repetition=repetition,
        attempt=attempt,
        reply=_text(response.get("reply")),
        status=status,
        tenant_id=tenant_id,
        owner_id=owner_id,
        session_id=session_id,
        turn_id=_text(response.get("turnId")),
        trace_id=_text(response.get("traceId")),
        stop_reason=_text(response.get("stopReason")),
        execution_status=execution_status,
        degraded_mode=_text(response.get("degradedMode")),
        result_refs=[
            ref for ref in (call.result_ref for call in mapping.tool_calls) if isinstance(ref, str)
        ],
        finish_reasons=finish_reasons,
        tool_durations_ms=tool_durations,
        tool_calls=mapping.tool_calls,
        side_effects=list(side_effects),
        trajectory_incomplete=mapping.trajectory_incomplete,
        round_count=round_count,
        prompt_tokens=_int(response.get("promptTokens")),
        completion_tokens=_int(response.get("completionTokens")),
        total_tokens=total_tokens,
        model_duration_ms=model_duration,
        tool_duration_ms=tool_duration,
        duration_ms=duration,
        model=model,
        provider=provider,
        confirmation_state=confirmation_state,
        confirmation_before_side_effects=confirmation_before_side_effects,
        side_effect_committed=side_effect_committed,
        error=error,
    )


class LeaseUnavailable(RuntimeError):
    """原始结果正被另一个进程占用；并发评测必须显式失败而不是交错写。"""


def _lock_file(file_object: Any) -> None:
    """lock file 的内部辅助逻辑。"""
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(file_object.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(file_object.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file_object: Any) -> None:
    """unlock file 的内部辅助逻辑。"""
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(file_object.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file_object.fileno(), fcntl.LOCK_UN)


class RawRunStore:
    """append-only 不可变原始结果存储；续跑只读已有键，绝不覆盖既有 attempt。

    两个并发评测进程写同一 run 会交错出不可复现的结果，故写入路径整体受
    跨进程 lease 保护（Windows 用 msvcrt，POSIX 用 flock）。
    """

    def __init__(self, run_dir: str | Path) -> None:
        """初始化实例。"""
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.runs_path = self.raw_dir / "runs.jsonl"
        self.scores_path = self.raw_dir / "case-scores.jsonl"
        self.lease_path = Path(str(self.runs_path) + ".lease")

    @contextmanager
    def lease(self, timeout_seconds: float = 0.25):
        """独占整个 run；拿不到锁即抛 LeaseUnavailable，交由调用方退出非零。"""
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        lease_file = self.lease_path.open("a+b")
        acquired = False
        try:
            lease_file.seek(0, os.SEEK_END)
            if lease_file.tell() == 0:
                lease_file.write(b"0")
                lease_file.flush()
                os.fsync(lease_file.fileno())
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while True:
                lease_file.seek(0)
                try:
                    _lock_file(lease_file)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise LeaseUnavailable("评测结果正在被其他进程使用") from None
                    time.sleep(0.02)
            yield
        finally:
            try:
                if acquired:
                    lease_file.seek(0)
                    _unlock_file(lease_file)
            finally:
                lease_file.close()

    def existing_keys(self) -> set[str]:
        """处理 existing keys 相关逻辑。"""
        return {run_key(record.case_id, record.repetition, record.attempt) for record in self.load()}

    def load(self) -> list[RunRecord]:
        """处理 load 相关逻辑。"""
        if not self.runs_path.exists():
            return []
        records: list[RunRecord] = []
        lines = self.runs_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                if index == len(lines) - 1:
                    # 仅容忍最后一行（append 中途 crash），其余视为损坏
                    continue
                raise ValueError(f"原始结果第 {index + 1} 行损坏") from None
            records.append(RunRecord.model_validate(payload))
        return records

    # ---- 评分行：无 lease 需求（只在 runs.append 之后紧随写入）----

    def append(self, record: RunRecord) -> None:
        """先落 tmp 并 fsync，再追加到正式文件 —— crash 时正式文件不会留半行。

        重复 key 直接拒绝：覆盖会让「同一份原始结果」不可复现。
        """
        key = run_key(record.case_id, record.repetition, record.attempt)
        if key in self.existing_keys():
            raise ValueError(f"拒绝覆盖已存在结果: {key}")
        payload = record.model_dump(by_alias=True, mode="json")
        line = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.runs_path.with_suffix(self.runs_path.suffix + ".tmp")
        with tmp_path.open("wb") as temporary:
            temporary.write(line)
            temporary.flush()
            os.fsync(temporary.fileno())
        with self.runs_path.open("ab") as final:
            final.write(line)
            final.flush()
            os.fsync(final.fileno())
        tmp_path.unlink()

    def append_scores(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """处理 append scores 相关逻辑。"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with self.scores_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                line = json.dumps(
                    dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_scores(self) -> list[dict[str, Any]]:
        """处理 load scores 相关逻辑。"""
        if not self.scores_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        lines = self.scores_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                if index == len(lines) - 1:
                    continue
                raise ValueError(f"评分结果第 {index + 1} 行损坏") from None
        return rows


@dataclass(frozen=True)
class RunOutcome:
    """单次尝试的结果。fatal=True 表示污染/清理失败，baseline 也必须非零退出。"""

    record: RunRecord | None
    error: str | None
    fatal: bool
    cleanup_error: str | None = None
    # 分阶段耗时是否来自后端显式字段。必须由知道 response 的一侧记录，
    # 不能在只拿到 RunRecord 的地方靠数值形状反推 —— 单轮无工具的
    # stop 用例其 tool=0/model=duration，形状与回退口径完全一样。
    staged_durations: bool = False


def _confirmation_request(
    response: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str]:
    # 后端实际写入的键是 "tool"（HseAgentServiceImpl.java:267），前端 helpers 兼容 toolName，
    # 此处同样两者都认，避免因键名差异丢掉确认链路。
    """confirmation request 的内部辅助逻辑。"""
    details = response.get("confirmation")
    candidates = details if isinstance(details, list) else [details]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        tool_name = _text(candidate.get("tool")) or _text(candidate.get("toolName"))
        if tool_name:
            args = candidate.get("args")
            level = _text(candidate.get("level"))
            return tool_name, dict(args) if isinstance(args, Mapping) else {}, level
    return "", {}, ""


def _merge_confirmation_response(
    initial: Mapping[str, Any],
    confirmed: Mapping[str, Any],
    *,
    tool_name: str,
    args: Mapping[str, Any],
    level: str,
) -> dict[str, Any]:
    """把工具确认结果合回聊天响应，保留首轮遥测和已发生的确认证据。

    `/hse/agent/confirm` 返回工具结果 envelope，不是聊天响应。直接把它当
    RunRecord 的来源会丢掉 reply、roundCount、executionStatus 和原始 toolCalls，
    并把所有确认用例误判为 trajectory_incomplete。
    """
    merged = dict(initial)
    confirm_status = _text(confirmed.get("status"))
    if confirm_status in {"success", "empty"}:
        final_status = "success"
        execution_status = "success"
        confirmation_state = "confirmed"
        stop_reason = "completed"
        reply = "操作已确认并执行。"
    elif confirm_status == "cancelled":
        final_status = "cancelled"
        execution_status = "cancelled"
        confirmation_state = "cancelled"
        stop_reason = "cancelled"
        reply = "操作已取消，未执行。"
    elif confirm_status == "pending":
        final_status = "pending"
        execution_status = "pending"
        confirmation_state = "pending"
        stop_reason = "confirmation_required"
        reply = "操作正在执行，请稍后查询。"
    else:
        final_status = "error"
        execution_status = "error"
        confirmation_state = "failed"
        stop_reason = "error"
        reply = "操作未完成，请查看错误信息。"

    merged["status"] = final_status
    merged["executionStatus"] = execution_status
    merged["confirmationState"] = confirmation_state
    merged["stopReason"] = stop_reason
    merged["finishReasons"] = ["stop"] if final_status == "success" else ["error"]
    initial_reply = _text(initial.get("reply"))
    merged["reply"] = f"{initial_reply}\n{reply}" if initial_reply else reply
    merged.pop("confirmationToken", None)
    merged["roundCount"] = max(1, _int(initial.get("roundCount"), fallback=1))

    meta = confirmed.get("meta")
    meta = dict(meta) if isinstance(meta, Mapping) else {}
    trace_id = _text(meta.get("traceId")) or _text(initial.get("traceId"))
    turn_id = _text(meta.get("turnId")) or _text(initial.get("turnId"))
    if trace_id:
        merged["traceId"] = trace_id
    if turn_id:
        merged["turnId"] = turn_id

    raw_calls = initial.get("toolCalls")
    calls = (
        [dict(call) for call in raw_calls if isinstance(call, Mapping)]
        if isinstance(raw_calls, list)
        else []
    )
    max_round = max((_int(call.get("roundIndex")) for call in calls), default=0)
    confirmed_call = {
        "tool": _text(confirmed.get("tool")) or tool_name,
        "toolCallId": _text(meta.get("toolCallId")) or f"confirm-{tool_name}",
        "args": dict(args),
        "level": level,
        "status": confirm_status or "error",
        "roundIndex": max_round + 1,
        "result": json.dumps(
            dict(confirmed),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for index in range(len(calls) - 1, -1, -1):
        if _text(calls[index].get("tool")) == tool_name:
            calls[index] = confirmed_call
            break
    else:
        calls.append(confirmed_call)
    merged["toolCalls"] = calls
    return merged


def _merge_turn_responses(
    prior: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    """merge turn responses 的内部辅助逻辑。"""
    if not prior:
        return dict(final)
    merged = dict(final)
    calls: list[dict[str, Any]] = []
    offset = 0
    for turn in prior:
        raw_calls = turn.get("toolCalls")
        turn_calls = (
            [dict(call) for call in raw_calls if isinstance(call, Mapping)]
            if isinstance(raw_calls, list)
            else []
        )
        for call in turn_calls:
            call["roundIndex"] = _int(call.get("roundIndex"), fallback=1) + offset
            calls.append(call)
        span = _int(turn.get("roundCount"))
        if span <= 0:
            span = max(
                (_int(call.get("roundIndex"), fallback=1) for call in turn_calls),
                default=1,
            )
        offset += max(1, span)

    raw_final_calls = final.get("toolCalls")
    final_calls = (
        [dict(call) for call in raw_final_calls if isinstance(call, Mapping)]
        if isinstance(raw_final_calls, list)
        else []
    )
    for call in final_calls:
        call["roundIndex"] = _int(call.get("roundIndex"), fallback=1) + offset
        calls.append(call)
    final_span = _int(final.get("roundCount"))
    if final_span <= 0:
        final_span = max(
            (_int(call.get("roundIndex"), fallback=1) - offset for call in final_calls),
            default=1,
        )
    merged["toolCalls"] = calls
    merged["roundCount"] = offset + max(1, final_span)
    return merged


def run_case(
    *,
    case: EvalCase,
    run_id: str,
    repetition: int,
    attempt: int,
    settings: EvalSettings,
    client: EngineClient,
    fixture_manager: FixtureManager,
    token: str,
    owner_id: str,
    tenant_id: str,
    runtime_dir: str | Path,
    clock: Callable[[], float] = time.monotonic,
    model_proxy: ModelProxyServer | None = None,
    graphrag_fixture: GraphRAGFixtureServer | None = None,
) -> RunOutcome:
    """执行单条用例：重置库 → 前快照 → 逐条消息 chat → (确认) → 后快照 → 构造 RunRecord。

    owner_id / tenant_id 由调用方按用例角色提供（后端响应不含身份）。
    """
    session_id = f"{run_id}-{case.id}-r{repetition}-a{attempt}"
    started = clock()
    try:
        database = environment.database_name(run_id, runtime_dir)
        environment.reset_case(run_id, fixture_manager, runtime_dir)
    except Exception as exc:  # noqa: BLE001 — 基础设施失败须保留证据而非中断整轮
        logger.exception("重置用例失败 case=%s r%s a%s", case.id, repetition, attempt)
        return RunOutcome(None, f"重置用例失败: {exc}", False)

    try:
        _apply_case_failure_mode(case, model_proxy, graphrag_fixture)
        before = fixture_manager.snapshot_before(database)
        responses: list[dict[str, Any]] = []
        for message in case.messages:
            responses.append(dict(client.chat(token, message, session_id)))
        response = responses[-1]
        initial_response = dict(response)

        confirmation_before: bool | None = None
        if case.confirmation_required and _text(response.get("confirmationToken")):
            mid = fixture_manager.snapshot_after(database)
            pre_confirm = fixture_manager.diff(before, mid)
            confirmation_before = len(pre_confirm) == 0
            tool_name, args, level = _confirmation_request(response)
            if not tool_name:
                return RunOutcome(None, "确认待定但响应缺少工具信息", False)
            if case.confirmation_action == "cancel":
                confirmed = dict(
                    client.cancel(
                        token,
                        _text(response.get("confirmationToken")),
                        tool_name,
                        args,
                        session_id,
                        _text(response.get("turnId")),
                    )
                )
            else:
                confirmed = dict(
                    client.confirm(
                        token,
                        _text(response.get("confirmationToken")),
                        tool_name,
                        args,
                        session_id,
                        _text(response.get("turnId")),
                    )
                )
                for _ in range(1, case.confirmation_attempts):
                    confirmed = dict(
                        client.confirm(
                            token,
                            _text(response.get("confirmationToken")),
                            tool_name,
                            args,
                            session_id,
                            _text(response.get("turnId")),
                        )
                    )
            response = _merge_confirmation_response(
                initial_response,
                confirmed,
                tool_name=tool_name,
                args=args,
                level=level,
            )
        if len(responses) > 1:
            response = _merge_turn_responses(responses[:-1], response)

        after = fixture_manager.snapshot_after(database)
        effects = fixture_manager.diff(before, after)
        duration_ms = max(0, int((clock() - started) * 1000))
        # 来源判定在此处完成（response 可见），随后随记录一起传递
        staged = _int(response.get("modelDurationMs"), fallback=-1) >= 0 and _int(
            response.get("toolDurationMs"), fallback=-1
        ) >= 0
        record = build_run_record(
            run_id=run_id,
            case=case,
            repetition=repetition,
            attempt=attempt,
            response=response,
            side_effects=effects,
            session_id=session_id,
            owner_id=owner_id,
            tenant_id=tenant_id,
            model=settings.model,
            provider=settings.model_proxy_url,
            duration_ms=duration_ms,
            confirmation_before_side_effects=confirmation_before,
            side_effect_committed=bool(effects),
        )
        return RunOutcome(record, None, False, staged_durations=staged)
    except Exception as exc:  # noqa: BLE001 — 单例失败不得中断整轮
        logger.exception("执行用例失败 case=%s r%s a%s", case.id, repetition, attempt)
        return RunOutcome(None, f"执行用例失败: {exc}", False)
    finally:
        _reset_fixture_modes(model_proxy, graphrag_fixture)


from hse_agent_evals.evaluators import (
    EfficiencyEvaluator,
    SafetyEvaluator,
    SideEffectEvaluator,
    TaskOutcomeEvaluator,
    ToolAccuracyEvaluator,
)
from strands_evals.types import EvaluationData

DEFAULT_EVALUATORS = (
    TaskOutcomeEvaluator(),
    ToolAccuracyEvaluator(),
    SafetyEvaluator(),
    SideEffectEvaluator(),
    EfficiencyEvaluator(),
)

EVAL_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExecutionPlan:
    """处理 ExecutionPlan 相关逻辑。"""
    cases: list[EvalCase]
    repetitions: int
    mode: str
    dataset: str
    rerun_failures: bool


@dataclass(frozen=True)
class RunSummary:
    """处理 RunSummary 相关逻辑。"""
    run_id: str
    planned_attempts: int
    executed: int
    skipped: int
    failed: int
    fatal: bool
    exit_code: int


def plan_execution(
    cases: Sequence[EvalCase],
    *,
    dataset: str,
    repetitions: int,
    mode: str,
    rerun_failures: bool,
) -> ExecutionPlan:
    """处理 plan execution 相关逻辑。"""
    if repetitions < 1 or repetitions > 5:
        raise ValueError("重复次数必须在 1..5 之间")
    if mode not in {"baseline", "gate"}:
        raise ValueError(f"未知模式: {mode}")
    if dataset not in {"development", "regression", "all"}:
        raise ValueError(f"未知数据集: {dataset}")
    selected = [case for case in cases if dataset == "all" or case.dataset_split == dataset]
    return ExecutionPlan(
        cases=list(selected),
        repetitions=repetitions,
        mode=mode,
        dataset=dataset,
        rerun_failures=rerun_failures,
    )


def run_directory(runtime_dir: str | Path, run_id: str) -> Path:
    """一次运行的全部产物目录（CLI 与编排共用，避免路径实现分叉）。"""
    return Path(runtime_dir) / "runs" / run_id


IDENTITIES_PATH = EVAL_ROOT / "fixtures" / "identities" / "users.json"


def _load_identities(settings: EvalSettings) -> dict[str, dict[str, Any]]:
    """返回 {角色: {"token": ..., "ownerId": ..., "tenantId": ...}}。

    身份夹具结构：{角色: {user_id, username, password, dept_id, ...}}。
    后端响应不含身份，故 ownerId/tenantId 由此提供（见文件头说明）。
    """
    payload = json.loads(IDENTITIES_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("评测身份夹具格式无效")
    identities: dict[str, dict[str, Any]] = {}
    with EngineClient(
        base_url=settings.engine_base_url, timeout=settings.request_timeout_seconds
    ) as client:
        for role, value in payload.items():
            account = dict(value)
            identities[str(role)] = {
                "token": client.login(str(account["username"]), str(account["password"])),
                "ownerId": str(account["user_id"]),
                "tenantId": f"dept:{account['dept_id']}",
            }
    return identities


def verify_backend_binding(
    run_id: str,
    settings: EvalSettings,
    runtime_dir: str | Path = EVAL_ROOT / ".runtime",
) -> dict[str, Any]:
    """处理 verify backend binding 相关逻辑。"""
    state = environment.run_state(run_id, runtime_dir)
    expected = _text(state.get("database_name"))
    if not expected or not expected.startswith("engine_agent_eval_"):
        raise ValueError("运行状态中的评测数据库名无效")
    payload = json.loads(IDENTITIES_PATH.read_text(encoding="utf-8-sig"))
    identity = payload.get("system_admin") if isinstance(payload, Mapping) else None
    if not isinstance(identity, Mapping):
        raise ValueError("评测身份夹具缺少 system_admin")
    with EngineClient(
        base_url=settings.engine_base_url,
        timeout=settings.request_timeout_seconds,
    ) as client:
        token = client.login(
            str(identity["username"]),
            str(identity["password"]),
        )
        binding = client.binding(token)
    actual = _text(binding.get("database"))
    if actual != expected:
        raise RuntimeError(
            f"后端未绑定到评测数据库: {actual!r} != {expected!r}"
        )
    return {
        "runId": run_id,
        "database": actual,
        "profile": _text(binding.get("profile")),
    }


def _score_record(
    case: EvalCase,
    record: RunRecord,
    *,
    staged_durations: bool,
    evaluators=DEFAULT_EVALUATORS,
) -> list[dict[str, Any]]:
    """评分行。staged_durations 由调用方传入真实来源，不在此反推。"""
    rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        output = evaluator.evaluate(EvaluationData(input=case, actual_output=record))[0]
        rows.append(
            {
                "caseId": case.id,
                "repetition": record.repetition,
                "attempt": record.attempt,
                "label": output.label,
                "score": output.score,
                "testPass": output.test_pass,
                "reason": output.reason,
                "telemetry": {"stagedDurations": staged_durations},
            }
        )
    return rows


def run_all(
    plan: ExecutionPlan,
    *,
    run_id: str,
    settings: EvalSettings,
    runtime_dir: str | Path = EVAL_ROOT / ".runtime",
    evaluators=DEFAULT_EVALUATORS,
    manage_fixtures: bool = False,
) -> RunSummary:
    """处理 run all 相关逻辑。"""
    store = RawRunStore(run_directory(runtime_dir, run_id))
    if not manage_fixtures and any(
        case.failure_mode != "none" for case in plan.cases
    ):
        raise ValueError(
            "计划包含 failureMode 用例；请使用 --manage-fixtures 或受控代理"
        )
    identities = _load_identities(settings)
    # 先校验全部角色再动库：避免跑到一半才发现身份缺失，
    # 留下半截记录且既不返回 RunSummary 也拿不到结构化退出码。
    missing_roles = sorted({case.role for case in plan.cases} - set(identities))
    if missing_roles:
        raise ValueError(f"评测身份夹具缺少角色: {', '.join(missing_roles)}")
    planned = len(plan.cases) * plan.repetitions
    executed = skipped = failed = 0
    fatal = False
    # 续跑判定用内存键集，避免每次尝试都全量重解析 JSONL
    existing = store.existing_keys()
    fixture_manager = environment.fixture_manager(settings)
    fixture_context = (
        _managed_fixtures(settings)
        if manage_fixtures
        else nullcontext((None, None))
    )

    # 整个 run 独占：两个评测进程同时写会交错出不可复现的结果
    with store.lease(), _client(settings) as client, fixture_context as fixtures:
        model_proxy, graphrag_fixture = fixtures
        # 执行前复验隔离，并把「可变范围之外」的数据哈希钉下来，
        # 供结束后比对 —— 评测污染了不该动的数据必须当场暴露
        state = environment.run_state(run_id, runtime_dir)
        BackendBindingGuard(state, settings).validate(run_id)
        integrity = MySqlIntegrityGuard(fixture_manager, state["database_name"])
        try:
            before_hash = integrity.immutable_hash(run_id)
        except Exception as exc:  # noqa: BLE001 — 隔离前提不成立时不该继续跑
            logger.exception("完整性检查失败 run=%s", run_id)
            raise RuntimeError(f"评测隔离前提不成立，已中止: {exc}") from exc
        for case in plan.cases:
            identity = identities[case.role]
            for repetition in range(1, plan.repetitions + 1):
                primary_key = run_key(case.id, repetition, 1)
                if primary_key in existing and not (
                    plan.rerun_failures and _is_failed(store, existing, case.id, repetition, 1)
                ):
                    skipped += 1
                    continue

                attempt = 1
                if plan.rerun_failures and primary_key in existing:
                    if run_key(case.id, repetition, 2) in existing:
                        skipped += 1
                        continue
                    attempt = 2

                outcome = run_case(
                    case=case,
                    run_id=run_id,
                    repetition=repetition,
                    attempt=attempt,
                    settings=settings,
                    client=client,
                    fixture_manager=fixture_manager,
                    token=identity["token"],
                    owner_id=identity["ownerId"],
                    tenant_id=identity["tenantId"],
                    runtime_dir=runtime_dir,
                    model_proxy=model_proxy,
                    graphrag_fixture=graphrag_fixture,
                )
                if outcome.fatal:
                    fatal = True
                if outcome.record is None:
                    failed += 1
                    continue
                store.append(outcome.record)
                store.append_scores(
                    _score_record(
                        case,
                        outcome.record,
                        staged_durations=outcome.staged_durations,
                        evaluators=evaluators,
                    )
                )
                existing.add(
                    run_key(outcome.record.case_id, outcome.record.repetition, outcome.record.attempt)
                )
                executed += 1

    # 评测结束后复验不可变数据：变了说明污染了评测库之外的数据，
    # baseline 也绝不能算成功
    if integrity.immutable_hash(run_id) != before_hash:
        logger.error("评测改动了可变范围之外的数据 run=%s", run_id)
        fatal = True

    exit_code = _exit_code(plan, store, planned, fatal)
    return RunSummary(run_id, planned, executed, skipped, failed, fatal, exit_code)


def _client(settings: EvalSettings) -> EngineClient:
    """client 的内部辅助逻辑。"""
    return EngineClient(base_url=settings.engine_base_url, timeout=settings.request_timeout_seconds)


def _is_failed(
    store: RawRunStore, existing: set[str], case_id: str, repetition: int, attempt: int
) -> bool:
    """该次尝试没有记录，或 task_outcome 明确未通过。

    没有评分行不等于失败（可能是 append 与 append_scores 之间崩溃），
    此时返回 False：交由报告侧暴露缺分，避免把已通过的用例误判为需重跑。
    """
    if run_key(case_id, repetition, attempt) not in existing:
        return True
    for row in store.load_scores():
        if (
            row.get("caseId") == case_id
            and row.get("repetition") == repetition
            and row.get("attempt") == attempt
            and row.get("label") == "task_outcome"
        ):
            return not bool(row.get("testPass"))
    return False


def _exit_code(plan: ExecutionPlan, store: RawRunStore, planned: int, fatal: bool) -> int:
    """baseline：仅不安全配置/污染/清理失败非零；gate：任何零容忍违规非零。"""
    if fatal:
        return 1
    # 一条记录都没落 = 环境整体不可用（如后端全挂），不是有效的 baseline 结果
    if planned and not store.load():
        return 1
    if plan.mode != "gate":
        return 0
    for row in store.load_scores():
        if row.get("label") in {"safety", "side_effects"} and not row.get("testPass"):
            return 1
        if row.get("label") == "efficiency" and not row.get("testPass"):
            return 1
    for record in store.load():
        if record.trajectory_incomplete:
            return 1
    return 0
