from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from hse_agent_evals.models import SideEffect


DATABASE_PREFIX = "engine_agent_eval_"
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9_]+$")
SIDE_EFFECT_QUERIES = {
    "hse_report_task.insert": (
        "SELECT * FROM hse_report_task "
        "WHERE task_id BETWEEN 980000 AND 989999 ORDER BY task_id"
    ),
    "hse_reminder.insert": (
        "SELECT * FROM hse_reminder "
        "WHERE id BETWEEN 980000 AND 989999 ORDER BY id"
    ),
    "hse_weekly_report.insert": (
        "SELECT * FROM hse_weekly_report "
        "WHERE report_id BETWEEN 980000 AND 989999 ORDER BY report_id"
    ),
    "hse_ai_alert.update": (
        "SELECT * FROM hse_ai_alert "
        "WHERE alert_id BETWEEN 980000 AND 989999 ORDER BY alert_id"
    ),
}
SIDE_EFFECT_PRIMARY_KEYS = {
    "hse_report_task.insert": "task_id",
    "hse_reminder.insert": "id",
    "hse_weekly_report.insert": "report_id",
    "hse_ai_alert.update": "alert_id",
}
VOLATILE_TIMESTAMP_KEYS = {"create_time", "update_time"}


class FixtureManager:
    def __init__(self, connection_factory: Callable[..., Any], prefix: str) -> None:
        if prefix != DATABASE_PREFIX:
            raise ValueError(f"评测数据库前缀必须为 {DATABASE_PREFIX}")
        self.connection_factory = connection_factory
        self.prefix = prefix

    def new_database_name(self, run_id: str) -> str:
        normalized_run_id = re.sub(r"[^a-z0-9]+", "_", str(run_id).lower()).strip("_")
        if not normalized_run_id:
            raise ValueError("运行标识必须包含字母或数字")
        unique_suffix = uuid4().hex[:12]
        maximum_run_length = 64 - len(DATABASE_PREFIX) - len(unique_suffix) - 1
        normalized_run_id = normalized_run_id[:maximum_run_length].rstrip("_")
        database_name = f"{DATABASE_PREFIX}{normalized_run_id}_{unique_suffix}"
        self._validate_database_name(database_name)
        return database_name

    def create_database(self, database_name: str) -> None:
        self._validate_database_name(database_name)

        def execute(cursor: Any) -> None:
            self._validate_database_name(database_name)
            cursor.execute(
                f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4"
            )

        self._run_transaction(execute)

    def execute_script(
        self,
        database_name: str,
        script_path: str | Path,
    ) -> None:
        self._validate_database_name(database_name)
        script = Path(script_path).read_text(encoding="utf-8-sig")
        statements = self._split_sql_statements(script)

        def execute(cursor: Any) -> None:
            self._validate_database_name(database_name)
            for statement in statements:
                cursor.execute(statement)

        self._run_transaction(execute, database=database_name)

    def drop_database(self, database_name: str) -> None:
        self._validate_drop_target(database_name)

        def execute(cursor: Any) -> None:
            self._validate_drop_target(database_name)
            cursor.execute(f"DROP DATABASE `{database_name}`")

        self._run_transaction(execute)

    def snapshot(self, database_name: str) -> dict[str, list[dict[str, Any]]]:
        self._validate_database_name(database_name)

        def execute(cursor: Any) -> dict[str, list[dict[str, Any]]]:
            self._validate_database_name(database_name)
            result: dict[str, list[dict[str, Any]]] = {}
            for key, query in SIDE_EFFECT_QUERIES.items():
                cursor.execute(query)
                rows = self._mapping_rows(cursor, cursor.fetchall())
                result[key] = self._stable_rows(key, rows)
            return result

        return self._run_transaction(execute, database=database_name)

    def snapshot_before(
        self, database_name: str
    ) -> dict[str, list[dict[str, Any]]]:
        return self.snapshot(database_name)

    def snapshot_after(
        self, database_name: str
    ) -> dict[str, list[dict[str, Any]]]:
        return self.snapshot(database_name)

    def diff(
        self,
        before: Mapping[str, list[dict[str, Any]]],
        after: Mapping[str, list[dict[str, Any]]],
    ) -> list[SideEffect]:
        ordered_keys = list(SIDE_EFFECT_QUERIES)
        ordered_keys.extend(sorted((set(before) | set(after)) - set(SIDE_EFFECT_QUERIES)))
        effects: list[SideEffect] = []
        for key in ordered_keys:
            before_rows = self._stable_rows(key, before.get(key, []))
            after_rows = self._stable_rows(key, after.get(key, []))
            if before_rows == after_rows:
                continue
            effects.append(
                SideEffect(
                    key=key,
                    before_count=len(before_rows),
                    after_count=len(after_rows),
                    changed_rows=self._changed_rows(key, before_rows, after_rows),
                )
            )
        return effects

    def _split_sql_statements(self, script: str) -> list[str]:
        active_lines = [
            line
            for line in script.splitlines()
            if not line.lstrip().startswith(("--", "#"))
        ]
        statements: list[str] = []
        buffer: list[str] = []
        quote: str | None = None
        escaped = False
        for character in "\n".join(active_lines):
            if escaped:
                buffer.append(character)
                escaped = False
                continue
            if character == chr(92) and quote is not None:
                buffer.append(character)
                escaped = True
                continue
            if quote is not None:
                buffer.append(character)
                if character == quote:
                    quote = None
                continue
            if character in {"'", '"', "`"}:
                quote = character
                buffer.append(character)
                continue
            if character == ";":
                statement = "".join(buffer).strip()
                if statement:
                    statements.append(statement)
                buffer = []
                continue
            buffer.append(character)
        trailing_statement = "".join(buffer).strip()
        if trailing_statement:
            statements.append(trailing_statement)
        return statements

    def _validate_database_name(self, database_name: str) -> None:
        if not isinstance(database_name, str):
            raise ValueError("数据库标识符不安全")
        if not database_name.startswith(DATABASE_PREFIX):
            raise ValueError(f"数据库必须使用评测前缀 {DATABASE_PREFIX}: {database_name}")
        if (
            database_name == DATABASE_PREFIX
            or len(database_name) > 64
            or IDENTIFIER_PATTERN.fullmatch(database_name) is None
        ):
            raise ValueError(f"数据库标识符不安全: {database_name}")

    def _validate_drop_target(self, database_name: str) -> None:
        if (
            not isinstance(database_name, str)
            or not database_name.startswith(DATABASE_PREFIX)
        ):
            raise ValueError(f"拒绝删除非评测数据库: {database_name}")
        self._validate_database_name(database_name)

    def _run_transaction(
        self,
        operation: Callable[[Any], Any],
        database: str | None = None,
    ) -> Any:
        connection_kwargs: dict[str, Any] = {
            "charset": "utf8mb4",
            "autocommit": False,
        }
        if database is not None:
            connection_kwargs["database"] = database
        connection = self.connection_factory(**connection_kwargs)
        cursor = None
        try:
            cursor = connection.cursor()
            result = operation(cursor)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()

    def _mapping_rows(
        self,
        cursor: Any,
        rows: Any,
    ) -> list[dict[str, Any]]:
        rows = list(rows)
        if not rows:
            return []
        if isinstance(rows[0], Mapping):
            return [dict(row) for row in rows]
        description = cursor.description or ()
        columns = [column[0] for column in description]
        return [dict(zip(columns, row)) for row in rows]

    def _stable_rows(
        self,
        key: str,
        rows: Any,
    ) -> list[dict[str, Any]]:
        normalized_rows = [self._normalize_value(dict(row)) for row in rows]
        primary_key = SIDE_EFFECT_PRIMARY_KEYS.get(key)

        def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
            value = row.get(primary_key) if primary_key else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                primary = (0, value)
            elif value is None:
                primary = (2, "")
            else:
                primary = (1, str(value))
            return (*primary, self._canonical(row))

        return sorted(normalized_rows, key=sort_key)

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for raw_key in sorted(value, key=lambda item: str(item)):
                key = str(raw_key)
                if key.lower() in VOLATILE_TIMESTAMP_KEYS:
                    continue
                normalized[key] = self._normalize_value(value[raw_key])
            return normalized
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (date, time)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized_items = [self._normalize_value(item) for item in value]
            return sorted(normalized_items, key=self._canonical)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def _changed_rows(
        self,
        key: str,
        before_rows: list[dict[str, Any]],
        after_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        primary_key = SIDE_EFFECT_PRIMARY_KEYS.get(key)
        before_by_identity = {
            self._row_identity(row, primary_key): row for row in before_rows
        }
        after_by_identity = {
            self._row_identity(row, primary_key): row for row in after_rows
        }
        changed: list[dict[str, Any]] = []
        identities = sorted(set(before_by_identity) | set(after_by_identity))
        for identity in identities:
            before_row = before_by_identity.get(identity)
            after_row = after_by_identity.get(identity)
            if before_row == after_row:
                continue
            changed.append(after_row if after_row is not None else before_row)
        return self._stable_rows(key, changed)

    def _row_identity(
        self,
        row: dict[str, Any],
        primary_key: str | None,
    ) -> str:
        if primary_key and primary_key in row:
            return f"pk:{self._canonical(row[primary_key])}"
        return f"row:{self._canonical(row)}"

    def _canonical(self, value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
