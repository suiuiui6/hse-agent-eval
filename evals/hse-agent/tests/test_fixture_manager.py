import importlib
import json
import os
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from hse_agent_evals.adapters.fixture_manager import SIDE_EFFECT_QUERIES, FixtureManager
from hse_agent_evals.models import SideEffect


DATABASE_PREFIX = "engine_agent_eval_"
EVAL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EVAL_ROOT.parents[1]
BASE_SEED_PATH = EVAL_ROOT / "fixtures" / "database" / "base_seed.sql"
RESET_MUTABLE_PATH = EVAL_ROOT / "fixtures" / "database" / "reset_mutable.sql"
USERS_PATH = EVAL_ROOT / "fixtures" / "identities" / "users.json"
PLAN_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-07-17-hse-agent-evaluation-harness.md"
)
MIGRATION_DIR = (
    REPOSITORY_ROOT
    / "Engine-admin"
    / "src"
    / "main"
    / "resources"
    / "db"
    / "migration"
)


def flyway_version_key(path):
    match = re.fullmatch(r"V(\d+(?:\.\d+)*)__.+\.sql", path.name)
    if match is None:
        raise AssertionError(f"无效的 Flyway 迁移文件名: {path.name}")
    return tuple(int(part) for part in match.group(1).split("."))


def flyway_migrations():
    return sorted(MIGRATION_DIR.glob("V*.sql"), key=flyway_version_key)


def query_database(connection_factory, database_name, query, parameters=()):
    connection = connection_factory(
        database=database_name,
        charset="utf8mb4",
        autocommit=False,
    )
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(query, parameters)
        rows = list(cursor.fetchall())
        connection.commit()
        return rows
    except Exception:
        connection.rollback()
        raise
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            connection.close()


def make_database_double():
    cursor = Mock()
    connection = Mock()
    connection.cursor.return_value = cursor
    factory = Mock(return_value=connection)
    return factory, connection, cursor


def load_environment():
    return importlib.import_module("hse_agent_evals.environment")


def make_environment_settings():
    return SimpleNamespace(
        mysql_host="127.0.0.1",
        mysql_port=3306,
        mysql_admin_password="runtime-secret-password",
        redis_database=15,
        engine_base_url="http://127.0.0.1:18081",
        model_proxy_url="http://127.0.0.1:11435/v1",
        ollama_base_url="http://127.0.0.1:11434/v1",
        graphrag_fixture_url="http://127.0.0.1:18080",
    )


def write_runtime_state(runtime_dir, run_id, database_name):
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / f"{run_id}.json"
    state_path.write_text(
        json.dumps({"run_id": run_id, "database_name": database_name}),
        encoding="utf-8",
    )
    return state_path


def test_database_name_has_required_prefix():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    database_name = manager.new_database_name("RUN-20260717")
    assert database_name.startswith(DATABASE_PREFIX)
    assert re.fullmatch(r"[a-z0-9_]+", database_name)


def test_database_names_have_unique_suffixes():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    assert manager.new_database_name("run-20260717") != manager.new_database_name("run-20260717")


def test_database_name_rejects_run_id_without_alphanumeric_characters():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    with pytest.raises(ValueError, match="运行标识"):
        manager.new_database_name("../")


def test_constructor_rejects_wrong_prefix():
    with pytest.raises(ValueError, match="前缀"):
        FixtureManager(connection_factory=Mock(), prefix="engine_eval_")


def test_drop_rejects_non_eval_database():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    with pytest.raises(ValueError, match="拒绝删除"):
        manager.drop_database("ENGINE-vue")


@pytest.mark.parametrize("operation", ["create_database", "drop_database"])
@pytest.mark.parametrize(
    "database_name",
    ["engine_agent_eval_run;drop", "engine_agent_eval_run-name", "engine_agent_eval_RUN",
     "engine_agent_eval_路径", "engine_agent_eval_"],
)
def test_database_operations_reject_unsafe_identifiers(operation, database_name):
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    with pytest.raises(ValueError):
        getattr(manager, operation)(database_name)


def test_create_database_uses_utf8mb4_and_commits():
    factory, connection, cursor = make_database_double()
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)
    tick = chr(96)

    manager.create_database("engine_agent_eval_run_1")

    factory.assert_called_once_with(charset="utf8mb4", autocommit=False)
    cursor.execute.assert_called_once_with(
        f"CREATE DATABASE {tick}engine_agent_eval_run_1{tick} CHARACTER SET utf8mb4"
    )
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_drop_database_revalidates_exact_prefix_and_commits():
    factory, connection, cursor = make_database_double()
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)
    tick = chr(96)

    manager.drop_database("engine_agent_eval_run_1")

    cursor.execute.assert_called_once_with(f"DROP DATABASE {tick}engine_agent_eval_run_1{tick}")
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_drop_database_rejects_mutated_prefix_before_connecting():
    factory = Mock()
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)
    manager.prefix = "engine_"

    with pytest.raises(ValueError, match="拒绝删除"):
        manager.drop_database("engine_prod")

    factory.assert_not_called()


@pytest.mark.parametrize("operation", ["create_database", "drop_database"])
def test_database_ddl_rolls_back_and_closes_on_failure(operation):
    factory, connection, cursor = make_database_double()
    cursor.execute.side_effect = RuntimeError("database unavailable")
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)

    with pytest.raises(RuntimeError, match="database unavailable"):
        getattr(manager, operation)("engine_agent_eval_run_1")

    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_execute_script_uses_selected_database_and_explicit_transaction(tmp_path):
    factory, connection, cursor = make_database_double()
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)
    script_path = tmp_path / "fixture.sql"
    script_path.write_text(
        "-- fixture statement\n"
        "SET @fixture_value = 1;\n"
        "INSERT INTO fixture_table (id) VALUES (1);\n",
        encoding="utf-8",
    )

    manager.execute_script("engine_agent_eval_run_1", script_path)

    factory.assert_called_once_with(
        database="engine_agent_eval_run_1",
        charset="utf8mb4",
        autocommit=False,
    )
    assert cursor.execute.call_args_list == [
        call("SET @fixture_value = 1"),
        call("INSERT INTO fixture_table (id) VALUES (1)"),
    ]
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_side_effect_queries_use_current_primary_keys():
    assert SIDE_EFFECT_QUERIES == {
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


def test_snapshot_is_stable_json_compatible_and_strips_timestamps():
    factory, connection, cursor = make_database_double()
    cursor.fetchall.side_effect = [
        [
            {
                "task_id": 980002,
                "amount": Decimal("2.50"),
                "create_time": datetime(2026, 7, 17, 10, 30),
                "metadata": {
                    "due_date": date(2026, 7, 19),
                    "update_time": datetime(2026, 7, 18, 9, 0),
                },
            },
            {
                "task_id": 980001,
                "amount": Decimal("1.25"),
                "create_time": datetime(2026, 7, 17, 10, 0),
                "payload": (b"ok",),
            },
        ],
        [],
        [],
        [],
    ]
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)

    result = manager.snapshot("engine_agent_eval_run_1")

    assert result["hse_report_task.insert"] == [
        {"amount": "1.25", "payload": ["ok"], "task_id": 980001},
        {
            "amount": "2.50",
            "metadata": {"due_date": "2026-07-19"},
            "task_id": 980002,
        },
    ]
    assert list(result) == list(SIDE_EFFECT_QUERIES)
    assert cursor.execute.call_args_list == [call(query) for query in SIDE_EFFECT_QUERIES.values()]
    factory.assert_called_once_with(
        database="engine_agent_eval_run_1", charset="utf8mb4", autocommit=False
    )
    connection.commit.assert_called_once_with()
    connection.rollback.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_snapshot_rolls_back_and_closes_on_query_failure():
    factory, connection, cursor = make_database_double()
    cursor.execute.side_effect = RuntimeError("snapshot unavailable")
    manager = FixtureManager(connection_factory=factory, prefix=DATABASE_PREFIX)

    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        manager.snapshot("engine_agent_eval_run_1")

    connection.commit.assert_not_called()
    connection.rollback.assert_called_once_with()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


def test_snapshot_before_and_after_delegate_to_snapshot():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    manager.snapshot = Mock(return_value={"rows": []})

    assert manager.snapshot_before("engine_agent_eval_run_1") == {"rows": []}
    assert manager.snapshot_after("engine_agent_eval_run_1") == {"rows": []}
    assert manager.snapshot.call_args_list == [
        call("engine_agent_eval_run_1"),
        call("engine_agent_eval_run_1"),
    ]


def test_diff_returns_deterministic_side_effects_for_changed_rows():
    manager = FixtureManager(connection_factory=Mock(), prefix=DATABASE_PREFIX)
    before = {
        "hse_report_task.insert": [{"task_id": 980002, "status": "todo"}],
        "hse_reminder.insert": [{"id": 980010, "is_read": "0"}],
        "hse_weekly_report.insert": [],
        "hse_ai_alert.update": [],
    }
    after = {
        "hse_report_task.insert": [
            {"task_id": 980001, "status": "todo"},
            {"task_id": 980002, "status": "todo"},
        ],
        "hse_reminder.insert": [{"id": 980010, "is_read": "1"}],
        "hse_weekly_report.insert": [],
        "hse_ai_alert.update": [],
    }

    assert manager.diff(before, after) == [
        SideEffect(
            key="hse_report_task.insert",
            before_count=1,
            after_count=2,
            changed_rows=[{"task_id": 980001, "status": "todo"}],
        ),
        SideEffect(
            key="hse_reminder.insert",
            before_count=1,
            after_count=1,
            changed_rows=[{"id": 980010, "is_read": "1"}],
        ),
    ]


def test_environment_exposes_all_lifecycle_commands():
    environment = load_environment()
    for command_name in ("prepare", "seed", "reset_case", "snapshot", "cleanup", "main"):
        assert callable(getattr(environment, command_name))


def test_prepare_creates_database_and_password_free_runtime_state(tmp_path):
    environment = load_environment()
    settings = make_environment_settings()
    fixture_manager = Mock()
    fixture_manager.new_database_name.return_value = "engine_agent_eval_run_20260717_unique"
    created_time = datetime(2026, 7, 17, 8, 30, tzinfo=timezone.utc)

    state = environment.prepare(
        run_id="run-20260717",
        settings=settings,
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
        created_time=created_time,
    )

    fixture_manager.new_database_name.assert_called_once_with("run-20260717")
    fixture_manager.create_database.assert_called_once_with(
        "engine_agent_eval_run_20260717_unique"
    )
    assert state == {
        "run_id": "run-20260717",
        "database_name": "engine_agent_eval_run_20260717_unique",
        "jdbc_url": (
            "jdbc:mysql://127.0.0.1:3306/"
            "engine_agent_eval_run_20260717_unique"
            "?useUnicode=true&characterEncoding=UTF-8"
        ),
        "redis_database": 15,
        "ports": {
            "mysql": 3306,
            "engine": 18081,
            "model_proxy": 11435,
            "ollama": 11434,
            "graphrag_fixture": 18080,
        },
        "created_time": "2026-07-17T08:30:00+00:00",
    }
    state_path = tmp_path / "run-20260717.json"
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    raw_state = state_path.read_text(encoding="utf-8")
    assert settings.mysql_admin_password not in raw_state
    assert "password" not in raw_state.lower()


def test_prepare_rejects_unsafe_runtime_identifier_before_database_creation(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()

    with pytest.raises(ValueError, match="运行标识"):
        environment.prepare(
            run_id="../outside",
            settings=make_environment_settings(),
            fixture_manager=fixture_manager,
            runtime_dir=tmp_path,
        )

    fixture_manager.new_database_name.assert_not_called()
    fixture_manager.create_database.assert_not_called()


def test_prepare_refuses_existing_runtime_state_before_planning_or_creating_database(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    state_path = write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_existing_database"
    )
    original_state = state_path.read_bytes()

    with pytest.raises(FileExistsError, match="运行状态已存在"):
        environment.prepare(
            run_id="run-20260717",
            settings=make_environment_settings(),
            fixture_manager=fixture_manager,
            runtime_dir=tmp_path,
        )

    fixture_manager.new_database_name.assert_not_called()
    fixture_manager.create_database.assert_not_called()
    assert state_path.read_bytes() == original_state


def test_prepare_does_not_create_database_when_runtime_state_cannot_be_written(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    fixture_manager.new_database_name.return_value = "engine_agent_eval_planned_database"
    blocked_runtime_dir = tmp_path / "runtime-file"
    blocked_runtime_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError):
        environment.prepare(
            run_id="run-20260717",
            settings=make_environment_settings(),
            fixture_manager=fixture_manager,
            runtime_dir=blocked_runtime_dir,
        )

    fixture_manager.create_database.assert_not_called()


def test_prepare_preserves_complete_runtime_state_when_database_creation_fails(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    database_name = "engine_agent_eval_planned_database"
    fixture_manager.new_database_name.return_value = database_name
    fixture_manager.create_database.side_effect = RuntimeError("create failed")
    created_time = datetime(2026, 7, 17, 8, 30, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="create failed"):
        environment.prepare(
            run_id="run-20260717",
            settings=make_environment_settings(),
            fixture_manager=fixture_manager,
            runtime_dir=tmp_path,
            created_time=created_time,
        )

    assert json.loads(
        (tmp_path / "run-20260717.json").read_text(encoding="utf-8")
    ) == {
        "run_id": "run-20260717",
        "database_name": database_name,
        "jdbc_url": (
            "jdbc:mysql://127.0.0.1:3306/engine_agent_eval_planned_database"
            "?useUnicode=true&characterEncoding=UTF-8"
        ),
        "redis_database": 15,
        "ports": {
            "mysql": 3306,
            "engine": 18081,
            "model_proxy": 11435,
            "ollama": 11434,
            "graphrag_fixture": 18080,
        },
        "created_time": "2026-07-17T08:30:00+00:00",
    }


def test_prepare_writes_complete_runtime_state_before_create_database_callback(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    database_name = "engine_agent_eval_planned_database"
    fixture_manager.new_database_name.return_value = database_name
    observed_states = []

    def assert_state_is_ready(callback_database_name):
        assert callback_database_name == database_name
        state = json.loads(
            (tmp_path / "run-20260717.json").read_text(encoding="utf-8")
        )
        assert set(state) == {
            "run_id",
            "database_name",
            "jdbc_url",
            "redis_database",
            "ports",
            "created_time",
        }
        assert state["run_id"] == "run-20260717"
        assert state["database_name"] == database_name
        observed_states.append(state)

    fixture_manager.create_database.side_effect = assert_state_is_ready

    state = environment.prepare(
        run_id="run-20260717",
        settings=make_environment_settings(),
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
        created_time=datetime(2026, 7, 17, 8, 30, tzinfo=timezone.utc),
    )

    assert observed_states == [state]


def test_seed_executes_base_seed_against_runtime_database(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_run_20260717_unique"
    )

    environment.seed(
        run_id="run-20260717",
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
    )

    fixture_manager.execute_script.assert_called_once_with(
        "engine_agent_eval_run_20260717_unique",
        environment.BASE_SEED_PATH,
    )


def test_reset_case_deletes_mutable_rows_then_restores_base_seed(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_run_20260717_unique"
    )

    environment.reset_case(
        run_id="run-20260717",
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
    )

    assert fixture_manager.execute_script.call_args_list == [
        call("engine_agent_eval_run_20260717_unique", environment.RESET_MUTABLE_PATH),
        call("engine_agent_eval_run_20260717_unique", environment.BASE_SEED_PATH),
    ]


def test_environment_snapshot_delegates_to_fixture_manager(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    fixture_manager.snapshot.return_value = {
        "hse_report_task.insert": [{"task_id": 980001}]
    }
    write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_run_20260717_unique"
    )

    result = environment.snapshot(
        run_id="run-20260717",
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
    )

    assert result == {"hse_report_task.insert": [{"task_id": 980001}]}
    fixture_manager.snapshot.assert_called_once_with(
        "engine_agent_eval_run_20260717_unique"
    )


def test_cleanup_drops_database_before_removing_runtime_state(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    state_path = write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_run_20260717_unique"
    )

    environment.cleanup(
        run_id="run-20260717",
        fixture_manager=fixture_manager,
        runtime_dir=tmp_path,
    )

    fixture_manager.drop_database.assert_called_once_with(
        "engine_agent_eval_run_20260717_unique"
    )
    assert not state_path.exists()


def test_cleanup_keeps_runtime_state_when_database_drop_fails(tmp_path):
    environment = load_environment()
    fixture_manager = Mock()
    fixture_manager.drop_database.side_effect = RuntimeError("drop failed")
    state_path = write_runtime_state(
        tmp_path, "run-20260717", "engine_agent_eval_run_20260717_unique"
    )

    with pytest.raises(RuntimeError, match="drop failed"):
        environment.cleanup(
            run_id="run-20260717",
            fixture_manager=fixture_manager,
            runtime_dir=tmp_path,
        )

    assert state_path.exists()


def test_cleanup_refuses_production_database_from_runtime_state(tmp_path):
    environment = load_environment()
    factory = Mock()
    fixture_manager = FixtureManager(
        connection_factory=factory,
        prefix=DATABASE_PREFIX,
    )
    state_path = write_runtime_state(tmp_path, "run-20260717", "ENGINE-vue")

    with pytest.raises(ValueError, match="拒绝删除"):
        environment.cleanup(
            run_id="run-20260717",
            fixture_manager=fixture_manager,
            runtime_dir=tmp_path,
        )

    factory.assert_not_called()
    assert state_path.exists()


def test_base_seed_contains_required_idempotent_eval_fixtures():
    seed = BASE_SEED_PATH.read_text(encoding="utf-8")
    lowered = seed.lower()

    assert re.search(r"\$2[aby]\$\d{2}\$[./a-zA-Z0-9]{53}", seed)
    for token in (
        "agent_eval_admin",
        "hse_user",
        "other_dept_user",
        "hse:ai:query",
        "engine.account.captchaEnabled",
        "sys_user",
        "sys_role",
        "sys_user_role",
        "sys_role_menu",
        "sys_dept",
        "hse_report_task",
        "hse_ai_alert",
        "hse_knowledge_index",
    ):
        assert token in seed
    assert re.search(
        r"engine\.account\.captchaEnabled.*?false",
        seed,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert "on duplicate key update" in lowered
    assert "drop database" not in lowered
    assert "truncate " not in lowered
    assert "delete from sys_" not in lowered
    assert seed.rstrip().endswith(";")


def test_base_seed_has_expected_departments_and_business_row_counts():
    seed = BASE_SEED_PATH.read_text(encoding="utf-8")

    assert re.search(r"\b210\b", seed)
    assert re.search(r"\b220\b", seed)
    for task_id in range(981000, 981005):
        assert str(task_id) in seed
    for alert_id in range(982000, 982004):
        assert str(alert_id) in seed
    for index_id in range(983000, 983003):
        assert str(index_id) in seed


def test_base_seed_bootstraps_knowledge_table_and_eval_auto_increment_ranges():
    seed = BASE_SEED_PATH.read_text(encoding="utf-8")
    lowered = seed.lower()

    create_index = lowered.index(
        "create table if not exists hse_knowledge_index"
    )
    insert_index = lowered.index("insert into hse_knowledge_index")
    assert create_index < insert_index
    for table in (
        "hse_agent_execution_log",
        "hse_reminder",
        "hse_weekly_report",
    ):
        assert re.search(
            rf"alter +table +{table} +auto_increment *= *980000 *;",
            seed,
            flags=re.IGNORECASE,
        )


def test_reset_mutable_deletes_only_eval_owned_rows():
    reset_sql = RESET_MUTABLE_PATH.read_text(encoding="utf-8")
    lowered = reset_sql.lower()
    delete_statements = re.findall(
        r"delete\s+from\s+([a-z0-9_]+)\s+where\s+([^;]+);",
        reset_sql,
        flags=re.IGNORECASE,
    )
    expected_primary_keys = {
        "hse_agent_execution_log": "execution_id",
        "hse_reminder": "id",
        "hse_report_task": "task_id",
        "hse_weekly_report": "report_id",
        "hse_ai_alert": "alert_id",
    }

    assert len(delete_statements) == len(expected_primary_keys)
    assert {table.lower() for table, _ in delete_statements} == set(expected_primary_keys)
    for table, predicate in delete_statements:
        table = table.lower()
        normalized_predicate = " ".join(predicate.lower().split())
        owned_range = f"{expected_primary_keys[table]} between 980000 and 989999"
        assert owned_range in normalized_predicate
        extra_predicate = normalized_predicate.replace(owned_range, "").strip()
        assert extra_predicate in {"", "or create_by = 'agent-eval'"}
    assert "truncate " not in lowered
    assert "drop " not in lowered


def test_identity_fixture_defines_required_login_roles():
    users = json.loads(USERS_PATH.read_text(encoding="utf-8"))

    assert {"admin", "hse_user", "other_dept_user"} <= set(users)
    assert users["system_admin"]["user_id"] == 1
    assert "*:*:*" in users["system_admin"]["permissions"]
    assert users["hse_user"]["username"] == "hse_user"
    assert users["hse_user"]["dept_id"] == 210
    assert users["other_dept_user"]["username"] == "other_dept_user"
    assert users["other_dept_user"]["dept_id"] == 220
    for role, identity in users.items():
        if role == "system_admin":
            continue
        assert 980000 <= identity["user_id"] <= 989999
        assert identity["password"]
        assert "hse:ai:query" in identity["permissions"]


def test_task5_plan_uses_current_reminder_primary_key():
    plan = PLAN_PATH.read_text(encoding="utf-8")
    task5 = plan.split("## Task 5:", maxsplit=1)[1].split(
        "## Task 6:", maxsplit=1
    )[0]

    assert (
        "SELECT * FROM hse_reminder "
        "WHERE id BETWEEN 980000 AND 989999 ORDER BY id"
    ) in task5
    assert "WHERE reminder_id BETWEEN" not in task5


def test_migration_files_follow_flyway_numeric_version_order():
    migration_names = [path.name for path in flyway_migrations()]

    assert migration_names.index(
        "V2.9__hse_parse10_professional_dept_idempotent.sql"
    ) < migration_names.index(
        "V2.10__hse_parse10_submission_matrix_rbac.sql"
    )
    assert migration_names.index(
        "V9.6__patch_box_index_fix.sql"
    ) < migration_names.index(
        "V10.0__fix_schema_idempotent.sql"
    )


@pytest.mark.integration
def test_mysql_database_lifecycle_when_local_credentials_are_available(tmp_path):
    password = os.getenv("AGENT_EVAL_MYSQL_ADMIN_PASSWORD")
    if password is None:
        pytest.skip("未提供本地 MySQL 评测环境变量")

    import pymysql

    def connection_factory(**kwargs):
        kwargs.setdefault("cursorclass", pymysql.cursors.DictCursor)
        return pymysql.connect(
            host=os.getenv("AGENT_EVAL_MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENT_EVAL_MYSQL_PORT", "3306")),
            user=os.getenv("AGENT_EVAL_MYSQL_ADMIN_USER", "root"),
            password=password,
            **kwargs,
        )

    manager = FixtureManager(connection_factory=connection_factory, prefix=DATABASE_PREFIX)
    database_name = manager.new_database_name("integration")
    database_created = False
    primary_error = None
    try:
        manager.create_database(database_name)
        database_created = True
        for migration_path in flyway_migrations():
            manager.execute_script(database_name, migration_path)
        manager.execute_script(database_name, BASE_SEED_PATH)

        marker = database_name.removeprefix(DATABASE_PREFIX)
        insert_path = tmp_path / "actual-inserts.sql"
        insert_path.write_text(
            "INSERT INTO hse_agent_execution_log "
            "(session_id, user_message, agent_response, tool_calls, status, tokens_used, duration_ms, "
            "round_count, prompt_tokens, completion_tokens, model_name, provider_name, "
            "model_duration_ms, tool_duration_ms, error_message, confirmation_state, "
            "side_effect_committed, created_at) VALUES "
            f"('integration_{marker}', '评测问题', '评测回答', '[]', 'success', 30, 50, "
            "2, 20, 10, 'eval-model', 'eval-provider', 35, 15, NULL, 'confirmed', 1, NOW());\n"
            "INSERT INTO hse_reminder "
            "(target_user_id, content, source_type, source_id, is_read, create_time) VALUES "
            f"(980001, 'integration_{marker}', 'report', 981000, '0', NOW());\n"
            "INSERT INTO hse_weekly_report "
            "(report_title, report_range, report_data, ai_summary, report_status, generated_by, "
            "generated_at, create_time) VALUES "
            f"('integration_{marker}', '2026-W29', '{{}}', '评测摘要', 'draft', 'agent', NOW(), NOW());\n",
            encoding="utf-8",
        )
        manager.execute_script(database_name, insert_path)

        execution_rows = query_database(
            connection_factory,
            database_name,
            "SELECT * FROM hse_agent_execution_log WHERE session_id = %s",
            (f"integration_{marker}",),
        )
        reminder_rows = query_database(
            connection_factory,
            database_name,
            "SELECT * FROM hse_reminder WHERE content = %s",
            (f"integration_{marker}",),
        )
        weekly_rows = query_database(
            connection_factory,
            database_name,
            "SELECT * FROM hse_weekly_report WHERE report_title = %s",
            (f"integration_{marker}",),
        )
        assert len(execution_rows) == len(reminder_rows) == len(weekly_rows) == 1
        execution_id = execution_rows[0]["execution_id"]
        reminder_id = reminder_rows[0]["id"]
        weekly_report_id = weekly_rows[0]["report_id"]
        for generated_id in (execution_id, reminder_id, weekly_report_id):
            assert 980000 <= generated_id <= 989999
        assert execution_rows[0]["round_count"] == 2
        assert execution_rows[0]["confirmation_state"] == "confirmed"
        assert execution_rows[0]["side_effect_committed"] == 1
        assert reminder_rows[0]["target_user_id"] == 980001

        current_snapshot = manager.snapshot(database_name)
        assert any(
            row["id"] == reminder_id
            for row in current_snapshot["hse_reminder.insert"]
        )
        assert any(
            row["report_id"] == weekly_report_id
            for row in current_snapshot["hse_weekly_report.insert"]
        )

        manager.execute_script(database_name, RESET_MUTABLE_PATH)
        remaining_rows = query_database(
            connection_factory,
            database_name,
            "SELECT "
            "(SELECT COUNT(*) FROM hse_agent_execution_log WHERE execution_id = %s) AS execution_count, "
            "(SELECT COUNT(*) FROM hse_reminder WHERE id = %s) AS reminder_count, "
            "(SELECT COUNT(*) FROM hse_weekly_report WHERE report_id = %s) AS weekly_count",
            (execution_id, reminder_id, weekly_report_id),
        )
        assert remaining_rows == [
            {"execution_count": 0, "reminder_count": 0, "weekly_count": 0}
        ]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if database_created:
            try:
                manager.drop_database(database_name)
            except Exception as cleanup_error:
                evidence = f"清理评测数据库 {database_name} 失败: {cleanup_error!r}"
                if primary_error is not None:
                    primary_error.add_note(evidence)
                else:
                    raise AssertionError(evidence) from cleanup_error


def test_environment_main_passes_a_fixture_manager_instance(monkeypatch, tmp_path):
    """environment CLI 必须传夹具管理器【实例】，而不是构建函数本身。

    回归：fixture_manager 由私有函数改名公开后，main() 一度直接传了函数对象，
    导致 prepare() 里 fixture_manager.new_database_name(...) 变成属性访问函数，
    运行时才炸。此处断言传下去的是具备夹具方法的对象。
    """
    environment = load_environment()
    captured = {}

    def fake_fixture_manager(settings):
        manager = Mock()
        manager.new_database_name.return_value = "engine_agent_eval_maincheck"
        captured["manager"] = manager
        return manager

    monkeypatch.setattr(environment, "fixture_manager", fake_fixture_manager)
    monkeypatch.setattr(environment, "EvalSettings", lambda: make_environment_settings())
    monkeypatch.setattr(
        environment, "prepare", lambda run_id, settings, manager: captured.update(manager=manager) or {}
    )

    environment.main(["prepare", "run-20260918"])

    manager = captured["manager"]
    assert hasattr(manager, "new_database_name")
    assert callable(manager.new_database_name)
