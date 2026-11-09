from __future__ import annotations

from unittest.mock import Mock

import pytest

from hse_agent_evals.guards import BackendBindingGuard, MySqlIntegrityGuard
from hse_agent_evals.adapters.fixture_manager import FixtureManager

from _factories import make_settings

DATABASE = "engine_agent_eval_run_x"


def _state(**overrides):
    """state 的内部辅助逻辑。"""
    settings = make_settings()
    state = {
        "run_id": "run-001",
        "database_name": DATABASE,
        "jdbc_url": f"jdbc:mysql://127.0.0.1:{settings.mysql_port}/{DATABASE}",
        "redis_database": 15,
        "ports": {
            "mysql": settings.mysql_port,
            "engine": 18081,
            "model_proxy": 11435,
            "ollama": 11434,
            "graphrag_fixture": 18080,
        },
    }
    state.update(overrides)
    return state


def test_binding_guard_accepts_matching_state():
    """验证 binding guard accepts matching state 场景。"""
    BackendBindingGuard(_state(), make_settings()).validate("run-001")


def test_binding_guard_rejects_port_drift():
    """prepare 之后端口被改动 → execute 会连到别的后端，必须拒绝。"""
    state = _state()
    state["ports"] = {**state["ports"], "engine": 9999}
    with pytest.raises(ValueError, match="绑定不安全"):
        BackendBindingGuard(state, make_settings()).validate("run-001")


def test_binding_guard_rejects_non_eval_database():
    """验证 binding guard rejects non eval database 场景。"""
    with pytest.raises(ValueError, match="绑定不安全"):
        BackendBindingGuard(_state(database_name="ENGINE-vue"), make_settings()).validate("run-001")


def test_binding_guard_rejects_jdbc_host_mismatch():
    """库名对但 jdbc 指向别处 → 记录会写到非评测库。"""
    state = _state(jdbc_url=f"jdbc:mysql://127.0.0.1:3306/engine_agent_eval_other")
    with pytest.raises(ValueError, match="绑定不安全"):
        BackendBindingGuard(state, make_settings()).validate("run-001")


def test_binding_guard_rejects_redis_db_drift():
    """验证 binding guard rejects redis db drift 场景。"""
    with pytest.raises(ValueError, match="绑定不安全"):
        BackendBindingGuard(_state(redis_database=0), make_settings()).validate("run-001")


def test_binding_guard_rejects_run_id_mismatch():
    """验证 binding guard rejects run id mismatch 场景。"""
    with pytest.raises(ValueError, match="绑定不安全"):
        BackendBindingGuard(_state(), make_settings()).validate("other-run")


def test_integrity_guard_rejects_non_eval_database():
    """验证 integrity guard rejects non eval database 场景。"""
    manager = FixtureManager(connection_factory=Mock(), prefix="engine_agent_eval_")
    with pytest.raises(ValueError, match="非评测数据库"):
        MySqlIntegrityGuard(manager, "ENGINE-vue")


def test_integrity_guard_asserts_server_reported_database():
    """真正的判据是服务端回报的 DATABASE()，不是我们以为连的库。"""
    factory = Mock()
    cursor = factory.return_value.cursor.return_value
    cursor.fetchone.return_value = {"DATABASE()": "engine_agent_eval_wrong"}
    manager = FixtureManager(connection_factory=factory, prefix="engine_agent_eval_")

    with pytest.raises(RuntimeError, match="数据库选择不匹配"):
        MySqlIntegrityGuard(manager, DATABASE).assert_safe("run-001")
    factory.return_value.rollback.assert_called()


def test_integrity_hash_excludes_volatile_tables_and_mutable_ranges():
    """执行日志会变、可变范围会变 —— 二者都不能进哈希，否则每次比对必然不等。"""
    factory = Mock()
    connection = factory.return_value
    cursor = connection.cursor.return_value
    cursor.fetchall.return_value = []
    cursor.description = []
    manager = FixtureManager(connection_factory=factory, prefix="engine_agent_eval_")

    # 表清单：一张排除的、一张可变的
    cursor.fetchall.side_effect = [
        [{"table_name": "hse_agent_execution_log"}, {"table_name": "hse_report_task"}],
        [],
    ]

    MySqlIntegrityGuard(manager, DATABASE).immutable_hash("run-001")

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert not any("hse_agent_execution_log" in sql for sql in executed), "排除表不该被查询"
    task_sql = next(sql for sql in executed if "hse_report_task" in sql)
    assert "NOT BETWEEN 980000 AND 989999" in task_sql, "可变范围必须排除在哈希之外"
