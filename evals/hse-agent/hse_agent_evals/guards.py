"""评测运行的隔离护栏。

来源：codex/qdrant-regulation-kb 分支的 runner.py 加固（提交 ff56106
"harden runner recovery and isolation"），移植进本站扁平实现。

三道护栏各自防一类真实事故：
- BackendBindingGuard：prepare 之后改了配置/环境再 execute，会连到非评测的后端；
- MySqlIntegrityGuard：评测过程写到了评测库之外，或动了可变数据范围以外的行；
- RawRunStore 的跨进程 lease 在 runner.py 内（与存储耦合，见那里）。

已知未覆盖：分支那版还校验一组后端环境变量（SPRING_DATASOURCE_URL /
AGENT_API_* / HSE_REGULATION_KB_BASE_URL 等）是否与 settings 一致。
本站的编排脚本尚未建立这套环境契约，因此先不移植该半段，
待编排脚本接管后端启动后再补 —— 否则会要求一个不存在的契约。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from hse_agent_evals.adapters.fixture_manager import DATABASE_PREFIX, FixtureManager


def _url_port(url: str) -> int:
    """url port 的内部辅助逻辑。"""
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


class BackendBindingGuard:
    """复验持久化的运行状态与当前 settings 一致。

    run 状态是 prepare 时写下的；若之后端口、库名或模型被改动，
    execute 会连到与准备阶段不同的后端，产出的记录不再可信。
    """

    def __init__(
        self,
        state: Mapping[str, Any],
        settings: Any,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """初始化实例。"""
        self.state = dict(state)
        self.settings = settings
        self.environ = dict(environ if environ is not None else {})

    def validate(self, run_id: str) -> None:
        """处理 validate 相关逻辑。"""
        database_name = self.state.get("database_name")
        jdbc_url = self.state.get("jdbc_url")
        ports = self.state.get("ports")
        if (
            self.state.get("run_id") != run_id
            or not isinstance(database_name, str)
            or not database_name.startswith(DATABASE_PREFIX)
            or not isinstance(jdbc_url, str)
            or not isinstance(ports, Mapping)
            or self.state.get("redis_database") != 15
        ):
            raise ValueError("评测后端绑定不安全")

        parsed = urlparse(jdbc_url.removeprefix("jdbc:"))
        try:
            jdbc_safe = (
                parsed.scheme == "mysql"
                and parsed.hostname is not None
                and ip_address(parsed.hostname).is_loopback
                and parsed.port == self.settings.mysql_port
                and parsed.path.lstrip("/") == database_name
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            jdbc_safe = False

        expected_ports = {
            "mysql": self.settings.mysql_port,
            "engine": _url_port(self.settings.engine_base_url),
            "model_proxy": _url_port(self.settings.model_proxy_url),
            "ollama": _url_port(self.settings.ollama_base_url),
            "graphrag_fixture": _url_port(self.settings.graphrag_fixture_url),
        }
        if (
            not jdbc_safe
            or dict(ports) != expected_ports
            or self.settings.redis_database != 15
        ):
            raise ValueError("评测后端绑定不安全")


class MySqlIntegrityGuard:
    """校验评测库选择正确，并能对不可变数据取哈希。

    EXCLUDED_TABLES 是评测过程中本身就会变的表（执行日志、迁移历史）；
    MUTABLE_RANGES 之外的业务行在评测期间必须保持不变 —— 否则说明
    评测改动了不该改的数据，或根本没连到评测库。
    """

    EXCLUDED_TABLES = {"hse_agent_execution_log", "flyway_schema_history"}
    MUTABLE_RANGES = {
        "hse_report_task": ("task_id", 980000, 989999),
        "hse_reminder": ("id", 980000, 989999),
        "hse_weekly_report": ("report_id", 980000, 989999),
        "hse_ai_alert": ("alert_id", 980000, 989999),
    }

    def __init__(self, fixture_manager: FixtureManager, database_name: str) -> None:
        """初始化实例。"""
        if not database_name.startswith(DATABASE_PREFIX):
            raise ValueError("完整性检查拒绝非评测数据库")
        self.fixture_manager = fixture_manager
        self.database_name = database_name

    def assert_safe(self, run_id: str) -> None:
        """确认当前连接落在本 run 的评测库上。

        run_id 仅用于报错定位；真正的判据是服务端回报的 DATABASE()。
        """
        connection = self.fixture_manager.connection_factory(
            database=self.database_name, charset="utf8mb4", autocommit=False
        )
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE()")
            row = cursor.fetchone()
            selected = next(iter(row.values())) if isinstance(row, Mapping) else row[0]
            if selected != self.database_name:
                raise RuntimeError(
                    f"数据库选择不匹配（run_id={run_id}）: {selected!r} != {self.database_name!r}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()

    def immutable_hash(self, run_id: str) -> str:
        """对可变范围之外的业务数据取稳定哈希，用于评测前后比对。"""
        connection = self.fixture_manager.connection_factory(
            database=self.database_name, charset="utf8mb4", autocommit=False
        )
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name",
                (self.database_name,),
            )
            tables = [
                str(next(iter(row.values())) if isinstance(row, Mapping) else row[0])
                for row in list(cursor.fetchall())
            ]
            digest = hashlib.sha256()
            for table in tables:
                if table in self.EXCLUDED_TABLES:
                    continue
                if not table.replace("_", "").isalnum():
                    raise RuntimeError("表名不安全")
                where = ""
                if table in self.MUTABLE_RANGES:
                    column, lower, upper = self.MUTABLE_RANGES[table]
                    where = f" WHERE `{column}` NOT BETWEEN {lower} AND {upper}"
                cursor.execute(f"SELECT * FROM `{table}`{where}")
                columns = [column[0] for column in (cursor.description or ())]
                rows = [
                    dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))
                    for row in cursor.fetchall()
                ]
                stable = sorted(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
                    )
                    for row in rows
                )
                digest.update(table.encode("utf-8"))
                for row in stable:
                    digest.update(row.encode("utf-8"))
            connection.commit()
            return digest.hexdigest()
        except Exception:
            connection.rollback()
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()
