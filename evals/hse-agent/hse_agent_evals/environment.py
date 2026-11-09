from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import pymysql

from hse_agent_evals.adapters.fixture_manager import (
    DATABASE_PREFIX,
    FixtureManager,
)
from hse_agent_evals.settings import EvalSettings


EVAL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = EVAL_ROOT / ".runtime"
BASE_SEED_PATH = EVAL_ROOT / "fixtures" / "database" / "base_seed.sql"
RESET_MUTABLE_PATH = EVAL_ROOT / "fixtures" / "database" / "reset_mutable.sql"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def prepare(
    run_id: str,
    settings: Any,
    fixture_manager: FixtureManager,
    runtime_dir: str | Path = RUNTIME_DIR,
    created_time: datetime | None = None,
) -> dict[str, Any]:
    state_path = _runtime_path(run_id, runtime_dir)
    if state_path.exists():
        raise FileExistsError(f"运行状态已存在: {state_path}")
    database_name = fixture_manager.new_database_name(run_id)
    timestamp = created_time or datetime.now(timezone.utc)
    state = {
        "run_id": run_id,
        "database_name": database_name,
        "jdbc_url": (
            f"jdbc:mysql://{settings.mysql_host}:{settings.mysql_port}/"
            f"{database_name}?useUnicode=true&characterEncoding=UTF-8"
        ),
        "redis_database": settings.redis_database,
        "ports": {
            "mysql": settings.mysql_port,
            "engine": _url_port(settings.engine_base_url),
            "model_proxy": _url_port(settings.model_proxy_url),
            "ollama": _url_port(settings.ollama_base_url),
            "graphrag_fixture": _url_port(settings.graphrag_fixture_url),
        },
        "created_time": timestamp.isoformat(),
    }
    serialized_state = (
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with state_path.open("x", encoding="utf-8", newline="\n") as state_file:
            state_file.write(serialized_state)
    except FileExistsError as error:
        raise FileExistsError(f"运行状态已存在: {state_path}") from error
    fixture_manager.create_database(database_name)
    return state


def seed(
    run_id: str,
    fixture_manager: FixtureManager,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> None:
    state = _load_state(run_id, runtime_dir)
    fixture_manager.execute_script(
        state["database_name"],
        BASE_SEED_PATH,
    )


def reset_case(
    run_id: str,
    fixture_manager: FixtureManager,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> None:
    state = _load_state(run_id, runtime_dir)
    database_name = state["database_name"]
    fixture_manager.execute_script(database_name, RESET_MUTABLE_PATH)
    fixture_manager.execute_script(database_name, BASE_SEED_PATH)


def snapshot(
    run_id: str,
    fixture_manager: FixtureManager,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> dict[str, list[dict[str, Any]]]:
    state = _load_state(run_id, runtime_dir)
    return fixture_manager.snapshot(state["database_name"])


def run_state(
    run_id: str,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> dict[str, Any]:
    """返回本次运行的持久化状态（含库名、jdbc_url、端口映射）。

    runner 需要用它做后端绑定复验，故公开暴露；database_name 亦由此派生，
    避免两处各自读取状态导致口径分叉。
    """
    return _load_state(run_id, runtime_dir)


def database_name(
    run_id: str,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> str:
    """返回本次运行对应的评测库名（runner 做副作用快照时需要）。"""
    return str(run_state(run_id, runtime_dir)["database_name"])


def cleanup(
    run_id: str,
    fixture_manager: FixtureManager,
    runtime_dir: str | Path = RUNTIME_DIR,
) -> None:
    state_path = _runtime_path(run_id, runtime_dir)
    state = _read_state(state_path)
    fixture_manager.drop_database(state["database_name"])
    state_path.unlink()


def _runtime_path(run_id: str, runtime_dir: str | Path) -> Path:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"运行标识不安全: {run_id}")
    return Path(runtime_dir) / f"{run_id}.json"


def _load_state(
    run_id: str,
    runtime_dir: str | Path,
) -> dict[str, Any]:
    return _read_state(_runtime_path(run_id, runtime_dir))


def _read_state(state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("评测运行状态格式无效")
    database_name = state.get("database_name")
    if not isinstance(database_name, str):
        raise ValueError("评测运行状态缺少数据库名称")
    return state


def _url_port(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def fixture_manager(settings: EvalSettings) -> FixtureManager:
    """构建夹具管理器；与 runner 编排共用同一入口，避免连接参数实现分叉。"""

    def connection_factory(**kwargs: Any) -> Any:
        return pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_admin_user,
            password=settings.mysql_admin_password.get_secret_value(),
            **kwargs,
        )

    return FixtureManager(
        connection_factory=connection_factory,
        prefix=DATABASE_PREFIX,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hse-agent-environment")
    parser.add_argument(
        "command",
        choices=("prepare", "seed", "reset_case", "snapshot", "cleanup"),
    )
    parser.add_argument("run_id")
    arguments = parser.parse_args(argv)
    settings = EvalSettings()
    manager = fixture_manager(settings)
    command = arguments.command
    if command == "prepare":
        result = prepare(arguments.run_id, settings, manager)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif command == "seed":
        seed(arguments.run_id, manager)
    elif command == "reset_case":
        reset_case(arguments.run_id, manager)
    elif command == "snapshot":
        result = snapshot(arguments.run_id, manager)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        cleanup(arguments.run_id, manager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
