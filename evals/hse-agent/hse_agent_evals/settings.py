from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_EVAL_", extra="forbid")

    engine_base_url: str = "http://127.0.0.1:18081"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_admin_user: str = "root"
    mysql_admin_password: SecretStr
    mysql_database_prefix: str = "engine_agent_eval_"
    redis_database: int = 15
    model_proxy_url: str = "http://127.0.0.1:11435/v1"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:7b"
    graphrag_fixture_url: str = "http://127.0.0.1:18080"
    repetitions: int = Field(default=3, ge=1, le=5)
    request_timeout_seconds: float = Field(default=90.0, gt=0)
    gpu_hourly_cost_cny: float | None = Field(default=None, ge=0)
    cpu_hourly_cost_cny: float | None = Field(default=None, ge=0)

    @field_validator(
        "engine_base_url", "model_proxy_url", "ollama_base_url", "graphrag_fixture_url"
    )
    @classmethod
    def require_loopback_url(cls, value: str) -> str:
        parsed = urlparse(value)
        try:
            port = parsed.port
            is_safe_endpoint = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.netloc)
                and parsed.hostname is not None
                and ip_address(parsed.hostname).is_loopback
                and (port is None or 1 <= port <= 65535)
            )
        except ValueError:
            is_safe_endpoint = False
        if not is_safe_endpoint:
            raise ValueError("评测服务必须绑定回环地址")
        return value.rstrip("/")

    @field_validator("mysql_host")
    @classmethod
    def require_loopback_mysql(cls, value: str) -> str:
        try:
            is_loopback = ip_address(value).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("评测数据库必须位于回环地址")
        return value

    @field_validator("mysql_database_prefix")
    @classmethod
    def require_eval_prefix(cls, value: str) -> str:
        if value != "engine_agent_eval_":
            raise ValueError("评测数据库前缀必须为 engine_agent_eval_")
        return value

    @field_validator("redis_database")
    @classmethod
    def require_eval_redis_db(cls, value: int) -> int:
        if value != 15:
            raise ValueError("评测 Redis DB 必须为 15")
        return value
