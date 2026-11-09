from pydantic import ValidationError
import pytest

from hse_agent_evals.settings import EvalSettings


def test_defaults_are_isolated_and_repeat_three_times():
    settings = EvalSettings(mysql_admin_password="eval-secret")
    assert settings.engine_base_url == "http://127.0.0.1:18081"
    assert settings.mysql_database_prefix == "engine_agent_eval_"
    assert settings.redis_database == 15
    assert settings.model == "qwen2.5:7b"
    assert settings.repetitions == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engine_base_url", "https://hse.example.com"),
        ("mysql_host", "10.0.0.8"),
        ("mysql_database_prefix", "ENGINE-vue"),
        ("redis_database", 0),
    ],
)
def test_rejects_non_isolated_configuration(field, value):
    values = {"mysql_admin_password": "eval-secret", field: value}
    with pytest.raises(ValidationError):
        EvalSettings(**values)


def test_loopback_rejection_has_auditable_message():
    with pytest.raises(ValidationError) as exc_info:
        EvalSettings(
            mysql_admin_password="eval-secret",
            engine_base_url="https://hse.example.com",
        )

    assert "\u8bc4\u6d4b" in str(exc_info.value)


def test_rejects_extra_configuration_fields():
    with pytest.raises(ValidationError) as exc_info:
        EvalSettings(mysql_admin_password="eval-secret", unsafe_override=True)

    assert "extra_forbidden" in str(exc_info.value)


def test_mysql_password_is_secret():
    settings = EvalSettings(mysql_admin_password="eval-secret")

    assert str(settings.mysql_admin_password) == "**********"
    assert settings.mysql_admin_password.get_secret_value() == "eval-secret"


@pytest.mark.parametrize(
    "engine_base_url",
    [
        "//127.0.0.1:18081",
        "ftp://127.0.0.1:18081",
        "http://127.0.0.1:99999",
    ],
)
def test_rejects_url_without_safe_http_endpoint(engine_base_url):
    with pytest.raises(ValidationError):
        EvalSettings(
            mysql_admin_password="eval-secret",
            engine_base_url=engine_base_url,
        )
