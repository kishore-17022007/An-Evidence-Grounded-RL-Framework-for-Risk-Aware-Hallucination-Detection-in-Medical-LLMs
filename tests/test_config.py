"""Tests for environment-based settings."""

import pytest

from config.settings import Settings
from src.common.exceptions import ConfigurationError


def test_settings_load_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.data_dir.as_posix() == "data"
    assert settings.db_path.as_posix() == "data/medverify.sqlite3"
    assert settings.seed == 17
    assert settings.offline_mode is True
    assert settings.log_level == "INFO"


def test_settings_parse_values() -> None:
    settings = Settings.from_env(
        {
            "MEDVERIFY_DATA_DIR": "~/medverify-data",
            "MEDVERIFY_DB_PATH": "~/medverify-data/history.sqlite3",
            "MEDVERIFY_SEED": "23",
            "MEDVERIFY_OFFLINE": "off",
            "MEDVERIFY_LOG_LEVEL": "debug",
        }
    )

    assert settings.data_dir.name == "medverify-data"
    assert settings.db_path.name == "history.sqlite3"
    assert settings.seed == 23
    assert settings.offline_mode is False
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    "environment, message",
    [
        ({"MEDVERIFY_SEED": "not-an-int"}, "MEDVERIFY_SEED must be an integer"),
        ({"MEDVERIFY_OFFLINE": "sometimes"}, "MEDVERIFY_OFFLINE must be a boolean"),
    ],
)
def test_settings_reject_invalid_values(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env(environment)


def test_settings_do_not_require_api_credentials() -> None:
    settings = Settings.from_env({})

    assert settings.offline_mode is True


def test_settings_reject_unknown_log_level() -> None:
    with pytest.raises(ConfigurationError, match="MEDVERIFY_LOG_LEVEL"):
        Settings.from_env({"MEDVERIFY_LOG_LEVEL": "VERBOSE"})


def test_verification_config_loads_limits_from_environment() -> None:
    from src.verification.config import VerificationConfig

    config = VerificationConfig.from_env(
        {
            "MEDVERIFY_EPMC_TIMEOUT": "4.5",
            "MEDVERIFY_EPMC_RETRIES": "1",
            "MEDVERIFY_MAX_RESULTS_PER_CLAIM": "3",
            "MEDVERIFY_ACTION_BUDGET": "6",
            "MEDVERIFY_MIN_COVERAGE": "0.6",
            "MEDVERIFY_AUDIT_PATH": "tmp/audit.jsonl",
        }
    )

    assert config.request_timeout_seconds == 4.5
    assert config.max_retries == 1
    assert config.max_results_per_claim == 3
    assert config.action_budget == 6
    assert config.minimum_coverage == 0.6
    assert config.audit_path.as_posix() == "tmp/audit.jsonl"