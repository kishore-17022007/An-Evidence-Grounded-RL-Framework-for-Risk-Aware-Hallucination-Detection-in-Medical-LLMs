"""Environment-based settings with offline-safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.common.exceptions import ConfigurationError


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be a boolean value (true/false, yes/no, on/off, or 1/0); "
        f"got {value!r}"
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings; no external credentials are required."""

    seed: int = 17
    offline_mode: bool = True
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    db_path: Path = Path("data/medverify.sqlite3")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load settings from environment variables or a test mapping."""
        values = os.environ if environ is None else environ
        raw_seed = values.get("MEDVERIFY_SEED", "17")
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ConfigurationError(
                f"MEDVERIFY_SEED must be an integer; got {raw_seed!r}"
            ) from error

        log_level = values.get("MEDVERIFY_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(
                "MEDVERIFY_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )

        return cls(
            seed=seed,
            offline_mode=_parse_bool(
                values.get("MEDVERIFY_OFFLINE", "true"), "MEDVERIFY_OFFLINE"
            ),
            log_level=log_level,
            data_dir=Path(values.get("MEDVERIFY_DATA_DIR", "data")).expanduser(),
            db_path=Path(
                values.get("MEDVERIFY_DB_PATH", "data/medverify.sqlite3")
            ).expanduser(),
        )