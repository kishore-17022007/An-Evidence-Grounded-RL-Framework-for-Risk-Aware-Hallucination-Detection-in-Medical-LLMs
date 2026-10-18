"""Configuration loading for local and offline MedVerify-RL runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _parse_bool(value: str, *, variable: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be a boolean value, received {value!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings sourced from environment variables."""

    data_dir: Path = Path("data")
    db_path: Path = Path("data/medverify.sqlite3")
    seed: int = 17
    offline_mode: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Create settings from the process environment or a supplied mapping."""
        values = os.environ if environ is None else environ
        seed_value = values.get("MEDVERIFY_SEED", "17")
        try:
            seed = int(seed_value)
        except ValueError as error:
            raise ValueError("MEDVERIFY_SEED must be an integer") from error

        return cls(
            data_dir=Path(values.get("MEDVERIFY_DATA_DIR", "data")).expanduser(),
            db_path=Path(
                values.get("MEDVERIFY_DB_PATH", "data/medverify.sqlite3")
            ).expanduser(),
            seed=seed,
            offline_mode=_parse_bool(
                values.get("MEDVERIFY_OFFLINE", "true"),
                variable="MEDVERIFY_OFFLINE",
            ),
        )