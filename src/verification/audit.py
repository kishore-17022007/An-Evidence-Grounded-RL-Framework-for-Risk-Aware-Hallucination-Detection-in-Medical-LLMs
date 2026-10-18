"""Append-only JSONL evidence ledger that never records credentials or full inputs."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.verification.errors import AuditWriteError
from src.verification.models import WorkflowEvent


_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|token|password|secret)", re.IGNORECASE)
_INLINE_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|password|secret)\s*[:=]\s*[^\s,;]+"
)
_LOCK = threading.Lock()


class AuditLedger:
    """Persist query/provenance/decision events without raw question/answer text."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: WorkflowEvent) -> None:
        try:
            payload = _json_safe(asdict(event))
            scrubbed = _scrub_secrets(payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(scrubbed, ensure_ascii=True, sort_keys=True) + "\n")
        except OSError as error:
            raise AuditWriteError(f"Unable to append verification audit event: {error}") from error


def _json_safe(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _scrub_secrets(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {name: _scrub_secrets(item, key=name) for name, item in value.items()}
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]
    if isinstance(value, str):
        value = _INLINE_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        return re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
    return value