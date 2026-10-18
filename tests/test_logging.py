"""Tests for structured JSON logging."""

import json
import logging

from src.common.logging_config import JsonFormatter


def test_json_formatter_includes_message_and_context() -> None:
    record = logging.LogRecord(
        name="medverify.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Action recorded",
        args=(),
        exc_info=None,
    )
    record.claim_id = "claim-1"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "Action recorded"
    assert payload["context"]["claim_id"] == "claim-1"