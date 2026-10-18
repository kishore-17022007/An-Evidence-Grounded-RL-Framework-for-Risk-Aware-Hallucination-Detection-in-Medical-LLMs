"""Typed domain records and validation for the verification workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.common.exceptions import SchemaValidationError


class VerificationLabel(StrEnum):
    """The complete set of verification outcomes supported by the project."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIABLE = "UNVERIFIABLE"


MetadataValue = str | int | float | bool | None


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a non-empty string")


def _require_probability(value: float | None, field_name: str) -> None:
    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
        raise SchemaValidationError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MedicalClaim:
    """One atomic statement intended for future evidence-based assessment."""

    claim_id: str
    text: str
    start_offset: int | None = None
    end_offset: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_text(self.text, "text")
        if (self.start_offset is None) != (self.end_offset is None):
            raise SchemaValidationError(
                "start_offset and end_offset must either both be set or both be omitted"
            )
        if self.start_offset is not None and (
            self.start_offset < 0 or self.end_offset <= self.start_offset
        ):
            raise SchemaValidationError("claim offsets must define a positive span")


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """Retrieved text with source identity and retrieval provenance."""

    evidence_id: str
    source_name: str
    text: str
    retrieval_method: str
    source_identifier: str | None = None
    source_url: str | None = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source_name", "text", "retrieval_method"):
            _require_text(getattr(self, name), name)
        if self.source_identifier is not None:
            _require_text(self.source_identifier, "source_identifier")
        if self.source_url is not None:
            _require_text(self.source_url, "source_url")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise SchemaValidationError("retrieved_at must include a timezone")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Claim-level outcome with explicit evidence links and uncertainty fields."""

    claim_id: str
    label: VerificationLabel
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    confidence: float | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        if not isinstance(self.label, VerificationLabel):
            raise SchemaValidationError("label must be a VerificationLabel value")
        _require_text(self.rationale, "rationale")
        _require_probability(self.confidence, "confidence")
        _require_probability(self.uncertainty, "uncertainty")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise SchemaValidationError("evidence_ids must not contain duplicates")


@dataclass(frozen=True, slots=True)
class AgentActionLog:
    """Auditable record of an agent decision; not evidence of policy behavior."""

    agent_name: str
    action: str
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    claim_id: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("agent_name", "action", "rationale"):
            _require_text(getattr(self, name), name)
        if self.claim_id is not None:
            _require_text(self.claim_id, "claim_id")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise SchemaValidationError("timestamp must include a timezone")


@dataclass(frozen=True, slots=True)
class ExperimentMetrics:
    """Container for measured experiment aggregates; empty metrics are valid."""

    experiment_id: str
    algorithm: str
    episode_count: int = 0
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.experiment_id, "experiment_id")
        _require_text(self.algorithm, "algorithm")
        if self.episode_count < 0:
            raise SchemaValidationError("episode_count must not be negative")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise SchemaValidationError("recorded_at must include a timezone")
        for name, value in self.metrics.items():
            _require_text(name, "metric name")
            if not math.isfinite(value):
                raise SchemaValidationError(f"metric {name!r} must be finite")