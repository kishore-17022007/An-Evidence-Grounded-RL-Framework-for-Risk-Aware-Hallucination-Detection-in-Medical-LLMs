"""Typed records shared by the verification workflow components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class VerificationLabel(StrEnum):
    """Allowed outcomes for a medical claim verification assessment."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIABLE = "UNVERIFIABLE"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_probability(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


@dataclass(frozen=True, slots=True)
class Claim:
    """An atomic factual statement extracted from an answer."""

    claim_id: str
    text: str
    answer_start: int | None = None
    answer_end: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_text(self.text, "claim text")
        if (self.answer_start is None) != (self.answer_end is None):
            raise ValueError("answer_start and answer_end must be set together")
        if self.answer_start is not None and self.answer_end <= self.answer_start:
            raise ValueError("answer_end must be greater than answer_start")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Retrieved evidence with source provenance kept distinct from explanations."""

    evidence_id: str
    source_id: str
    source_title: str
    source_url: str
    retrieved_at: datetime
    snippet: str
    retrieval_query: str
    source_type: str = "unknown"

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "source_id",
            "source_title",
            "source_url",
            "snippet",
            "retrieval_query",
            "source_type",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_aware_datetime(self.retrieved_at, "retrieved_at")


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """A label and uncertainty estimate, with IDs for supporting evidence."""

    claim_id: str
    label: VerificationLabel
    confidence: float
    uncertainty: float
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        _require_text(self.claim_id, "claim_id")
        _require_probability(self.confidence, "confidence")
        _require_probability(self.uncertainty, "uncertainty")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """Auditable record of an agent's selected action and stated rationale."""

    agent_name: str
    action: str
    reason: str
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    claim_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("agent_name", "action", "reason"):
            _require_text(getattr(self, field_name), field_name)
        if self.claim_id is not None:
            _require_text(self.claim_id, "claim_id")
        _require_aware_datetime(self.decided_at, "decided_at")


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Complete or partial verification result with auditable references."""

    question: str
    original_answer: str
    claims: tuple[Claim, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    assessments: tuple[ClaimAssessment, ...] = ()
    decisions: tuple[AgentDecision, ...] = ()
    overall_uncertainty: float = 1.0
    abstained: bool = True
    abstention_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.question, "question")
        _require_text(self.original_answer, "original_answer")
        _require_probability(self.overall_uncertainty, "overall_uncertainty")
        _require_aware_datetime(self.created_at, "created_at")
        if self.abstained and not (self.abstention_reason or "").strip():
            raise ValueError("abstained reports must include an abstention_reason")

        claim_ids = [claim.claim_id for claim in self.claims]
        evidence_ids = [record.evidence_id for record in self.evidence]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim IDs must be unique within a report")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique within a report")

        claim_id_set = set(claim_ids)
        evidence_id_set = set(evidence_ids)
        for assessment in self.assessments:
            if assessment.claim_id not in claim_id_set:
                raise ValueError(
                    f"assessment references unknown claim {assessment.claim_id!r}"
                )
            unknown_evidence = set(assessment.evidence_ids) - evidence_id_set
            if unknown_evidence:
                raise ValueError(
                    f"assessment references unknown evidence: {sorted(unknown_evidence)}"
                )
        for decision in self.decisions:
            if decision.claim_id is not None and decision.claim_id not in claim_id_set:
                raise ValueError(
                    f"decision references unknown claim {decision.claim_id!r}"
                )