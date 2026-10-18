"""Typed request, evidence, assessment, and report models for Phase 5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping

from src.common.exceptions import SchemaValidationError


class EvidenceStance(StrEnum):
    """Conservative stance of a specific excerpt relative to a claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    MIXED = "MIXED"
    UNCLEAR = "UNCLEAR"


class VerificationStatus(StrEnum):
    """User-facing claim status; intentionally separate from simulator labels."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIABLE = "UNVERIFIABLE"


class RetrievalStatus(StrEnum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    ERROR = "ERROR"
    SKIPPED_BUDGET = "SKIPPED_BUDGET"


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    """User-provided question, candidate answer, and optional non-policy context."""

    question: str
    candidate_answer: str
    user_context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise SchemaValidationError("question must be a non-empty string")
        if not isinstance(self.candidate_answer, str) or not self.candidate_answer.strip():
            raise SchemaValidationError("candidate_answer must be a non-empty string")
        if self.user_context is not None and not isinstance(self.user_context, str):
            raise SchemaValidationError("user_context must be a string when provided")


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """A sentence-level factual claim mapped exactly to its source answer span."""

    claim_id: str
    text: str
    source_start: int
    source_end: int
    extraction_method: str = "deterministic_sentence_segmenter"
    checkable: bool = True
    uncheckable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("claim_id", "text"):
            if not getattr(self, name).strip():
                raise SchemaValidationError(f"{name} must not be empty")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise SchemaValidationError("claim source offsets must define a positive span")


@dataclass(frozen=True, slots=True)
class EuropePMCRecord:
    """Publication metadata and abstract returned by the Europe PMC REST API."""

    api_source: str
    api_record_id: str
    title: str
    authors: tuple[str, ...]
    publication_year: int | None
    pmid: str | None
    pmcid: str | None
    doi: str | None
    abstract: str
    journal: str | None
    is_open_access: bool | None
    license: str | None
    source_url: str | None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.api_source.strip() or not self.api_record_id.strip():
            raise SchemaValidationError("Europe PMC source and API record ID are required")
        if not self.title.strip():
            raise SchemaValidationError("publication title must not be empty")
        if self.publication_year is not None and not 1000 <= self.publication_year <= 3000:
            raise SchemaValidationError("publication_year is outside the supported range")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise SchemaValidationError("retrieved_at must include a timezone")

    @property
    def primary_identifier(self) -> str | None:
        if self.pmid:
            return f"PMID:{self.pmid}"
        if self.pmcid:
            return self.pmcid if self.pmcid.startswith("PMC") else f"PMC{self.pmcid}"
        if self.doi:
            return f"DOI:{self.doi}"
        return None

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Return every source-supplied stable identifier, without creating IDs."""
        values: list[str] = []
        if self.pmid:
            values.append(f"PMID:{self.pmid}")
        if self.pmcid:
            values.append(self.pmcid if self.pmcid.startswith("PMC") else f"PMC{self.pmcid}")
        if self.doi:
            values.append(f"DOI:{self.doi}")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class EvidenceExcerpt:
    """A verbatim substring of an API-returned abstract and its provenance."""

    evidence_id: str
    excerpt: str
    source_start: int
    source_end: int
    publication: EuropePMCRecord
    lexical_relevance: float

    def __post_init__(self) -> None:
        if not self.excerpt.strip():
            raise SchemaValidationError("evidence excerpt must not be empty")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise SchemaValidationError("evidence offsets must define a positive span")
        if self.publication.abstract[self.source_start : self.source_end] != self.excerpt:
            raise SchemaValidationError("evidence excerpt must exactly match source abstract offsets")
        if not math.isfinite(self.lexical_relevance) or not 0 <= self.lexical_relevance <= 1:
            raise SchemaValidationError("lexical_relevance must be in [0, 1]")

    @property
    def citation_available(self) -> bool:
        return self.publication.primary_identifier is not None


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Evidence-linked claim decision with transparent baseline uncertainty."""

    claim: ExtractedClaim
    evidence: tuple[EvidenceExcerpt, ...]
    stance: EvidenceStance
    status: VerificationStatus
    confidence: float
    uncertainty: float
    confidence_method: str
    evidence_coverage: float
    missing_information: tuple[str, ...]
    rationale: str
    limitation: str | None
    source_quality_assessment: str = "Europe PMC indexing metadata only; study design and risk of bias were not assessed."
    assessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1 or not 0 <= self.uncertainty <= 1:
            raise SchemaValidationError("confidence and uncertainty must be in [0, 1]")
        if not 0 <= self.evidence_coverage <= 1:
            raise SchemaValidationError("evidence_coverage must be in [0, 1]")
        if self.assessed_at.tzinfo is None or self.assessed_at.utcoffset() is None:
            raise SchemaValidationError("assessed_at must include a timezone")


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """One auditable workflow, retrieval, policy, assessment, or stop event."""

    run_id: str
    timestamp: datetime
    event_type: str
    claim_id: str | None
    query_sent: str | None = None
    source_api: str | None = None
    retrieval_status: RetrievalStatus | None = None
    publication_identifiers: tuple[str, ...] = ()
    evidence_id: str | None = None
    evidence_excerpt: str | None = None
    policy_name: str | None = None
    observation: tuple[float, ...] | None = None
    selected_action: str | None = None
    tool_outcome: str | None = None
    assessment_status: VerificationStatus | None = None
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Final user report; raw question/answer are returned but not written to audit."""

    run_id: str
    created_at: datetime
    claims: tuple[ClaimAssessment, ...]
    events: tuple[WorkflowEvent, ...]
    action_budget: int
    actions_used: int
    stopped_reason: str
    workflow_method: str
    limitations: tuple[str, ...]
    original_question: str
    original_answer: str
    user_context: str | None = None

    @property
    def overall_status(self) -> VerificationStatus:
        if not self.claims:
            return VerificationStatus.UNVERIFIABLE
        statuses = {assessment.status for assessment in self.claims}
        if len(statuses) != 1:
            return VerificationStatus.INSUFFICIENT_EVIDENCE
        return next(iter(statuses))

    def to_dict(self) -> dict[str, Any]:
        """Serialize enums and dataclasses into JSON-compatible report data."""
        from dataclasses import asdict

        def convert(value: Any) -> Any:
            if isinstance(value, StrEnum):
                return value.value
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        payload = convert(asdict(self))
        payload["overall_status"] = self.overall_status.value
        for claim in payload["claims"]:
            for evidence in claim["evidence"]:
                publication = evidence["publication"]
                identifier = publication.get("pmid") or publication.get("pmcid") or publication.get("doi")
                evidence["citation_available"] = identifier is not None
                evidence["citation_identifier"] = identifier
                abstract = publication.pop("abstract", "")
                publication["abstract_available"] = bool(abstract)
        return payload