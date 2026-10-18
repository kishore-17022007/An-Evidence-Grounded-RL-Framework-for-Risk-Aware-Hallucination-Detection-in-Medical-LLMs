"""Tests for shared verification-domain contracts."""

from datetime import UTC, datetime

import pytest

from medverify_rl.domain import (
    Claim,
    ClaimAssessment,
    EvidenceRecord,
    VerificationLabel,
    VerificationReport,
)


def make_evidence(evidence_id: str = "ev-1") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id="pmid:12345",
        source_title="Example biomedical source",
        source_url="https://example.org/article",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        snippet="A source excerpt, not a generated explanation.",
        retrieval_query="example query",
        source_type="journal_article",
    )


def test_report_preserves_provenance_and_references() -> None:
    claim = Claim("claim-1", "A factual medical statement.", 0, 28)
    evidence = make_evidence()
    assessment = ClaimAssessment(
        claim_id=claim.claim_id,
        label=VerificationLabel.INSUFFICIENT_EVIDENCE,
        confidence=0.4,
        uncertainty=0.8,
        evidence_ids=(evidence.evidence_id,),
        explanation="The retrieved excerpt does not establish this claim.",
    )

    report = VerificationReport(
        question="What does the source say?",
        original_answer="A factual medical statement.",
        claims=(claim,),
        evidence=(evidence,),
        assessments=(assessment,),
        overall_uncertainty=0.8,
        abstained=False,
    )

    assert report.evidence[0].source_id == "pmid:12345"
    assert report.assessments[0].evidence_ids == ("ev-1",)
    assert report.assessments[0].label is VerificationLabel.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_assessment_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        ClaimAssessment(
            claim_id="claim-1",
            label=VerificationLabel.UNVERIFIABLE,
            confidence=confidence,
            uncertainty=0.5,
        )


def test_abstention_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="abstention_reason"):
        VerificationReport(
            question="A question",
            original_answer="An answer",
            abstained=True,
        )


def test_report_rejects_unresolved_evidence_reference() -> None:
    claim = Claim("claim-1", "A claim.")
    assessment = ClaimAssessment(
        claim_id=claim.claim_id,
        label=VerificationLabel.SUPPORTED,
        confidence=0.9,
        uncertainty=0.1,
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        VerificationReport(
            question="A question",
            original_answer="An answer",
            claims=(claim,),
            assessments=(assessment,),
            abstained=False,
        )