"""Tests for Phase 1 typed schemas and the synthetic fixture."""

import json
from pathlib import Path

import pytest

from src.common.exceptions import SchemaValidationError
from src.common.schemas import (
    AgentActionLog,
    EvidenceDocument,
    ExperimentMetrics,
    MedicalClaim,
    VerificationLabel,
    VerificationResult,
)


def test_verification_label_vocabulary_is_exact() -> None:
    assert {label.value for label in VerificationLabel} == {
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE",
        "UNVERIFIABLE",
    }


def test_evidence_document_keeps_source_and_retrieval_provenance() -> None:
    evidence = EvidenceDocument(
        evidence_id="ev-1",
        source_name="Synthetic fixture only",
        source_identifier="synthetic:fixture-001",
        source_url=None,
        text="Fictional placeholder text.",
        retrieval_method="local_fixture",
        metadata={"dataset_type": "synthetic_development_only"},
    )

    assert evidence.source_name == "Synthetic fixture only"
    assert evidence.source_identifier == "synthetic:fixture-001"
    assert evidence.retrieval_method == "local_fixture"


def test_verification_result_rejects_unknown_label() -> None:
    with pytest.raises(SchemaValidationError, match="VerificationLabel"):
        VerificationResult(
            claim_id="claim-1",
            label="LIKELY_TRUE",  # type: ignore[arg-type]
            rationale="Schema must reject labels outside the fixed vocabulary.",
        )


def test_schemas_accept_claim_agent_log_and_empty_metrics() -> None:
    claim = MedicalClaim("claim-1", "A fictional test claim.")
    action = AgentActionLog("supervisor", "inspect", "Schema-only test record")
    metrics = ExperimentMetrics("phase1", "not-run", episode_count=0)

    assert claim.claim_id == "claim-1"
    assert action.agent_name == "supervisor"
    assert metrics.metrics == {}


def test_claim_rejects_invalid_span() -> None:
    with pytest.raises(SchemaValidationError, match="positive span"):
        MedicalClaim("claim-1", "A claim.", start_offset=5, end_offset=2)


def test_sample_dataset_loads_and_is_marked_synthetic() -> None:
    sample_path = Path(__file__).parents[1] / "data" / "sample_cases.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    assert payload["dataset_type"] == "synthetic_development_only"
    assert payload["medical_benchmark"] is False
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["source_url"] is None