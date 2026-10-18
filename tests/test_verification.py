"""Offline tests for claim extraction, Europe PMC, assessment, audit, and policy safety."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests

from src.common.exceptions import SchemaValidationError
from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.environment import MedicalVerificationEnv
from src.verification.assessment import assess_claim, rank_evidence
from src.verification.audit import AuditLedger
from src.verification.claims import extract_claims
from src.verification.config import VerificationConfig
from src.verification.errors import EvidenceRetrievalError, PolicyCompatibilityError
from src.verification.europe_pmc import (
    EuropePMCClient,
    build_search_query,
    deduplicate_publications,
    parse_search_response,
)
from src.verification.models import (
    ClaimAssessment,
    EvidenceExcerpt,
    EvidenceStance,
    RetrievalStatus,
    VerificationRequest,
    VerificationReport,
    VerificationStatus,
    WorkflowEvent,
)
from src.verification.policy_bridge import SimulatorPolicyBridge
from src.verification.workflow import VerificationWorkflow


def sample_payload() -> dict[str, Any]:
    return {
        "hitCount": 1,
        "resultList": {
            "result": [
                {
                    "source": "MED",
                    "id": "12345",
                    "pmid": "12345",
                    "pmcid": "PMC987654",
                    "doi": "10.1000/example",
                    "title": "Example compound and outcome study",
                    "authorString": "Smith J; Doe A",
                    "pubYear": "2023",
                    "abstractText": "Compound X reduced outcome Y in adults. No serious adverse effects were observed.",
                    "journalTitle": "Example Journal",
                    "isOpenAccess": "N",
                    "license": None,
                }
            ]
        },
    }


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any | None = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.payload = sample_payload() if payload is None else payload
        self.headers = headers or {}

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append((url, params, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_claim_extraction_preserves_exact_original_offsets() -> None:
    answer = "The trial enrolled adults. Compound X reduced outcome Y in adults. It may cause nausea."
    claims = extract_claims(answer, run_id="test-run")

    assert len(claims) == 3
    for claim in claims:
        assert answer[claim.source_start : claim.source_end] == claim.text
    assert claims[1].text == "Compound X reduced outcome Y in adults."
    assert claims[1].checkable


def test_short_or_opinion_sentence_is_unverifiable() -> None:
    claims = extract_claims("Possibly. I think treatment X works well.", run_id="r")

    assert len(claims) == 2
    assert not claims[0].checkable
    assert not claims[1].checkable


def test_search_query_is_deterministic_and_nonempty() -> None:
    query = build_search_query(
        "Compound X reduced outcome Y in adults.",
        "Does compound X affect outcome Y?",
    )

    assert query.startswith("TITLE_ABS:(")
    assert "\"Compound\"" in query
    assert "\"reduced\"" in query


def test_europe_pmc_parser_retains_real_fields_and_does_not_invent_ids() -> None:
    records = parse_search_response(sample_payload())
    record = records[0]

    assert record.pmid == "12345"
    assert record.pmcid == "PMC987654"
    assert record.doi == "10.1000/example"
    assert record.authors == ("Smith J", "Doe A")
    assert record.publication_year == 2023
    assert record.is_open_access is False
    assert record.source_url == "https://europepmc.org/article/MED/12345"
    assert record.primary_identifier == "PMID:12345"
    assert parse_search_response({"resultList": {"result": [{"title": "no identifiers"}]}}) == ()


def test_parser_rejects_malformed_response() -> None:
    with pytest.raises(EvidenceRetrievalError, match="omitted resultList"):
        parse_search_response({"unexpected": []})


def test_publication_deduplication_uses_reliable_identifiers() -> None:
    records = parse_search_response(sample_payload())
    duplicate = replace(records[0], api_source="PMC", api_record_id="PMC987654")
    no_ids = replace(records[0], pmid=None, pmcid=None, doi=None, api_record_id="no-id-1")
    no_ids_duplicate = replace(no_ids, api_record_id="no-id-2")

    deduplicated = deduplicate_publications([records[0], duplicate, no_ids, no_ids_duplicate])

    assert len(deduplicated) == 3


def test_client_retries_429_and_caches_success() -> None:
    session = FakeSession(
        [FakeResponse(429, headers={"Retry-After": "0"}), FakeResponse()]
    )
    sleeps: list[float] = []
    config = VerificationConfig(max_retries=1, min_request_interval_seconds=0)
    client = EuropePMCClient(config, session=session, sleep=sleeps.append)

    first = client.search("TITLE_ABS:(compound)")
    second = client.search("TITLE_ABS:(compound)")

    assert len(first) == 1 and second == first
    assert second[0].abstract == first[0].abstract
    assert len(session.calls) == 2
    assert session.calls[1][1]["resultType"] == "core"
    assert sleeps == [0.0]


def test_client_retries_timeout_then_succeeds() -> None:
    session = FakeSession([requests.Timeout("timeout"), FakeResponse()])
    sleeps: list[float] = []
    client = EuropePMCClient(
        VerificationConfig(max_retries=1, retry_backoff_seconds=0.01, min_request_interval_seconds=0),
        session=session,
        sleep=sleeps.append,
    )

    assert len(client.search("TITLE_ABS:(compound)")) == 1
    assert len(session.calls) == 2
    assert sleeps == [0.01]


def test_client_raises_meaningful_error_after_retry_exhaustion() -> None:
    session = FakeSession([requests.Timeout("secret query must not be logged")])
    client = EuropePMCClient(
        VerificationConfig(max_retries=0, min_request_interval_seconds=0),
        session=session,
        sleep=lambda _: None,
    )

    with pytest.raises(EvidenceRetrievalError, match="1 attempt"):
        client.search("TITLE_ABS:(compound)")


def test_ranked_evidence_excerpt_is_exact_source_substring() -> None:
    claim = extract_claims("Compound X reduced outcome Y in adults.", run_id="r")[0]
    record = parse_search_response(sample_payload())[0]

    evidence = rank_evidence(claim, [record], limit=3)

    assert evidence
    for item in evidence:
        assert record.abstract[item.source_start : item.source_end] == item.excerpt


def test_evidence_excerpt_rejects_fabricated_text() -> None:
    record = parse_search_response(sample_payload())[0]
    with pytest.raises(SchemaValidationError, match="exactly match source"):
        EvidenceExcerpt("e", "Fabricated sentence.", 0, 5, record, 0.5)


def test_assessment_conservative_support_and_unclear_mapping() -> None:
    config = VerificationConfig(minimum_coverage=0.5)
    record = parse_search_response(sample_payload())[0]
    supported_claim = extract_claims("Compound X reduced outcome Y in adults.", run_id="r")[0]
    supporting = rank_evidence(supported_claim, [record], limit=4)
    supported = assess_claim(
        supported_claim,
        supporting,
        retrieval_status=RetrievalStatus.SUCCESS,
        retrieval_error=None,
        config=config,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    unclear_claim = extract_claims("Compound Z dramatically cures condition W in everyone.", run_id="r")[0]
    unclear_evidence = rank_evidence(unclear_claim, [record], limit=4)
    unclear = assess_claim(
        unclear_claim,
        unclear_evidence,
        retrieval_status=RetrievalStatus.SUCCESS,
        retrieval_error=None,
        config=config,
    )

    assert supported.stance is EvidenceStance.SUPPORTS
    assert supported.status is VerificationStatus.SUPPORTED
    assert "uncalibrated" in supported.confidence_method
    assert unclear.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert unclear.stance is EvidenceStance.UNCLEAR


def test_empty_search_and_api_error_are_not_contradiction() -> None:
    claim = extract_claims("A sufficiently long factual statement about a condition.", run_id="r")[0]
    config = VerificationConfig()
    empty = assess_claim(
        claim, (), retrieval_status=RetrievalStatus.EMPTY, retrieval_error=None, config=config
    )
    failure = assess_claim(
        claim, (), retrieval_status=RetrievalStatus.ERROR, retrieval_error="Timeout", config=config
    )

    assert empty.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert failure.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert empty.stance is not EvidenceStance.CONTRADICTS
    assert "does not establish" in empty.rationale


def test_returned_but_unmatched_publications_are_not_labeled_empty_search() -> None:
    claim = extract_claims("Quantum widget cures a fictional syndrome in adults.", run_id="r")[0]
    record = parse_search_response(sample_payload())[0]
    assessment = assess_claim(
        claim,
        (),
        retrieval_status=RetrievalStatus.NO_RELEVANT_EVIDENCE,
        retrieval_error=None,
        config=VerificationConfig(),
    )
    assert assessment.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert "records, but no sufficiently relevant" in assessment.rationale
    assert record.pmid is not None


def test_mixed_evidence_maps_to_insufficient_and_confidence_is_limited() -> None:
    claim = extract_claims("Compound X reduced outcome Y in adults.", run_id="r")[0]
    supporting_record = parse_search_response(sample_payload())[0]
    conflicting_abstract = "Compound X did not reduce outcome Y in adults."
    conflicting_record = replace(
        supporting_record,
        api_record_id="conflict-1",
        pmid="999999",
        abstract=conflicting_abstract,
    )
    support_excerpt = "Compound X reduced outcome Y in adults."
    conflict_excerpt = conflicting_abstract
    evidence = (
        EvidenceExcerpt(
            "support",
            support_excerpt,
            supporting_record.abstract.index(support_excerpt),
            supporting_record.abstract.index(support_excerpt) + len(support_excerpt),
            supporting_record,
            1.0,
        ),
        EvidenceExcerpt(
            "conflict",
            conflict_excerpt,
            0,
            len(conflict_excerpt),
            conflicting_record,
            1.0,
        ),
    )

    assessment = assess_claim(
        claim,
        evidence,
        retrieval_status=RetrievalStatus.SUCCESS,
        retrieval_error=None,
        config=VerificationConfig(minimum_coverage=0.5),
    )

    assert assessment.stance is EvidenceStance.MIXED
    assert assessment.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert assessment.confidence <= 0.35


def test_unclear_stance_confidence_is_not_reported_as_high() -> None:
    claim = extract_claims("Compound Z dramatically cures condition W in everyone.", run_id="r")[0]
    record = parse_search_response(sample_payload())[0]
    assessment = assess_claim(
        claim,
        rank_evidence(claim, [record], limit=3),
        retrieval_status=RetrievalStatus.SUCCESS,
        retrieval_error=None,
        config=VerificationConfig(minimum_coverage=0.01),
    )

    assert assessment.stance is EvidenceStance.UNCLEAR
    assert assessment.confidence <= 0.4
    assert "uncalibrated" in assessment.confidence_method


def test_identifierless_publication_has_no_fabricated_source_url() -> None:
    record = parse_search_response(
        {
            "resultList": {
                "result": [
                    {
                        "source": "OTHER",
                        "id": "opaque-id",
                        "title": "Record without public identifier",
                        "abstractText": "A sufficiently long source abstract.",
                    }
                ]
            }
        }
    )[0]

    assert record.primary_identifier is None
    assert record.source_url is None


def test_identifierless_evidence_cannot_produce_supported_status() -> None:
    record = parse_search_response(
        {
            "resultList": {
                "result": [
                    {
                        "source": "OTHER",
                        "id": "opaque-id",
                        "title": "Record without public identifier",
                        "pubYear": "2025",
                        "abstractText": "Compound X reduced outcome Y in adults.",
                    }
                ]
            }
        }
    )[0]
    claim = extract_claims("Compound X reduced outcome Y in adults.", run_id="r")[0]
    assessment = assess_claim(
        claim,
        rank_evidence(claim, [record], limit=2),
        retrieval_status=RetrievalStatus.SUCCESS,
        retrieval_error=None,
        config=VerificationConfig(minimum_coverage=0.2),
    )

    assert assessment.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert any("citation cannot be verified" in item for item in assessment.missing_information)


def test_mixed_overall_report_abstains_to_insufficient_evidence() -> None:
    first_claim = extract_claims("Compound X improves outcome Y in adults.", run_id="r1")[0]
    second_claim = extract_claims("Compound Z changes outcome W in adults.", run_id="r2")[0]

    def result(claim, status):
        stance = EvidenceStance.SUPPORTS if status is VerificationStatus.SUPPORTED else EvidenceStance.CONTRADICTS
        return ClaimAssessment(
            claim=claim,
            evidence=(),
            stance=stance,
            status=status,
            confidence=0.4,
            uncertainty=0.6,
            confidence_method="uncalibrated test value",
            evidence_coverage=0.0,
            missing_information=(),
            rationale="test",
            limitation=None,
        )

    report = VerificationReport(
        run_id="run",
        created_at=datetime.now(UTC),
        claims=(
            result(first_claim, VerificationStatus.SUPPORTED),
            result(second_claim, VerificationStatus.CONTRADICTED),
        ),
        events=(),
        action_budget=4,
        actions_used=2,
        stopped_reason="claims_processed",
        workflow_method="test",
        limitations=(),
        original_question="Question?",
        original_answer="Two claims.",
    )

    assert report.overall_status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_audit_ledger_records_events_and_scrubs_secret_fields(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    ledger = AuditLedger(path)
    ledger.append(
        WorkflowEvent(
            run_id="run-1",
            timestamp=datetime.now(UTC),
            event_type="search",
            claim_id="claim-1",
            query_sent="TITLE_ABS:(compound)",
            tool_outcome="Bearer abc.secret.token",
            rationale="api_key=never-write-this",
        )
    )

    row = json.loads(path.read_text(encoding="utf-8"))

    assert row["run_id"] == "run-1"
    assert row["query_sent"] == "TITLE_ABS:(compound)"
    assert "never-write-this" not in path.read_text(encoding="utf-8")
    assert "abc.secret.token" not in path.read_text(encoding="utf-8")


def test_policy_bridge_refuses_live_state_and_accepts_only_15_floats() -> None:
    bridge = SimulatorPolicyBridge()
    with pytest.raises(PolicyCompatibilityError, match="15-feature simulator"):
        bridge.live_workflow_decision()
    with pytest.raises(PolicyCompatibilityError, match=r"shape \(15,\)"):
        bridge.live_workflow_decision(observation=np.zeros(16, dtype=np.float32))

    env = MedicalVerificationEnv()
    observation, _ = env.reset(seed=9)
    agent = DQNAgent.from_env(env, DQNConfig(device="cpu"), seed=9)
    policy_bridge = SimulatorPolicyBridge(dqn=agent)
    decision = policy_bridge.live_workflow_decision(observation=observation)

    assert len(decision.observation) == 15
    assert decision.policy_name == "DQN simulator policy"


def test_offline_workflow_is_deterministic_and_audited(tmp_path: Path) -> None:
    payload = sample_payload()
    session = FakeSession([FakeResponse(payload=payload)])
    config = VerificationConfig(
        max_results_per_claim=2,
        action_budget=4,
        min_request_interval_seconds=0,
        audit_path=tmp_path / "audit.jsonl",
    )
    client = EuropePMCClient(config, session=session, sleep=lambda _: None)
    workflow = VerificationWorkflow(config, literature_client=client)
    request = VerificationRequest(
        "Does compound X reduce outcome Y?",
        "Compound X reduced outcome Y in adults.",
        user_context="Context must not appear in the audit.",
    )

    report = workflow.verify(request)
    ledger_text = config.audit_path.read_text(encoding="utf-8")

    assert report.claims
    assert report.claims[0].claim.text == request.candidate_answer
    assert report.claims[0].evidence
    assert report.claims[0].evidence[0].excerpt in payload["resultList"]["result"][0]["abstractText"]
    assert "Context must not appear" not in ledger_text
    assert "Ground Truth" not in ledger_text
    assert report.actions_used <= config.action_budget
    assert report.events
    assert report.user_context == request.user_context
    claim = report.claims[0].claim
    assert request.candidate_answer[claim.source_start : claim.source_end] == claim.text
    serialized = report.to_dict()
    publication = serialized["claims"][0]["evidence"][0]["publication"]
    assert "abstract" not in publication
    assert publication["abstract_available"] is True


def test_action_budget_marks_remaining_claims_insufficient(tmp_path: Path) -> None:
    config = VerificationConfig(action_budget=2, audit_path=tmp_path / "audit.jsonl")
    workflow = VerificationWorkflow(
        config,
        literature_client=EuropePMCClient(
            config,
            session=FakeSession([FakeResponse()]),
            sleep=lambda _: None,
        ),
    )
    report = workflow.verify(
        VerificationRequest(
            "Question about compound X?",
            "Compound X reduced outcome Y in adults. Compound X reduced outcome Z in adults.",
        )
    )

    assert report.stopped_reason == "action_budget_exhausted"
    assert len(report.claims) == 2
    assert report.claims[1].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert report.actions_used == 2


def test_workflow_network_failure_is_insufficient_and_audited(tmp_path: Path) -> None:
    config = VerificationConfig(
        max_retries=0,
        min_request_interval_seconds=0,
        audit_path=tmp_path / "audit.jsonl",
    )
    client = EuropePMCClient(
        config,
        session=FakeSession([requests.Timeout("simulated timeout")]),
        sleep=lambda _: None,
    )
    report = VerificationWorkflow(config, literature_client=client).verify(
        VerificationRequest(
            "Does compound X change outcome Y?",
            "Compound X changes outcome Y in adult patients.",
        )
    )

    assert report.claims[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert report.claims[0].evidence == ()
    assert any(event.retrieval_status is RetrievalStatus.ERROR for event in report.events)
    assert report.run_id in config.audit_path.read_text(encoding="utf-8")


def test_report_citations_use_only_identifiers_returned_by_api(tmp_path: Path) -> None:
    payload = sample_payload()
    config = VerificationConfig(
        minimum_coverage=0.1,
        audit_path=tmp_path / "audit.jsonl",
        min_request_interval_seconds=0,
    )
    workflow = VerificationWorkflow(
        config,
        literature_client=EuropePMCClient(
            config, session=FakeSession([FakeResponse(payload=payload)]), sleep=lambda _: None
        ),
    )
    report = workflow.verify(
        VerificationRequest(
            "Question about compound X and outcome Y?",
            "Compound X reduced outcome Y in adults.",
        )
    )
    identifiers = {
        identifier
        for record in payload["resultList"]["result"]
        for identifier in (record.get("pmid"), record.get("pmcid"), record.get("doi"))
        if identifier
    }

    for claim in report.claims:
        for excerpt in claim.evidence:
            publication = excerpt.publication
            actual_ids = {publication.pmid, publication.pmcid, publication.doi} - {None}
            assert actual_ids.issubset(identifiers)
            assert publication.source_url.startswith("https://")


def test_offline_cli_fixture_is_disclosed_and_not_a_real_source(tmp_path: Path) -> None:
    from scripts.verify_answer import main

    output = tmp_path / "report.json"
    audit = tmp_path / "audit.jsonl"
    exit_code = main(
        [
            "--question", "Question about fictional compound X?",
            "--answer", "Compound X changes fictional outcome Y in adult patients.",
            "--offline",
            "--output", str(output),
            "--audit-path", str(audit),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert any("OFFLINE MODE" in item for item in report["limitations"])
    assert report["claims"][0]["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["claims"][0]["evidence"] == []
    assert "Europe PMC returned" not in report["claims"][0]["rationale"]


def test_input_personal_identifiers_rejected(tmp_path: Path) -> None:
    config = VerificationConfig(audit_path=tmp_path / "audit.jsonl")
    workflow = VerificationWorkflow(config, literature_client=EuropePMCClient(config))

    with pytest.raises(ValueError, match="personal or credential identifier"):
        workflow.verify(
            VerificationRequest("Question?", "My number is 212-555-0198 and I take medication X.")
        )


def test_privacy_rejection_happens_before_network_call(tmp_path: Path) -> None:
    session = FakeSession([])
    config = VerificationConfig(audit_path=tmp_path / "audit.jsonl")
    workflow = VerificationWorkflow(
        config,
        literature_client=EuropePMCClient(config, session=session),
    )
    with pytest.raises(ValueError, match="No API request was sent"):
        workflow.verify(
            VerificationRequest(
                "Question about drug X?",
                "Patient email patient@example.org reports drug X helped symptoms.",
            )
        )
    assert session.calls == []