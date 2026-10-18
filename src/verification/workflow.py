"""Deterministic live evidence-verification workflow service."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.verification.assessment import assess_claim, rank_evidence
from src.verification.audit import AuditLedger
from src.verification.claims import extract_claims, normalize_search_text
from src.verification.config import VerificationConfig
from src.verification.errors import EvidenceRetrievalError, VerificationInputError
from src.verification.europe_pmc import API_NAME, EuropePMCClient, build_search_query
from src.verification.models import (
    ClaimAssessment,
    EvidenceStance,
    RetrievalStatus,
    VerificationReport,
    VerificationRequest,
    VerificationStatus,
    WorkflowEvent,
)
from src.verification.policy_bridge import SimulatorPolicyBridge


LOGGER = logging.getLogger(__name__)
_POTENTIALLY_PERSONAL = re.compile(
    r"(?:\b\d{3}[-. ]?\d{2}[-. ]?\d{4}\b|\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b(?:api[_-]?key|access[_-]?token|password)\s*[:=])",
    re.IGNORECASE,
)


class VerificationWorkflow:
    """Service orchestrating deterministic components and Europe PMC retrieval.

    No language model is configured. DQN/HRL policies remain simulator-only and
    are never passed claim text or retrieved evidence as a fabricated observation.
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
        *,
        literature_client: EuropePMCClient | None = None,
        audit_ledger: AuditLedger | None = None,
        policy_bridge: SimulatorPolicyBridge | None = None,
    ) -> None:
        self.config = config or VerificationConfig()
        self.literature_client = literature_client or EuropePMCClient(self.config)
        self.audit_ledger = audit_ledger or AuditLedger(self.config.audit_path)
        self.policy_bridge = policy_bridge or SimulatorPolicyBridge()

    def verify(self, request: VerificationRequest) -> VerificationReport:
        """Verify claims using official API abstracts and the transparent baseline."""
        normalized = self._validate_and_normalize(request)
        run_id = str(uuid.uuid4())
        events: list[WorkflowEvent] = []
        assessments: list[ClaimAssessment] = []
        actions_used = 0
        stop_reason = "all_claims_processed"
        limitations = [
            "This educational prototype is not clinically validated and is not for diagnosis or treatment decisions.",
            "Claim extraction is deterministic sentence segmentation, not an LLM or trained claim extractor.",
            "Assessment is a conservative lexical/exact-text baseline, not semantic entailment or calibrated probability.",
            "DQN/HRL checkpoints were trained in a simulator and are not invoked for live literature decisions.",
            "Europe PMC abstracts/metadata were used only; this workflow does not fetch full text.",
        ]
        if normalized.user_context:
            limitations.append(
                "Optional user context is retained in the returned request context but is not sent to Europe PMC or written to the audit ledger."
            )
        source_api = getattr(self.literature_client, "api_name", API_NAME)
        is_live_api = getattr(self.literature_client, "live_api", True)

        claims = extract_claims(normalized.candidate_answer, run_id=run_id)
        self._record(
            events,
            WorkflowEvent(
                run_id=run_id,
                timestamp=datetime.now(UTC),
                event_type="input_validated_and_claims_extracted",
                claim_id=None,
                source_api="deterministic_sentence_segmenter",
                tool_outcome=f"extracted_claim_count={len(claims)}",
                rationale="Question/answer validated; source spans remain indexed to the original candidate answer; raw inputs are not written to audit.",
            ),
        )

        for claim in claims:
            # Explicitly record that existing RL observation/action semantics cannot
            # safely represent live text, without fabricating an observation vector.
            self._record(
                events,
                WorkflowEvent(
                    run_id=run_id,
                    timestamp=datetime.now(UTC),
                    event_type="policy_compatibility_check",
                    claim_id=claim.claim_id,
                    policy_name="Phase 3 DQN / Phase 4 HRL",
                    selected_action="not_invoked",
                    tool_outcome=(
                        "Existing policies accept the 15-feature simulator state; "
                        "live claim and retrieved abstract are not encoded."
                    ),
                    rationale="No silent observation dimension change or synthetic live-state encoding.",
                ),
            )

            if not claim.checkable:
                assessment = assess_claim(
                    claim,
                    (),
                    retrieval_status=RetrievalStatus.SKIPPED_BUDGET,
                    retrieval_error=None,
                    config=self.config,
                )
                assessments.append(assessment)
                self._record_assessment(events, run_id, assessment)
                continue

            if actions_used + 2 > self.config.action_budget:
                stop_reason = "action_budget_exhausted"
                assessment = assess_claim(
                    claim,
                    (),
                    retrieval_status=RetrievalStatus.SKIPPED_BUDGET,
                    retrieval_error=None,
                    config=self.config,
                )
                assessments.append(assessment)
                self._record(
                    events,
                    WorkflowEvent(
                        run_id=run_id,
                        timestamp=datetime.now(UTC),
                        event_type="workflow_stopped",
                        claim_id=claim.claim_id,
                        retrieval_status=RetrievalStatus.SKIPPED_BUDGET,
                        tool_outcome="No search or assessment action remained in the configured budget.",
                        assessment_status=assessment.status,
                        rationale="Budget limit; claim left unresolved rather than forced to a label.",
                    ),
                )
                self._record_assessment(events, run_id, assessment)
                # Remaining claims receive explicit insufficient-evidence records.
                for remaining in claims[len(assessments) :]:
                    if remaining.claim_id == claim.claim_id:
                        continue
                    rest = assess_claim(
                        remaining,
                        (),
                        retrieval_status=RetrievalStatus.SKIPPED_BUDGET,
                        retrieval_error=None,
                        config=self.config,
                    )
                    assessments.append(rest)
                    self._record_assessment(events, run_id, rest)
                break

            query = build_search_query(
                normalize_search_text(claim.text),
                normalize_search_text(normalized.question),
            )
            actions_used += 1
            self._record(
                events,
                WorkflowEvent(
                    run_id=run_id,
                    timestamp=datetime.now(UTC),
                    event_type="literature_search_requested",
                    claim_id=claim.claim_id,
                    query_sent=query,
                    source_api=getattr(self.literature_client, "api_name", API_NAME),
                    retrieval_status=None,
                    policy_name="deterministic_workflow",
                    selected_action="SEARCH_EUROPE_PMC",
                    tool_outcome=(
                        "Sanitized query sent over HTTPS to the official Europe PMC REST endpoint."
                        if is_live_api
                        else "Sanitized query evaluated against the explicit local synthetic fixture; no network request was made."
                    ),
                    rationale="Search terms are derived from the exact claim and question; user_context is not transmitted.",
                ),
            )

            retrieval_error: str | None = None
            try:
                publications = self.literature_client.search(
                    query, max_results=self.config.max_results_per_claim
                )
                retrieved_status = RetrievalStatus.SUCCESS if publications else RetrievalStatus.EMPTY
            except EvidenceRetrievalError as error:
                publications = ()
                retrieved_status = RetrievalStatus.ERROR
                retrieval_error = type(error).__name__
            actions_used += 1
            evidence = rank_evidence(
                claim,
                publications,
                limit=self.config.max_results_per_claim,
            )
            self._record(
                events,
                WorkflowEvent(
                    run_id=run_id,
                    timestamp=datetime.now(UTC),
                    event_type="literature_search_completed",
                    claim_id=claim.claim_id,
                    query_sent=query,
                    source_api=getattr(self.literature_client, "api_name", API_NAME),
                    retrieval_status=(
                        RetrievalStatus.ERROR if retrieved_status is RetrievalStatus.ERROR else
                        RetrievalStatus.SUCCESS if evidence else
                        RetrievalStatus.NO_RELEVANT_EVIDENCE if publications else
                        RetrievalStatus.EMPTY
                    ),
                    publication_identifiers=tuple(
                        identifier
                        for record in publications
                        for identifier in record.identifiers
                    ),
                    tool_outcome=f"publications={len(publications)} relevant_abstract_excerpts={len(evidence)}",
                    rationale=(
                        f"{retrieval_error}"
                        if retrieval_error
                        else "Metadata/abstracts only; exact source identifiers retained where supplied."
                    ),
                ),
            )
            assessment = assess_claim(
                claim,
                evidence,
                retrieval_status=(
                    retrieved_status
                    if retrieved_status is not RetrievalStatus.SUCCESS or evidence
                    else RetrievalStatus.NO_RELEVANT_EVIDENCE
                ),
                retrieval_error=retrieval_error,
                config=self.config,
            )
            assessments.append(assessment)
            self._record_assessment(events, run_id, assessment)

        if len(assessments) >= len(claims) and stop_reason != "action_budget_exhausted":
            stop_reason = "claims_processed"
        if not claims:
            stop_reason = "no_factual_claims_extracted"
            limitations.append("No sentence-level claims were extracted; no literature search was performed.")

        stop_status = (
            assessments[-1].status
            if assessments
            else VerificationStatus.UNVERIFIABLE
        )
        self._record(
            events,
            WorkflowEvent(
                run_id=run_id,
                timestamp=datetime.now(UTC),
                event_type="workflow_stopped",
                claim_id=None,
                tool_outcome=stop_reason,
                assessment_status=stop_status,
                rationale=f"actions_used={actions_used} action_budget={self.config.action_budget}",
            ),
        )
        LOGGER.info(
            "Verification workflow completed",
            extra={
                "run_id": run_id,
                "claim_count": len(assessments),
                "actions_used": actions_used,
                "stop_reason": stop_reason,
            },
        )
        return VerificationReport(
            run_id=run_id,
            created_at=datetime.now(UTC),
            claims=tuple(assessments),
            events=tuple(events),
            action_budget=self.config.action_budget,
            actions_used=actions_used,
            stopped_reason=stop_reason,
            workflow_method="deterministic_workflow_europe_pmc_lexical_baseline",
            limitations=tuple(dict.fromkeys(limitations)),
            original_question=normalized.question,
            original_answer=normalized.candidate_answer,
            user_context=normalized.user_context,
        )

    def _validate_and_normalize(self, request: VerificationRequest) -> VerificationRequest:
        # Preserve caller text exactly so claim offsets continue to index the answer.
        question = request.question
        answer = request.candidate_answer
        context = request.user_context
        if len(question) > self.config.max_question_characters:
            raise VerificationInputError("question exceeds configured character limit")
        if len(answer) > self.config.max_answer_characters:
            raise VerificationInputError("candidate answer exceeds configured character limit")
        if context and len(context) > self.config.max_context_characters:
            raise VerificationInputError("user_context exceeds configured character limit")
        if _POTENTIALLY_PERSONAL.search(question) or _POTENTIALLY_PERSONAL.search(answer):
            raise VerificationInputError(
                "Input appears to contain a personal or credential identifier; "
                "remove it before using this research tool. No API request was sent."
            )
        if context and _POTENTIALLY_PERSONAL.search(context):
            raise VerificationInputError(
                "user_context appears to contain personal or credential data; remove it before use."
            )
        claims = extract_claims(answer, run_id="validation")
        if len(claims) > self.config.max_claims:
            raise VerificationInputError("candidate answer exceeds configured claim count")
        return VerificationRequest(question, answer, context)

    def _record_assessment(
        self,
        events: list[WorkflowEvent],
        run_id: str,
        assessment: ClaimAssessment,
    ) -> None:
        for excerpt in assessment.evidence:
            publication = excerpt.publication
            self._record(
                events,
                WorkflowEvent(
                    run_id=run_id,
                    timestamp=datetime.now(UTC),
                    event_type="evidence_assessed",
                    claim_id=assessment.claim.claim_id,
                    source_api=getattr(self.literature_client, "api_name", API_NAME),
                    publication_identifiers=publication.identifiers,
                    evidence_id=excerpt.evidence_id,
                    evidence_excerpt=excerpt.excerpt,
                    assessment_status=assessment.status,
                    policy_name="deterministic_lexical_baseline",
                    rationale=(
                        f"stance={assessment.stance.value}; coverage={assessment.evidence_coverage:.3f}; "
                        f"citation_available={publication.primary_identifier is not None}"
                    ),
                ),
            )
        self._record(
            events,
            WorkflowEvent(
                run_id=run_id,
                timestamp=datetime.now(UTC),
                event_type="claim_decision",
                claim_id=assessment.claim.claim_id,
                assessment_status=assessment.status,
                policy_name="deterministic_lexical_baseline",
                selected_action=assessment.stance.value,
                tool_outcome=assessment.status.value,
                rationale=assessment.rationale,
            ),
        )

    def _record(self, events: list[WorkflowEvent], event: WorkflowEvent) -> None:
        events.append(event)
        self.audit_ledger.append(event)
