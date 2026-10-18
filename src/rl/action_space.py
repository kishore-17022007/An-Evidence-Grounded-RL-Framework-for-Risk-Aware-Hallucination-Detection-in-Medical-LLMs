"""Discrete action vocabulary for medical claim verification episodes."""

from __future__ import annotations

from enum import IntEnum

from src.common.schemas import VerificationLabel


class MedicalAction(IntEnum):
    """Actions available to an agent; integer values match Gymnasium indices."""

    RETRIEVE_EVIDENCE = 0
    VERIFY_CURRENT_CLAIM = 1
    INVESTIGATE_CONTRADICTION = 2
    REQUEST_INDEPENDENT_SOURCE = 3
    FINALIZE_SUPPORTED = 4
    FINALIZE_CONTRADICTED = 5
    FINALIZE_INSUFFICIENT_EVIDENCE = 6
    ABSTAIN_OR_ESCALATE = 7


ACTION_DESCRIPTIONS: dict[MedicalAction, str] = {
    MedicalAction.RETRIEVE_EVIDENCE: "Retrieve evidence for the current claim",
    MedicalAction.VERIFY_CURRENT_CLAIM: "Assess the claim against currently observed evidence",
    MedicalAction.INVESTIGATE_CONTRADICTION: "Search for evidence that could contradict the claim",
    MedicalAction.REQUEST_INDEPENDENT_SOURCE: "Request an independent evidence source",
    MedicalAction.FINALIZE_SUPPORTED: "Finalize the claim as supported",
    MedicalAction.FINALIZE_CONTRADICTED: "Finalize the claim as contradicted",
    MedicalAction.FINALIZE_INSUFFICIENT_EVIDENCE: "Finalize with insufficient evidence",
    MedicalAction.ABSTAIN_OR_ESCALATE: "Abstain or escalate for human review",
}


FINAL_ACTION_LABELS: dict[MedicalAction, VerificationLabel] = {
    MedicalAction.FINALIZE_SUPPORTED: VerificationLabel.SUPPORTED,
    MedicalAction.FINALIZE_CONTRADICTED: VerificationLabel.CONTRADICTED,
    MedicalAction.FINALIZE_INSUFFICIENT_EVIDENCE: VerificationLabel.INSUFFICIENT_EVIDENCE,
    MedicalAction.ABSTAIN_OR_ESCALATE: VerificationLabel.UNVERIFIABLE,
}