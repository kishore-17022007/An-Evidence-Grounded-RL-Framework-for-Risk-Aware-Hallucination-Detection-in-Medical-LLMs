"""Leakage-resistant fixed-size encoding of observable episode state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


OBSERVATION_FEATURES: tuple[str, ...] = (
    "claim_complexity",
    "current_evidence_relevance",
    "evidence_coverage",
    "verification_confidence",
    "contradiction_signal",
    "search_count_normalized",
    "remaining_budget_fraction",
    "workflow_stage_normalized",
    "previous_verification_confidence",
    "verification_count_normalized",
    "retrieval_redundancy_fraction",
    "independent_source_fraction",
    "contradiction_investigation_fraction",
    "uncertainty_estimate",
    "evidence_item_count_normalized",
)


@dataclass(frozen=True, slots=True)
class ObservableState:
    """Only inference-time signals; intentionally has no ground-truth field."""

    claim_complexity: float
    evidence_relevance: float
    evidence_coverage: float
    verification_confidence: float
    contradiction_signal: float
    search_count: int
    remaining_budget: int
    max_action_budget: int
    workflow_stage: int
    previous_verification_confidence: float
    verification_count: int
    retrieval_count: int
    independent_source_count: int
    contradiction_investigation_count: int
    evidence_item_count: int
    max_action_count: int


def encode_state(state: ObservableState) -> NDArray[np.float32]:
    """Encode public state into 15 float features, each bounded in [0, 1].

    Feature order and meaning are documented in ``OBSERVATION_FEATURES``.
    Counts are divided by their episode limits and capped at one. Ground truth,
    scenario identifiers, and future transitions are not accepted as inputs.
    """
    max_budget = max(1, state.max_action_budget)
    max_actions = max(1, state.max_action_count)
    values = np.asarray(
        [
            state.claim_complexity,
            state.evidence_relevance,
            state.evidence_coverage,
            state.verification_confidence,
            state.contradiction_signal,
            min(1.0, state.search_count / max_budget),
            state.remaining_budget / max_budget,
            min(1.0, state.workflow_stage / 4.0),
            state.previous_verification_confidence,
            min(1.0, state.verification_count / max_actions),
            min(1.0, max(0, state.retrieval_count - 1) / max_budget),
            min(1.0, state.independent_source_count / max_budget),
            min(1.0, state.contradiction_investigation_count / max_budget),
            1.0 - state.verification_confidence,
            min(1.0, state.evidence_item_count / max_budget),
        ],
        dtype=np.float32,
    )
    return np.clip(values, 0.0, 1.0)