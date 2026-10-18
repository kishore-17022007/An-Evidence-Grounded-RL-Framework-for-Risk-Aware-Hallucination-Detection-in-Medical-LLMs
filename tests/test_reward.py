"""Unit tests for budget-aware terminal and action reward components."""

import pytest

from src.common.schemas import VerificationLabel
from src.rl.reward import (
    RewardContext,
    RewardWeights,
    decision_reward,
    retrieval_reward,
)


def adequate_context() -> RewardContext:
    return RewardContext(0.9, 0.9, 0.9, 0.1, 1)


def inadequate_context() -> RewardContext:
    return RewardContext(0.2, 0.1, 0.2, 0.1, 0)


@pytest.mark.parametrize(
    "label, context",
    [
        (VerificationLabel.SUPPORTED, adequate_context()),
        (VerificationLabel.CONTRADICTED, adequate_context()),
        (VerificationLabel.INSUFFICIENT_EVIDENCE, inadequate_context()),
    ],
)
def test_correct_decision_reward_is_positive(
    label: VerificationLabel, context: RewardContext
) -> None:
    reward = decision_reward(
        label,
        label,
        context,
        RewardWeights(),
    )

    assert reward.total > 0
    if label in {VerificationLabel.SUPPORTED, VerificationLabel.CONTRADICTED}:
        assert reward.evidence_quality > 0


def test_missed_hallucination_is_penalized_more_than_false_alarm() -> None:
    missed_hallucination = decision_reward(
        VerificationLabel.SUPPORTED,
        VerificationLabel.CONTRADICTED,
        adequate_context(),
        RewardWeights(),
    )
    false_alarm = decision_reward(
        VerificationLabel.CONTRADICTED,
        VerificationLabel.SUPPORTED,
        adequate_context(),
        RewardWeights(),
    )

    assert missed_hallucination.total < false_alarm.total


def test_abstention_reward_depends_on_evidence_adequacy() -> None:
    justified = decision_reward(
        VerificationLabel.UNVERIFIABLE,
        VerificationLabel.SUPPORTED,
        inadequate_context(),
        RewardWeights(),
        abstained=True,
    )
    unjustified = decision_reward(
        VerificationLabel.UNVERIFIABLE,
        VerificationLabel.SUPPORTED,
        adequate_context(),
        RewardWeights(),
        abstained=True,
    )

    assert justified.total > 0
    assert unjustified.total < 0


def test_conflicting_evidence_justifies_abstention() -> None:
    conflicting = RewardContext(0.8, 0.8, 0.7, 0.9, 1)
    reward = decision_reward(
        VerificationLabel.UNVERIFIABLE,
        VerificationLabel.SUPPORTED,
        conflicting,
        RewardWeights(),
        abstained=True,
    )

    assert reward.total > 0


def test_retrieval_cost_and_redundancy_reduce_reward() -> None:
    useful = retrieval_reward(RewardWeights(), redundant=False, already_sufficient=False)
    redundant = retrieval_reward(RewardWeights(), redundant=True, already_sufficient=False)
    unnecessary = retrieval_reward(RewardWeights(), redundant=False, already_sufficient=True)

    assert useful > redundant
    assert useful > unnecessary


def test_similarity_without_verification_never_gets_quality_bonus() -> None:
    high_similarity_only = RewardContext(1.0, 0.1, 0.9, 0.9, 0)
    reward = decision_reward(
        VerificationLabel.SUPPORTED,
        VerificationLabel.SUPPORTED,
        high_similarity_only,
        RewardWeights(),
    )

    assert reward.evidence_quality == 0.0
    assert reward.total < 0.0


def test_reward_weights_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        RewardWeights(cost_weight=-0.1)