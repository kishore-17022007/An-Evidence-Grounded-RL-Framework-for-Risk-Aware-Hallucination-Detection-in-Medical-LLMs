"""Configurable quality, risk, abstention, and action-cost reward design."""

from __future__ import annotations

from dataclasses import dataclass, fields

from src.common.schemas import VerificationLabel


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Reward scales; quality and cost weights expose the trade-off directly."""

    quality_weight: float = 1.0
    cost_weight: float = 0.25
    correct_decision: float = 5.0
    missed_hallucination_penalty: float = 9.0
    false_alarm_penalty: float = 5.0
    other_incorrect_penalty: float = 3.0
    justified_abstention_reward: float = 2.0
    unjustified_abstention_penalty: float = 2.0
    reasonable_uncertainty_reward: float = 1.0
    evidence_quality_bonus: float = 2.0
    unsupported_assertion_penalty: float = 2.0
    action_cost: float = 0.10
    unnecessary_retrieval_penalty: float = 0.75
    redundant_action_penalty: float = 0.50
    exhausted_budget_penalty: float = 1.0

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Observable episode quality at decision time; does not include hidden truth."""

    evidence_relevance: float
    evidence_coverage: float
    verification_confidence: float
    contradiction_signal: float
    verification_count: int


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Inspectable components for simulation diagnostics and unit tests."""

    decision_quality: float = 0.0
    evidence_quality: float = 0.0
    action_cost: float = 0.0
    penalty: float = 0.0

    @property
    def total(self) -> float:
        return self.decision_quality + self.evidence_quality - self.action_cost - self.penalty


def action_cost_reward(weights: RewardWeights, *, redundant: bool = False) -> float:
    """Return the negative cost for an ordinary or redundant action."""
    multiplier = weights.cost_weight
    cost = weights.action_cost
    if redundant:
        cost += weights.redundant_action_penalty
    return -multiplier * cost


def retrieval_reward(
    weights: RewardWeights, *, redundant: bool, already_sufficient: bool
) -> float:
    """Penalize retrieval that repeats itself or adds little to sufficient evidence."""
    penalty = weights.action_cost
    if redundant:
        penalty += weights.redundant_action_penalty
    if already_sufficient:
        penalty += weights.unnecessary_retrieval_penalty
    return -weights.cost_weight * penalty


def decision_reward(
    decision: VerificationLabel,
    ground_truth: VerificationLabel,
    context: RewardContext,
    weights: RewardWeights,
    *,
    abstained: bool = False,
) -> RewardBreakdown:
    """Score a terminal decision, using hidden truth only for training reward.

    Evidence relevance alone never earns a positive evidence-quality bonus:
    verification must have occurred and coverage must meet the adequacy threshold.
    """
    context_has_conflict = context.contradiction_signal >= 0.5
    evidence_adequate = (
        context.evidence_coverage >= 0.5
        and context.evidence_relevance >= 0.5
        and context.verification_count > 0
        and context.verification_confidence >= 0.5
    )

    if abstained:
        justified = (
            ground_truth in {
                VerificationLabel.INSUFFICIENT_EVIDENCE,
                VerificationLabel.UNVERIFIABLE,
            }
            or not evidence_adequate
            or context_has_conflict
        )
        quality = (
            weights.justified_abstention_reward
            if justified
            else -weights.unjustified_abstention_penalty
        )
        return RewardBreakdown(decision_quality=weights.quality_weight * quality)

    if decision is ground_truth and (
        decision
        not in {VerificationLabel.SUPPORTED, VerificationLabel.CONTRADICTED}
        or evidence_adequate
    ):
        quality = weights.correct_decision
    elif (
        ground_truth is VerificationLabel.CONTRADICTED
        and decision is VerificationLabel.SUPPORTED
    ):
        quality = -weights.missed_hallucination_penalty
    elif (
        ground_truth is VerificationLabel.SUPPORTED
        and decision is VerificationLabel.CONTRADICTED
    ):
        quality = -weights.false_alarm_penalty
    elif decision is VerificationLabel.INSUFFICIENT_EVIDENCE and (
        not evidence_adequate or context_has_conflict
    ):
        quality = weights.reasonable_uncertainty_reward
    elif decision in {VerificationLabel.SUPPORTED, VerificationLabel.CONTRADICTED} and not evidence_adequate:
        quality = -weights.unsupported_assertion_penalty
    else:
        quality = -weights.other_incorrect_penalty

    evidence_quality = 0.0
    unsupported_penalty = 0.0
    if decision in {VerificationLabel.SUPPORTED, VerificationLabel.CONTRADICTED}:
        if evidence_adequate:
            evidence_quality = (
                weights.quality_weight
                * weights.evidence_quality_bonus
                * min(context.evidence_coverage, context.verification_confidence)
            )

    return RewardBreakdown(
        decision_quality=weights.quality_weight * quality,
        evidence_quality=evidence_quality,
        penalty=unsupported_penalty,
    )


def answer_variant_reward(
    decision: VerificationLabel,
    answer_level_target: str,
    context: RewardContext,
    weights: RewardWeights,
    *,
    abstained: bool = False,
) -> RewardBreakdown:
    """Score a MedHallu answer-origin target after the policy decides.

    The dataset provides answer text fields rather than a four-class label.
    This reward maps candidate provenance to a binary answer-level expected
    decision only at terminal reward time; it does not relabel stored records.
    """
    expected_decision = {
        "ground_truth_answer": VerificationLabel.SUPPORTED,
        "hallucinated_answer": VerificationLabel.CONTRADICTED,
    }.get(answer_level_target)
    if expected_decision is None:
        raise ValueError(f"unsupported MedHallu answer-level target: {answer_level_target}")
    return decision_reward(
        decision,
        expected_decision,
        context,
        weights,
        abstained=abstained,
    )