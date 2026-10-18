"""Balanced, fictional scenarios for training and testing the environment."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Mapping, Protocol

from src.common.schemas import VerificationLabel


class ScenarioType(StrEnum):
    """Scenario coverage classes; these are not real-world dataset labels."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ObservableScenario:
    """Inference-time signals and transition dynamics, separate from truth."""

    claim_complexity: float
    initial_relevance: float
    initial_coverage: float
    prior_confidence: float
    contradiction_signal: float
    retrieval_relevance_gain: float
    retrieval_coverage_gain: float
    investigation_contradiction_gain: float
    independent_coverage_gain: float
    independent_contradiction_signal: float

    def __post_init__(self) -> None:
        for item in fields(self):
            field_name = item.name
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")


class EpisodeScenario(Protocol):
    """Common fields required by the environment for synthetic/data episodes."""

    scenario_id: str
    signals: ObservableScenario
    metadata: Mapping[str, str | bool]


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    """One synthetic training episode with isolated hidden training truth."""

    scenario_id: str
    scenario_type: ScenarioType
    signals: ObservableScenario
    training_label: VerificationLabel = field(repr=False)
    metadata: Mapping[str, str | bool] = field(
        default_factory=lambda: {"synthetic": True, "dataset": "development_only"}
    )

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if self.metadata.get("synthetic") is not True:
            raise ValueError("all generated scenarios must be marked synthetic")


_BASE_PROFILES: dict[
    ScenarioType,
    tuple[VerificationLabel, tuple[float, ...]],
] = {
    ScenarioType.SUPPORTED: (
        VerificationLabel.SUPPORTED,
        (0.35, 0.10, 0.05, 0.62, 0.02, 0.78, 0.76, 0.10, 0.42, 0.08),
    ),
    ScenarioType.CONTRADICTED: (
        VerificationLabel.CONTRADICTED,
        (0.68, 0.08, 0.05, 0.40, 0.10, 0.48, 0.40, 0.72, 0.38, 0.82),
    ),
    ScenarioType.INSUFFICIENT_EVIDENCE: (
        VerificationLabel.INSUFFICIENT_EVIDENCE,
        (0.52, 0.05, 0.02, 0.28, 0.08, 0.22, 0.18, 0.12, 0.16, 0.10),
    ),
    ScenarioType.CONFLICTING_EVIDENCE: (
        VerificationLabel.INSUFFICIENT_EVIDENCE,
        (0.74, 0.44, 0.46, 0.48, 0.55, 0.42, 0.30, 0.24, 0.28, 0.72),
    ),
    ScenarioType.UNCERTAIN: (
        VerificationLabel.UNVERIFIABLE,
        (0.88, 0.12, 0.10, 0.22, 0.18, 0.25, 0.20, 0.18, 0.16, 0.24),
    ),
}


def generate_synthetic_scenarios(
    episodes_per_type: int = 4, *, seed: int = 17
) -> list[SyntheticScenario]:
    """Generate equally represented fictional scenario types deterministically."""
    if episodes_per_type < 1:
        raise ValueError("episodes_per_type must be at least 1")

    rng = random.Random(seed)
    scenarios: list[SyntheticScenario] = []
    signal_fields = tuple(ObservableScenario.__dataclass_fields__)
    scenario_number = 0
    for scenario_type, (training_label, base_values) in _BASE_PROFILES.items():
        for index in range(episodes_per_type):
            jittered = tuple(
                min(1.0, max(0.0, value + rng.uniform(-0.035, 0.035)))
                for value in base_values
            )
            signals = ObservableScenario(**dict(zip(signal_fields, jittered)))
            scenarios.append(
                SyntheticScenario(
                    scenario_id=f"synthetic-episode-{scenario_number:06d}",
                    scenario_type=scenario_type,
                    signals=signals,
                    training_label=training_label,
                    metadata={
                        "synthetic": True,
                        "dataset": "fictional_training_scenarios",
                        "scenario_type": scenario_type.value,
                    },
                )
            )
            scenario_number += 1
    return scenarios