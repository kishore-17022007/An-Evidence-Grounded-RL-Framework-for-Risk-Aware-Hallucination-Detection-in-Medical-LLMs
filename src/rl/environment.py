"""Budget-limited Gymnasium environment for simulated claim verification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.common.schemas import VerificationLabel
from src.rl.action_space import ACTION_DESCRIPTIONS, FINAL_ACTION_LABELS, MedicalAction
from src.rl.reward import (
    RewardContext,
    RewardWeights,
    answer_variant_reward,
    action_cost_reward,
    decision_reward,
    retrieval_reward,
)
from src.rl.state_encoder import ObservableState, encode_state
from src.rl.synthetic_scenarios import (
    EpisodeScenario,
    ObservableScenario,
    SyntheticScenario,
    generate_synthetic_scenarios,
)


class MedicalVerificationEnv(gym.Env[NDArray[np.float32], int]):
    """One-candidate, finite-budget simulation; not a medical verification engine.

    Synthetic and MedHallu scenarios use the same Gymnasium interface. MedHallu
    policy context contains only its question/candidate answer, and dataset
    ``Knowledge`` passages are released only following retrieval actions. Hidden
    answer-level targets are kept private and read only for terminal reward.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        scenarios: Sequence[EpisodeScenario] | None = None,
        *,
        max_action_budget: int = 8,
        max_action_count: int | None = None,
        reward_weights: RewardWeights | None = None,
        training_rewards: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if max_action_budget < 1:
            raise ValueError("max_action_budget must be at least 1")
        if max_action_count is not None and max_action_count < 1:
            raise ValueError("max_action_count must be at least 1")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self._scenarios = tuple(
            scenarios if scenarios is not None else generate_synthetic_scenarios()
        )
        if not self._scenarios:
            raise ValueError("at least one synthetic scenario is required")

        self.max_action_budget = max_action_budget
        self.max_action_count = max_action_count or max_action_budget
        self.reward_weights = reward_weights or RewardWeights()
        self.training_rewards = training_rewards
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(MedicalAction))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        self._signals: ObservableScenario | None = None
        self._training_label: VerificationLabel | None = None
        self._answer_level_target: str | None = None
        self._scenario_id = ""
        self._is_medhallu = False
        self._is_synthetic = True
        self._question: str | None = None
        self._candidate_answer: str | None = None
        self._record_id: str | None = None
        self._configuration: str | None = None
        self._source_split: str | None = None
        self._knowledge_passages: tuple[str, ...] = ()
        self._knowledge_cursor = 0
        self._actions_taken = 0
        self._remaining_budget = max_action_budget
        self._search_count = 0
        self._retrieval_count = 0
        self._verification_count = 0
        self._investigation_count = 0
        self._independent_source_count = 0
        self._evidence_item_count = 0
        self._workflow_stage = 0
        self._evidence_relevance = 0.0
        self._evidence_coverage = 0.0
        self._verification_confidence = 0.0
        self._previous_verification_confidence = 0.0
        self._contradiction_signal = 0.0
        self._terminated = False
        self._truncated = False
        self._last_decision: VerificationLabel | None = None
        self._last_abstained = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a reproducible episode; optionally select ``scenario_index``."""
        super().reset(seed=seed)
        options = options or {}
        scenario_index = options.get("scenario_index")
        if scenario_index is None:
            scenario_index = int(self.np_random.integers(len(self._scenarios)))
        if not isinstance(scenario_index, int) or not 0 <= scenario_index < len(self._scenarios):
            raise ValueError("scenario_index must refer to a configured scenario")

        scenario = self._scenarios[scenario_index]
        self._signals = scenario.signals
        self._training_label = getattr(scenario, "training_label", None)
        self._answer_level_target = getattr(scenario, "answer_level_target", None)
        self._scenario_id = scenario.scenario_id
        self._is_medhallu = hasattr(scenario, "knowledge_passages")
        self._is_synthetic = bool(getattr(scenario, "metadata", {}).get("synthetic", True))
        self._question = getattr(scenario, "question", None)
        self._candidate_answer = getattr(scenario, "candidate_answer", None)
        self._record_id = getattr(scenario, "record_id", None)
        self._configuration = getattr(scenario, "configuration", None)
        self._source_split = getattr(scenario, "source_split", None)
        self._knowledge_passages = tuple(getattr(scenario, "knowledge_passages", ()))
        self._knowledge_cursor = 0
        self._actions_taken = 0
        self._remaining_budget = self.max_action_budget
        self._search_count = 0
        self._retrieval_count = 0
        self._verification_count = 0
        self._investigation_count = 0
        self._independent_source_count = 0
        self._evidence_item_count = (
            0
            if self._is_medhallu
            else int(scenario.signals.initial_coverage > 0)
        )
        self._workflow_stage = 0
        self._evidence_relevance = 0.0 if self._is_medhallu else scenario.signals.initial_relevance
        self._evidence_coverage = 0.0 if self._is_medhallu else scenario.signals.initial_coverage
        self._verification_confidence = scenario.signals.prior_confidence
        self._previous_verification_confidence = scenario.signals.prior_confidence
        self._contradiction_signal = scenario.signals.contradiction_signal
        self._terminated = False
        self._truncated = False
        self._last_decision = None
        self._last_abstained = False
        info: dict[str, Any] = {"synthetic": self._is_synthetic}
        if self._is_medhallu:
            info["policy_context"] = {
                "question": self._question,
                "candidate_answer": self._candidate_answer,
            }
        return self._observation(), info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one action, returning Gymnasium observation/reward/end flags/info."""
        if self._signals is None or (
            self._training_label is None and self._answer_level_target is None
        ):
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("episode is over; call reset() before stepping again")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.action_space.n - 1}]")
        if self._remaining_budget <= 0:
            self._truncated = True
            info = self._info(action=None, redundant=False)
            return self._observation(), -self.reward_weights.exhausted_budget_penalty, False, True, info

        selected = MedicalAction(int(action))
        redundant = False
        new_evidence: dict[str, Any] | None = None
        independent_source_available: bool | None = None
        already_sufficient = self._evidence_coverage >= 0.85 and self._evidence_relevance >= 0.85
        reward = action_cost_reward(self.reward_weights)

        if selected is MedicalAction.RETRIEVE_EVIDENCE:
            if self._is_medhallu:
                new_evidence = self._reveal_next_knowledge("evidence_retrieval")
                redundant = new_evidence is None
                reward = retrieval_reward(
                    self.reward_weights,
                    redundant=redundant,
                    already_sufficient=already_sufficient,
                )
                if new_evidence is not None:
                    self._retrieval_count += 1
                    self._search_count += 1
                    self._evidence_item_count += 1
            else:
                redundant = self._retrieval_count > 0
                reward = retrieval_reward(
                    self.reward_weights,
                    redundant=redundant,
                    already_sufficient=already_sufficient,
                )
                if not redundant:
                    self._evidence_relevance = max(
                        self._evidence_relevance, self._signals.retrieval_relevance_gain
                    )
                    self._evidence_coverage = max(
                        self._evidence_coverage, self._signals.retrieval_coverage_gain
                    )
                self._retrieval_count += 1
                self._search_count += 1
                self._evidence_item_count += int(not redundant)
            self._workflow_stage = max(self._workflow_stage, 1)
        elif selected is MedicalAction.VERIFY_CURRENT_CLAIM:
            redundant = self._verification_count > 0
            if not redundant:
                self._previous_verification_confidence = self._verification_confidence
                self._verification_confidence = min(
                    1.0,
                    0.45 * self._evidence_coverage
                    + 0.30 * self._evidence_relevance
                    + 0.25 * self._signals.prior_confidence,
                )
                self._verification_count += 1
                self._workflow_stage = max(self._workflow_stage, 2)
            else:
                reward = action_cost_reward(self.reward_weights, redundant=True)
        elif selected is MedicalAction.INVESTIGATE_CONTRADICTION:
            if self._is_medhallu:
                new_evidence = self._reveal_next_knowledge("contradiction_investigation")
                redundant = new_evidence is None
                if not redundant:
                    self._investigation_count += 1
                    self._search_count += 1
                    self._evidence_item_count += 1
                    self._workflow_stage = max(self._workflow_stage, 3)
            elif self._investigation_count == 0:
                redundant = False
                self._contradiction_signal = max(
                    self._contradiction_signal,
                    self._signals.investigation_contradiction_gain,
                )
                self._investigation_count += 1
                self._search_count += 1
                self._evidence_item_count += 1
                self._workflow_stage = max(self._workflow_stage, 3)
            else:
                redundant = True
                reward = action_cost_reward(self.reward_weights, redundant=True)
        elif selected is MedicalAction.REQUEST_INDEPENDENT_SOURCE:
            redundant = self._independent_source_count > 0
            if self._is_medhallu:
                independent_source_available = False
                redundant = self._independent_source_count > 0
                if not redundant:
                    self._independent_source_count += 1
            elif not redundant:
                self._evidence_coverage = min(
                    1.0,
                    self._evidence_coverage + self._signals.independent_coverage_gain,
                )
                self._contradiction_signal = max(
                    self._contradiction_signal,
                    self._signals.independent_contradiction_signal,
                )
                self._independent_source_count += 1
                self._search_count += 1
                self._evidence_item_count += 1
                self._workflow_stage = max(self._workflow_stage, 4)
            else:
                reward = action_cost_reward(self.reward_weights, redundant=True)
        elif selected in FINAL_ACTION_LABELS:
            self._last_decision = FINAL_ACTION_LABELS[selected]
            self._last_abstained = selected is MedicalAction.ABSTAIN_OR_ESCALATE
            context = self._reward_context()
            if self.training_rewards and self._training_label is not None:
                breakdown = decision_reward(
                    self._last_decision,
                    self._training_label,
                    context,
                    self.reward_weights,
                    abstained=self._last_abstained,
                )
                reward += breakdown.total
            elif self.training_rewards and self._answer_level_target is not None:
                breakdown = answer_variant_reward(
                    self._last_decision,
                    self._answer_level_target,
                    context,
                    self.reward_weights,
                    abstained=self._last_abstained,
                )
                reward += breakdown.total
            self._terminated = True
            self._workflow_stage = 4

        self._actions_taken += 1
        self._remaining_budget -= 1
        if not self._terminated and (
            self._remaining_budget == 0 or self._actions_taken >= self.max_action_count
        ):
            self._truncated = True
            reward -= self.reward_weights.cost_weight * self.reward_weights.exhausted_budget_penalty

        info = self._info(action=selected, redundant=redundant)
        if new_evidence is not None:
            info["newly_retrieved_evidence"] = new_evidence
        if independent_source_available is not None:
            info["independent_source_available"] = independent_source_available
        return self._observation(), float(reward), self._terminated, self._truncated, info

    def render(self) -> str | None:
        """Return a concise public-state summary in ANSI mode."""
        if self.render_mode != "ansi":
            return None
        return (
            f"stage={self._workflow_stage} searches={self._search_count} "
            f"coverage={self._evidence_coverage:.2f} "
            f"confidence={self._verification_confidence:.2f} "
            f"budget={self._remaining_budget}"
        )

    def _observation(self) -> NDArray[np.float32]:
        if self._signals is None:
            raise RuntimeError("reset() must be called before reading an observation")
        state = ObservableState(
            claim_complexity=self._signals.claim_complexity,
            evidence_relevance=self._evidence_relevance,
            evidence_coverage=self._evidence_coverage,
            verification_confidence=self._verification_confidence,
            contradiction_signal=self._contradiction_signal,
            search_count=self._search_count,
            remaining_budget=self._remaining_budget,
            max_action_budget=self.max_action_budget,
            workflow_stage=self._workflow_stage,
            previous_verification_confidence=self._previous_verification_confidence,
            verification_count=self._verification_count,
            retrieval_count=self._retrieval_count,
            independent_source_count=self._independent_source_count,
            contradiction_investigation_count=self._investigation_count,
            evidence_item_count=self._evidence_item_count,
            max_action_count=self.max_action_count,
        )
        return encode_state(state)

    def _reward_context(self) -> RewardContext:
        return RewardContext(
            evidence_relevance=self._evidence_relevance,
            evidence_coverage=self._evidence_coverage,
            verification_confidence=self._verification_confidence,
            contradiction_signal=self._contradiction_signal,
            verification_count=self._verification_count,
        )

    def _reveal_next_knowledge(self, retrieval_method: str) -> dict[str, Any] | None:
        """Reveal one dataset-provided passage and update public evidence signals."""
        if self._knowledge_cursor >= len(self._knowledge_passages):
            return None
        passage_index = self._knowledge_cursor
        passage = self._knowledge_passages[passage_index]
        self._knowledge_cursor += 1
        candidate_tokens = set((self._candidate_answer or "").casefold().split())
        passage_tokens = set(passage.casefold().split())
        lexical_relevance = (
            len(candidate_tokens & passage_tokens) / len(candidate_tokens)
            if candidate_tokens
            else 0.0
        )
        self._evidence_relevance = max(
            self._evidence_relevance, min(1.0, lexical_relevance)
        )
        self._evidence_coverage = self._knowledge_cursor / max(
            1, len(self._knowledge_passages)
        )
        return {
            "evidence_id": f"{self._record_id}:Knowledge:{passage_index}",
            "source_name": "MedHallu Knowledge field",
            "source_identifier": None,
            "source_url": "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu",
            "text": passage,
            "retrieval_method": retrieval_method,
            "metadata": {
                "dataset_id": "UTAustin-AIHealth/MedHallu",
                "configuration": self._configuration,
                "source_split": self._source_split,
                "record_id": self._record_id,
                "provenance_note": "Dataset context passage; original biomedical publication identifier is not supplied in this record.",
                "lexical_relevance_is_not_verification": True,
            },
        }

    def _info(self, action: MedicalAction | None, *, redundant: bool) -> dict[str, Any]:
        return {
            "synthetic": self._is_synthetic,
            "action": ACTION_DESCRIPTIONS[action] if action is not None else None,
            "redundant": redundant,
            "actions_taken": self._actions_taken,
            "search_count": self._search_count,
            "remaining_budget": self._remaining_budget,
            "decision": self._last_decision.value if self._last_decision else None,
            "abstained": self._last_abstained,
        }