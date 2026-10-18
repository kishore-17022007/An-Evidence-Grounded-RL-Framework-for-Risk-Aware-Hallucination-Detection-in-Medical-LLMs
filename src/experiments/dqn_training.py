"""Training and validation orchestration for DQN on MedHallu simulations."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.schemas import VerificationLabel
from src.experiments.medhallu_adapter import MedHalluEpisode
from src.rl.action_space import MedicalAction
from src.rl.dqn import DQNAgent, DQNConfig, Transition
from src.rl.environment import MedicalVerificationEnv


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Run-level parameters, including explicitly separate train/validation splits."""

    episodes: int = 100
    seed: int = 17
    max_action_budget: int = 8
    training_configuration: str = "pqa_labeled"
    training_split: str = "train"
    validation_split: str = "validation"
    training_scenario_limit: int | None = None
    validation_scenario_limit: int | None = None
    epsilon_decay: float = 0.995
    output_root: Path = Path("results/dqn")
    checkpoint_root: Path = Path("models/dqn")

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.max_action_budget < 1:
            raise ValueError("max_action_budget must be at least 1")
        if self.training_split != "train":
            raise ValueError("DQN training must use the project train split")
        if self.validation_split != "validation":
            raise ValueError("model selection must use the validation split")
        if self.training_scenario_limit is not None and self.training_scenario_limit < 1:
            raise ValueError("training_scenario_limit must be positive")
        if self.validation_scenario_limit is not None and self.validation_scenario_limit < 1:
            raise ValueError("validation_scenario_limit must be positive")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")


def _make_env(
    scenarios: list[MedHalluEpisode],
    max_action_budget: int,
    *,
    training_rewards: bool = True,
) -> MedicalVerificationEnv:
    if not scenarios:
        raise ValueError("scenario collection is empty; check data and split paths")
    return MedicalVerificationEnv(
        scenarios=scenarios,
        max_action_budget=max_action_budget,
        max_action_count=max_action_budget,
        training_rewards=training_rewards,
    )


def evaluate_greedily(
    agent: DQNAgent,
    scenarios: list[MedHalluEpisode],
    *,
    max_action_budget: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate policy predictions first; inspect answer targets only afterward."""
    if not scenarios:
        raise ValueError("validation scenarios are empty")
    selected = scenarios[:limit] if limit is not None else scenarios
    env = _make_env(selected, max_action_budget, training_rewards=False)
    predictions: list[dict[str, Any]] = []
    for index, scenario in enumerate(selected):
        observation, _ = env.reset(
            seed=index, options={"scenario_index": index}
        )
        terminated = truncated = False
        final_decision: str | None = None
        while not (terminated or truncated):
            action = agent.select_action(observation, explore=False, epsilon=0.0)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated:
                final_decision = info["decision"]
        predictions.append(
            {
                "scenario_id": scenario.scenario_id,
                "configuration": scenario.configuration,
                "record_id": scenario.record_id,
                "prediction": final_decision,
                "abstained": final_decision == VerificationLabel.UNVERIFIABLE.value,
            }
        )

    # Targets are read only after policy rollout has completed for all scenarios.
    target_to_decision = {
        "ground_truth_answer": VerificationLabel.SUPPORTED.value,
        "hallucinated_answer": VerificationLabel.CONTRADICTED.value,
    }
    target_hits = 0
    assessed = 0
    abstentions = 0
    for prediction, scenario in zip(predictions, selected):
        expected = target_to_decision.get(scenario.answer_level_target)
        abstentions += int(prediction["abstained"])
        if expected is not None and prediction["prediction"] is not None:
            assessed += 1
            target_hits += int(prediction["prediction"] == expected)
    return {
        "scenario_count": len(selected),
        "prediction_count": len(predictions),
        "answer_origin_decision_accuracy": target_hits / assessed if assessed else None,
        "assessed_count": assessed,
        "abstention_count": abstentions,
        "predictions": predictions,
        "metric_note": (
            "Post-rollout answer-origin alignment only; not medical truth verification "
            "or claim-level four-class accuracy."
        ),
    }


def train_dqn(
    training_scenarios: list[MedHalluEpisode],
    validation_scenarios: list[MedHalluEpisode],
    *,
    config: TrainingConfig,
    dqn_config: DQNConfig,
    data_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect real environment transitions and optimize a DQN for each episode."""
    if not training_scenarios:
        raise ValueError("training scenarios are empty; no training transitions can be collected")
    if not validation_scenarios:
        raise ValueError("validation scenarios are empty; model selection is unavailable")
    if any(item.project_split not in (None, "train") for item in training_scenarios):
        raise ValueError("training_scenarios contains non-train project_split data")
    if any(item.project_split not in (None, "validation") for item in validation_scenarios):
        raise ValueError("validation_scenarios contains non-validation project_split data")
    if any(item.configuration != config.training_configuration for item in training_scenarios):
        raise ValueError("training scenarios do not match the configured dataset configuration")
    if any(item.configuration != config.training_configuration for item in validation_scenarios):
        raise ValueError("validation scenarios do not match the configured dataset configuration")
    training_record_ids = {item.record_id for item in training_scenarios}
    validation_record_ids = {item.record_id for item in validation_scenarios}
    overlap = training_record_ids & validation_record_ids
    if overlap:
        examples = sorted(overlap)[:3]
        raise ValueError(
            "training and validation splits share source record IDs; "
            f"refusing leakage (examples: {examples})"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.seed)
    rng = random.Random(config.seed)
    train_env = _make_env(training_scenarios, config.max_action_budget)
    agent = DQNAgent.from_env(train_env, dqn_config, seed=config.seed)
    initial_parameters = [
        parameter.detach().clone()
        for parameter in agent.policy_network.parameters()
    ]

    train_started = time.perf_counter()
    episode_logs: list[dict[str, Any]] = []
    aggregate_actions: Counter[str] = Counter()
    best_validation_accuracy = -1.0
    checkpoint_path = config.checkpoint_root / f"medhallu_{config.training_configuration}_dqn.pt"
    config.output_root.mkdir(parents=True, exist_ok=True)
    result_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_dir = config.output_root / f"{config.training_configuration}_{result_id}"
    result_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = config.checkpoint_root / (
        f"medhallu_{config.training_configuration}_{result_id}_best.pt"
    )

    for episode_index in range(config.episodes):
        scenario_index = rng.randrange(len(training_scenarios))
        scenario = training_scenarios[scenario_index]
        observation, _ = train_env.reset(
            seed=config.seed + episode_index,
            options={"scenario_index": scenario_index},
        )
        total_reward = 0.0
        losses: list[float] = []
        actions: Counter[str] = Counter()
        terminated = truncated = False
        episode_length = 0
        while not (terminated or truncated):
            current_epsilon = max(
                dqn_config.epsilon_end,
                dqn_config.epsilon_start * config.epsilon_decay**agent.environment_steps,
            )
            action = agent.select_action(observation, explore=True, epsilon=current_epsilon)
            next_observation, reward, terminated, truncated, _ = train_env.step(action)
            # Store only policy-visible values, never info, scenario, or target fields.
            agent.observe(
                Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)
            action_name = MedicalAction(action).name
            actions[action_name] += 1
            aggregate_actions[action_name] += 1
            total_reward += reward
            episode_length += 1
            observation = next_observation

        episode_logs.append(
            {
                "episode": episode_index + 1,
                "scenario_id": scenario.scenario_id,
                "reward": total_reward,
                "length": episode_length,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "epsilon": current_epsilon,
                "terminated": terminated,
                "truncated": truncated,
                "action_counts": dict(actions),
            }
        )

        # Validation is inference-only. Targets are inspected after each rollout.
        if (episode_index + 1) % max(1, min(10, config.episodes)) == 0 or episode_index + 1 == config.episodes:
            validation_result = evaluate_greedily(
                agent,
                validation_scenarios,
                max_action_budget=config.max_action_budget,
                limit=config.validation_scenario_limit,
            )
            score = validation_result["answer_origin_decision_accuracy"]
            if score is not None and score > best_validation_accuracy:
                best_validation_accuracy = score
                agent.save_checkpoint(
                    checkpoint_path,
                    extra={
                        "training_configuration": config.training_configuration,
                        "training_split": config.training_split,
                        "validation_split": config.validation_split,
                        "validation_accuracy": score,
                        "validation_metric_note": validation_result["metric_note"],
                    },
                )

    runtime_seconds = time.perf_counter() - train_started
    final_validation = evaluate_greedily(
        agent,
        validation_scenarios,
        max_action_budget=config.max_action_budget,
        limit=config.validation_scenario_limit,
    )
    # Always save the final model as well; best validation checkpoint remains separately available.
    final_checkpoint_path = checkpoint_path.with_name(
        f"medhallu_{config.training_configuration}_{result_id}_final.pt"
    )
    agent.save_checkpoint(
        final_checkpoint_path,
        extra={
            "training_configuration": config.training_configuration,
            "training_split": config.training_split,
            "validation_split": config.validation_split,
            "validation_accuracy": final_validation["answer_origin_decision_accuracy"],
            "validation_metric_note": final_validation["metric_note"],
        },
    )

    run_record: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "checkpoint_root": str(config.checkpoint_root),
        },
        "dqn_config": asdict(dqn_config),
        "seed": config.seed,
        "device": str(agent.device),
        "training_scenario_count": len(training_scenarios),
        "validation_scenario_count": len(validation_scenarios),
        "training_configuration": config.training_configuration,
        "training_split": config.training_split,
        "validation_split": config.validation_split,
        "test_split_loaded": False,
        "episode_count": len(episode_logs),
        "runtime_seconds": runtime_seconds,
        "total_environment_steps": agent.environment_steps,
        "optimization_steps": agent.optimization_steps,
        "parameters_updated": any(
            not torch.equal(initial, current.detach())
            for initial, current in zip(
                initial_parameters, agent.policy_network.parameters()
            )
        ),
        "action_frequencies": dict(aggregate_actions),
        "episode_logs": episode_logs,
        "validation": final_validation,
        "best_validation_accuracy": best_validation_accuracy,
        "best_checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
        "final_checkpoint": str(final_checkpoint_path),
        "data_provenance": data_provenance or {},
        "scientific_limitation": (
            "Answer-origin simulation reward alignment only; MedHallu targets are answer-level "
            "and do not establish claim-level medical truth verification."
        ),
    }
    metadata_path = result_dir / "run.json"
    metadata_path.write_text(
        json.dumps(run_record, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (result_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as output:
        fieldnames = ["episode", "scenario_id", "reward", "length", "mean_loss", "epsilon", "terminated", "truncated", "action_counts"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_logs:
            writer.writerow({**row, "action_counts": json.dumps(row["action_counts"], sort_keys=True)})

    run_record["result_dir"] = str(result_dir)
    return run_record