"""Train and evaluate options-based SMDP-DQN on MedHallu simulations."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.schemas import VerificationLabel
from src.experiments.dqn_training import _make_env
from src.experiments.medhallu_adapter import MedHalluEpisode
from src.rl.action_space import MedicalAction
from src.rl.dqn import DQNConfig, Transition
from src.rl.environment import MedicalVerificationEnv
from src.rl.hrl import (
    OPTION_SPECS,
    HRLConfig,
    HierarchicalAgent,
    OptionId,
    SMDPTransition,
    option_termination_reason,
    switchable_options,
)


@dataclass(frozen=True, slots=True)
class HRLTrainingConfig:
    """Run-level split and budget settings for hierarchical training."""

    episodes: int = 100
    seed: int = 17
    max_action_budget: int = 8
    configuration: str = "pqa_labeled"
    training_split: str = "train"
    validation_split: str = "validation"
    train_scenario_limit: int | None = None
    validation_scenario_limit: int | None = None
    validation_interval: int = 10
    output_root: Path = Path("results/hrl")
    checkpoint_root: Path = Path("models/hrl")

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.max_action_budget < 1:
            raise ValueError("episodes and max_action_budget must be positive")
        if self.training_split != "train" or self.validation_split != "validation":
            raise ValueError("HRL uses only project train for learning and validation for selection")
        if self.train_scenario_limit is not None and self.train_scenario_limit < 1:
            raise ValueError("train_scenario_limit must be positive")
        if self.validation_scenario_limit is not None and self.validation_scenario_limit < 1:
            raise ValueError("validation_scenario_limit must be positive")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")


def _validate_splits(
    training: list[MedHalluEpisode],
    validation: list[MedHalluEpisode],
    config: HRLTrainingConfig,
) -> None:
    if not training:
        raise ValueError("HRL training scenarios are empty")
    if not validation:
        raise ValueError("HRL validation scenarios are empty")
    if any(item.project_split not in (None, "train") for item in training):
        raise ValueError("HRL training scenarios contain non-train project_split data")
    if any(item.project_split not in (None, "validation") for item in validation):
        raise ValueError("HRL validation scenarios contain non-validation project_split data")
    if any(item.configuration != config.configuration for item in training + validation):
        raise ValueError("HRL scenarios do not match the configured dataset configuration")
    overlap = {item.record_id for item in training} & {item.record_id for item in validation}
    if overlap:
        raise ValueError(f"HRL train/validation overlap by source record ID: {sorted(overlap)[:3]}")


def evaluate_hrl(
    agent: HierarchicalAgent,
    scenarios: list[MedHalluEpisode],
    *,
    max_action_budget: int,
    limit: int | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Run target-free policy rollouts, then compute answer-origin alignment."""
    if not scenarios:
        raise ValueError("HRL evaluation scenarios are empty")
    selected = scenarios[:limit] if limit is not None else scenarios
    env = _make_env(selected, max_action_budget, training_rewards=False)
    predictions: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []
    option_counts: Counter[str] = Counter()
    option_durations: dict[str, list[int]] = defaultdict(list)
    primitive_actions: Counter[str] = Counter()

    for index, scenario in enumerate(selected):
        observation, _ = env.reset(seed=seed + index, options={"scenario_index": index})
        terminated = truncated = False
        total_reward = 0.0
        episode_length = 0
        final_decision: str | None = None
        previous_option: OptionId | None = None
        while not (terminated or truncated):
            options = switchable_options(observation, previous_option)
            if not options:
                raise RuntimeError("no initiable HRL options while environment budget remains")
            option = agent.manager.select_option(
                observation, options, explore=False, epsilon=0.0
            )
            spec = OPTION_SPECS[option]
            previous_option = option
            duration = 0
            option_counts[spec.name] += 1
            while duration < spec.max_duration and not (terminated or truncated):
                action = agent.choose_primitive(
                    option, observation, explore=False, epsilon=0.0
                )
                if action not in {int(item) for item in spec.primitive_actions}:
                    raise RuntimeError(f"worker {spec.name} emitted invalid primitive action {action}")
                previous = observation
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                episode_length += 1
                duration += 1
                primitive_actions[MedicalAction(action).name] += 1
                if terminated:
                    final_decision = info["decision"]
                reason = option_termination_reason(
                    option,
                    previous,
                    observation,
                    duration=duration,
                    terminated=terminated,
                    truncated=truncated,
                )
                if reason is not None:
                    break
            option_durations[spec.name].append(duration)
        returns.append(total_reward)
        lengths.append(episode_length)
        predictions.append(
            {
                "scenario_id": scenario.scenario_id,
                "record_id": scenario.record_id,
                "prediction": final_decision,
                "abstained": final_decision == VerificationLabel.UNVERIFIABLE.value,
            }
        )

    # The policy rollouts above have completed before any target access.
    target_map = {
        "ground_truth_answer": VerificationLabel.SUPPORTED.value,
        "hallucinated_answer": VerificationLabel.CONTRADICTED.value,
    }
    hits = 0
    assessed = 0
    for prediction, scenario in zip(predictions, selected):
        expected = target_map.get(scenario.answer_level_target)
        if expected is not None and prediction["prediction"] is not None:
            assessed += 1
            hits += int(prediction["prediction"] == expected)
    return {
        "scenario_count": len(selected),
        "mean_episode_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "answer_origin_alignment": hits / assessed if assessed else None,
        "assessed_count": assessed,
        "option_selection_counts": dict(option_counts),
        "mean_option_durations": {
            name: float(np.mean(values)) for name, values in option_durations.items()
        },
        "primitive_action_counts": dict(primitive_actions),
        "predictions": predictions,
        "metric_note": (
            "Target-free policy rollouts followed by answer-origin scoring; not medical truth "
            "verification or claim-level four-class performance."
        ),
    }


def train_hrl(
    training_scenarios: list[MedHalluEpisode],
    validation_scenarios: list[MedHalluEpisode],
    *,
    run_config: HRLTrainingConfig,
    hrl_config: HRLConfig,
    data_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train manager SMDP-DQN and independent masked DQN workers from scratch."""
    _validate_splits(training_scenarios, validation_scenarios, run_config)
    random.seed(run_config.seed)
    np.random.seed(run_config.seed)
    torch.manual_seed(run_config.seed)
    rng = random.Random(run_config.seed)
    env = _make_env(training_scenarios, run_config.max_action_budget)
    agent = HierarchicalAgent.from_env(env, hrl_config, seed=run_config.seed)
    initial_manager = [p.detach().clone() for p in agent.manager.online.parameters()]
    initial_workers = {
        option: [p.detach().clone() for p in worker.policy_network.parameters()]
        for option, worker in agent.workers.items()
    }

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_dir = run_config.output_root / f"{run_config.configuration}_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_config.checkpoint_root / f"{run_config.configuration}_{run_id}"
    episode_logs: list[dict[str, Any]] = []
    option_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    duration_samples: dict[str, list[int]] = defaultdict(list)
    worker_losses: dict[str, list[float]] = defaultdict(list)
    manager_losses: list[float] = []
    best_validation_return = -float("inf")
    started = time.perf_counter()
    epsilon_manager = hrl_config.manager_epsilon_start

    for episode_index in range(run_config.episodes):
        scenario_index = rng.randrange(len(training_scenarios))
        scenario = training_scenarios[scenario_index]
        observation, _ = env.reset(
            seed=run_config.seed + episode_index,
            options={"scenario_index": scenario_index},
        )
        terminated = truncated = False
        episode_reward = 0.0
        episode_length = 0
        episode_options: Counter[str] = Counter()
        episode_actions: Counter[str] = Counter()
        option_records: list[dict[str, Any]] = []
        previous_option: OptionId | None = None

        while not (terminated or truncated):
            valid_options = switchable_options(observation, previous_option)
            if not valid_options:
                raise RuntimeError("HRL manager has no initiable option before episode end")
            epsilon_manager = max(
                hrl_config.manager_epsilon_end,
                hrl_config.manager_epsilon_start
                * hrl_config.manager_epsilon_decay**agent.manager.environment_steps,
            )
            option = agent.manager.select_option(
                observation,
                valid_options,
                explore=True,
                epsilon=epsilon_manager,
            )
            spec = OPTION_SPECS[option]
            previous_option = option
            start_observation = observation.copy()
            discounted_option_return = 0.0
            option_duration = 0
            option_reward = 0.0
            termination_reason = "maximum_option_duration"
            episode_options[spec.name] += 1
            option_counts[spec.name] += 1

            while option_duration < spec.max_duration and not (terminated or truncated):
                worker = agent.workers[option]
                worker_epsilon = worker.epsilon()
                action = agent.choose_primitive(
                    option,
                    observation,
                    explore=True,
                    epsilon=worker_epsilon,
                )
                valid_primitives = {int(item) for item in spec.primitive_actions}
                if action not in valid_primitives:
                    raise RuntimeError(
                        f"option worker {spec.name} generated disallowed action {action}"
                    )
                previous_observation = observation.copy()
                next_observation, reward, terminated, truncated, _ = env.step(action)
                primitive_counts[MedicalAction(action).name] += 1
                episode_actions[MedicalAction(action).name] += 1
                loss = agent.train_worker_transition(
                    option,
                    Transition(
                        previous_observation,
                        action,
                        reward,
                        next_observation,
                        terminated,
                        truncated,
                    ),
                )
                if loss is not None:
                    worker_losses[spec.name].append(loss)
                discounted_option_return += (
                    hrl_config.manager_gamma**option_duration
                ) * reward
                option_reward += reward
                episode_reward += reward
                episode_length += 1
                option_duration += 1
                observation = next_observation
                termination_reason = option_termination_reason(
                    option,
                    previous_observation,
                    observation,
                    duration=option_duration,
                    terminated=terminated,
                    truncated=truncated,
                ) or "continue"
                if termination_reason != "continue":
                    break

            option_next_options = (
                switchable_options(observation, option)
                if not (terminated or truncated)
                else ()
            )
            agent.manager.replay.add(
                SMDPTransition(
                    observation=start_observation,
                    option=int(option),
                    discounted_return=discounted_option_return,
                    next_observation=observation,
                    duration=option_duration,
                    terminated=terminated,
                    truncated=truncated,
                    next_options=tuple(int(item) for item in option_next_options),
                )
            )
            agent.manager.environment_steps += option_duration
            manager_loss = agent.manager.optimize()
            if manager_loss is not None:
                manager_losses.append(manager_loss)
            duration_samples[spec.name].append(option_duration)
            option_records.append(
                {
                    "option": spec.name,
                    "duration": option_duration,
                    "discounted_return": discounted_option_return,
                    "undiscounted_return": option_reward,
                    "termination_reason": termination_reason,
                }
            )

        episode_logs.append(
            {
                "episode": episode_index + 1,
                "scenario_id": scenario.scenario_id,
                "return": episode_reward,
                "length": episode_length,
                "manager_epsilon": epsilon_manager,
                "terminated": terminated,
                "truncated": truncated,
                "option_counts": dict(episode_options),
                "primitive_action_counts": dict(episode_actions),
                "options": option_records,
            }
        )

        if (episode_index + 1) % run_config.validation_interval == 0 or episode_index + 1 == run_config.episodes:
            validation_result = evaluate_hrl(
                agent,
                validation_scenarios,
                max_action_budget=run_config.max_action_budget,
                limit=run_config.validation_scenario_limit,
                seed=run_config.seed + 100_000,
            )
            selection_score = validation_result["mean_episode_return"]
            if selection_score > best_validation_return:
                best_validation_return = selection_score
                agent.save_checkpoints(
                    checkpoint_dir / "best",
                    extra={
                        "selection_metric": "validation_mean_episode_return_target_free",
                        "selection_value": selection_score,
                        "validation_answer_origin_alignment": validation_result[
                            "answer_origin_alignment"
                        ],
                        "training_split": run_config.training_split,
                        "validation_split": run_config.validation_split,
                    },
                )

    final_validation = evaluate_hrl(
        agent,
        validation_scenarios,
        max_action_budget=run_config.max_action_budget,
        limit=run_config.validation_scenario_limit,
        seed=run_config.seed + 100_000,
    )
    agent.save_checkpoints(
        checkpoint_dir / "final",
        extra={
            "selection_metric": "final_training_episode_checkpoint",
            "validation_answer_origin_alignment": final_validation["answer_origin_alignment"],
            "training_split": run_config.training_split,
            "validation_split": run_config.validation_split,
        },
    )
    runtime = time.perf_counter() - started
    manager_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_manager, agent.manager.online.parameters())
    )
    workers_updated = {
        OPTION_SPECS[option].name: any(
            not torch.equal(before, after.detach())
            for before, after in zip(
                initial_workers[option], agent.workers[option].policy_network.parameters()
            )
        )
        for option in OptionId
    }
    run_record: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "algorithm": "options_hierarchical_smdp_dqn",
        "manager_algorithm": "SMDP Q-learning with duration-discounted targets",
        "worker_algorithm": "independent masked DQN per option, trained on environment rewards",
        "parameter_sharing": False,
        "worker_initialization": "Each worker initialized independently from seeded scratch; no flat DQN transfer.",
        "run_config": {**asdict(run_config), "output_root": str(run_config.output_root), "checkpoint_root": str(run_config.checkpoint_root)},
        "hrl_config": asdict(hrl_config),
        "seed": run_config.seed,
        "device": str(agent.manager.device),
        "training_configuration": run_config.configuration,
        "training_split": run_config.training_split,
        "validation_split": run_config.validation_split,
        "test_split_loaded": False,
        "training_scenario_count": len(training_scenarios),
        "validation_scenario_count": len(validation_scenarios),
        "episode_count": len(episode_logs),
        "runtime_seconds": runtime,
        "manager_optimization_steps": agent.manager.optimization_steps,
        "manager_parameters_updated": manager_updated,
        "worker_optimization_steps": {
            OPTION_SPECS[option].name: agent.workers[option].optimization_steps
            for option in OptionId
        },
        "worker_parameters_updated": workers_updated,
        "mean_worker_loss": {
            name: float(np.mean(losses)) for name, losses in worker_losses.items()
        },
        "mean_manager_loss": float(np.mean(manager_losses)) if manager_losses else None,
        "option_selection_frequencies": dict(option_counts),
        "mean_option_durations": {
            name: float(np.mean(values)) for name, values in duration_samples.items()
        },
        "primitive_action_frequencies": dict(primitive_counts),
        "episode_logs": episode_logs,
        "validation": final_validation,
        "best_validation_mean_return": best_validation_return,
        "best_checkpoint_selection_metric": "validation_mean_episode_return_target_free",
        "best_checkpoint_dir": (
            str(checkpoint_dir / "best")
            if (checkpoint_dir / "best" / "manager.pt").is_file()
            else None
        ),
        "final_checkpoint_dir": str(checkpoint_dir / "final"),
        "result_dir": str(result_dir),
        "data_provenance": data_provenance or {},
        "scientific_limitation": (
            "Environment-reward learning against MedHallu answer-origin targets only; "
            "not claim-level or medical-truth verification."
        ),
    }
    (result_dir / "run.json").write_text(
        json.dumps(run_record, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (result_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ["episode", "scenario_id", "return", "length", "manager_epsilon", "terminated", "truncated", "option_counts", "primitive_action_counts", "options"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in episode_logs:
            writer.writerow({
                **row,
                "option_counts": json.dumps(row["option_counts"], sort_keys=True),
                "primitive_action_counts": json.dumps(row["primitive_action_counts"], sort_keys=True),
                "options": json.dumps(row["options"], sort_keys=True),
            })
    return run_record