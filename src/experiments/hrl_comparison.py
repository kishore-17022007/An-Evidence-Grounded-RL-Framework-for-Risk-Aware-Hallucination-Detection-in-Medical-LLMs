"""Paired validation-only comparison of flat DQN and options-based HRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.schemas import VerificationLabel
from src.experiments.hrl_training import evaluate_hrl
from src.experiments.medhallu_adapter import MedHalluEpisode
from src.rl.action_space import MedicalAction
from src.rl.dqn import DQNAgent, DQNConfig
from src.rl.environment import MedicalVerificationEnv
from src.rl.hrl import HierarchicalAgent, HRLConfig


def evaluate_flat_dqn(
    checkpoint: Path,
    scenarios: list[MedHalluEpisode],
    *,
    max_action_budget: int,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    """Run flat DQN predictions first, then read answer-origin targets."""
    if not scenarios:
        raise ValueError("comparison validation scenarios are empty")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("observation_dim") != 15 or payload.get("action_dim") != len(MedicalAction):
        raise ValueError("flat DQN checkpoint dimensions do not match current environment spaces")
    model_config = payload.get("config", {})
    dqn_config = DQNConfig(
        hidden_dims=tuple(model_config.get("hidden_dims", (128, 128))),
        device=device,
    )
    env = MedicalVerificationEnv(
        scenarios=scenarios,
        max_action_budget=max_action_budget,
        max_action_count=max_action_budget,
        training_rewards=False,
    )
    agent = DQNAgent.from_env(env, dqn_config, seed=seed)
    agent.load_checkpoint(checkpoint, load_optimizer=False)

    predictions: list[dict[str, Any]] = []
    returns: list[float] = []
    lengths: list[int] = []
    actions: dict[str, int] = {}
    for index, scenario in enumerate(scenarios):
        observation, _ = env.reset(
            seed=seed + index, options={"scenario_index": index}
        )
        total_reward = 0.0
        length = 0
        terminated = truncated = False
        prediction: str | None = None
        while not (terminated or truncated):
            action = agent.select_action(observation, explore=False, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            action_name = MedicalAction(action).name
            actions[action_name] = actions.get(action_name, 0) + 1
            if terminated:
                prediction = info["decision"]
        returns.append(total_reward)
        lengths.append(length)
        predictions.append(
            {
                "scenario_id": scenario.scenario_id,
                "record_id": scenario.record_id,
                "prediction": prediction,
            }
        )

    target_map = {
        "ground_truth_answer": VerificationLabel.SUPPORTED.value,
        "hallucinated_answer": VerificationLabel.CONTRADICTED.value,
    }
    hits = 0
    assessed = 0
    for prediction, scenario in zip(predictions, scenarios):
        target = target_map.get(scenario.answer_level_target)
        if target is not None and prediction["prediction"] is not None:
            assessed += 1
            hits += int(prediction["prediction"] == target)
    return {
        "scenario_count": len(scenarios),
        "mean_episode_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "answer_origin_alignment": hits / assessed if assessed else None,
        "assessed_count": assessed,
        "primitive_action_counts": actions,
        "predictions": predictions,
        "metric_note": "Post-rollout answer-origin alignment only; not medical truth verification.",
    }


def compare_on_validation(
    dqn_checkpoint: Path,
    hrl_checkpoint_dir: Path,
    scenarios: list[MedHalluEpisode],
    *,
    max_action_budget: int,
    seed: int,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate both learned systems on the exact same ordered validation episodes."""
    if any(scenario.project_split not in (None, "validation") for scenario in scenarios):
        raise ValueError("comparison input must consist only of validation scenarios")
    if len({scenario.configuration for scenario in scenarios}) > 1:
        raise ValueError("comparison scenarios must come from one configuration")

    env = MedicalVerificationEnv(
        scenarios=scenarios,
        max_action_budget=max_action_budget,
        max_action_count=max_action_budget,
        training_rewards=False,
    )
    hrl_agent = HierarchicalAgent.from_env(
        env,
        HRLConfig(device=device),
        seed=seed,
    )
    hrl_agent.load_checkpoints(hrl_checkpoint_dir, load_optimizer=False)
    dqn_result = evaluate_flat_dqn(
        dqn_checkpoint,
        scenarios,
        max_action_budget=max_action_budget,
        seed=seed,
        device=device,
    )
    hrl_result = evaluate_hrl(
        hrl_agent,
        scenarios,
        max_action_budget=max_action_budget,
        seed=seed,
    )
    return {
        "comparison": "flat_dqn_vs_options_smdp_dqn",
        "seed": seed,
        "max_action_budget": max_action_budget,
        "validation_scenario_count": len(scenarios),
        "training_or_test_splits_loaded": False,
        "same_scenarios_and_seed": True,
        "flat_dqn_checkpoint": str(dqn_checkpoint),
        "hrl_checkpoint_directory": str(hrl_checkpoint_dir),
        "flat_dqn": dqn_result,
        "hrl": hrl_result,
        "interpretation": (
            "A paired validation-only pipeline comparison. Answer-origin alignment is derived "
            "from MedHallu answer fields and is not a medical truth or clinical-performance metric. "
            "Small smoke samples do not establish algorithm superiority."
        ),
    }


def write_comparison(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")