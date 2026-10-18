"""Integration tests for the Gymnasium medical-verification simulation."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from src.common.schemas import VerificationLabel
from src.rl.action_space import MedicalAction
from src.rl.environment import MedicalVerificationEnv
from src.rl.synthetic_scenarios import (
    ScenarioType,
    SyntheticScenario,
    generate_synthetic_scenarios,
)


def test_spaces_and_reset_observation() -> None:
    env = MedicalVerificationEnv()

    observation, info = env.reset(seed=12)

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 8
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["synthetic"] is True
    assert "training_label" not in info
    assert "scenario_id" not in info


def test_step_retrieve_verify_and_finalize() -> None:
    env = MedicalVerificationEnv(max_action_budget=5)
    observation, _ = env.reset(seed=2, options={"scenario_index": 0})

    observation, _, terminated, truncated, _ = env.step(MedicalAction.RETRIEVE_EVIDENCE)
    assert not terminated and not truncated
    assert env.observation_space.contains(observation)
    assert observation[5] > 0.0

    observation, _, terminated, truncated, _ = env.step(MedicalAction.VERIFY_CURRENT_CLAIM)
    assert not terminated and not truncated
    assert observation[3] > 0.0

    _, reward, terminated, truncated, info = env.step(MedicalAction.FINALIZE_SUPPORTED)
    assert terminated and not truncated
    assert info["decision"] == VerificationLabel.SUPPORTED.value
    assert reward > 0


def test_hidden_label_does_not_change_observation_or_info() -> None:
    scenario = generate_synthetic_scenarios(episodes_per_type=1, seed=3)[0]
    changed_label = replace(
        scenario,
        training_label=VerificationLabel.CONTRADICTED,
    )
    first = MedicalVerificationEnv(scenarios=[scenario])
    second = MedicalVerificationEnv(scenarios=[changed_label])

    first_observation, first_info = first.reset(seed=1)
    second_observation, second_info = second.reset(seed=1)

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info
    assert "training_label" not in first_info
    assert "scenario_id" not in first_info


def test_budget_exhaustion_truncates_without_endless_loop() -> None:
    env = MedicalVerificationEnv(max_action_budget=2)
    env.reset(seed=4, options={"scenario_index": 0})
    env.step(MedicalAction.RETRIEVE_EVIDENCE)

    _, reward, terminated, truncated, info = env.step(MedicalAction.VERIFY_CURRENT_CLAIM)

    assert not terminated
    assert truncated
    assert info["remaining_budget"] == 0
    assert reward < 0
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(MedicalAction.ABSTAIN_OR_ESCALATE)


def test_redundant_retrieval_is_reported_and_penalized() -> None:
    env = MedicalVerificationEnv(max_action_budget=4)
    env.reset(seed=4)
    _, first_reward, _, _, _ = env.step(MedicalAction.RETRIEVE_EVIDENCE)
    _, second_reward, _, _, info = env.step(MedicalAction.RETRIEVE_EVIDENCE)

    assert info["redundant"] is True
    assert second_reward < first_reward


def test_invalid_actions_fail_without_advancing_episode() -> None:
    env = MedicalVerificationEnv()
    observation, _ = env.reset(seed=4)

    with pytest.raises(ValueError, match="action must be an integer"):
        env.step(99)

    np.testing.assert_array_equal(observation, env._observation())


def test_reset_seed_reproduces_scenario_and_observation() -> None:
    env = MedicalVerificationEnv()
    observation_a, info_a = env.reset(seed=199)
    observation_b, info_b = env.reset(seed=199)

    np.testing.assert_array_equal(observation_a, observation_b)
    assert info_a == info_b


def test_synthetic_scenario_types_are_balanced_and_marked() -> None:
    scenarios = generate_synthetic_scenarios(episodes_per_type=3, seed=42)

    assert len(scenarios) == 5 * 3
    assert {scenario.scenario_type for scenario in scenarios} == set(ScenarioType)
    assert all(scenario.metadata["synthetic"] is True for scenario in scenarios)
    for scenario_type in ScenarioType:
        assert sum(item.scenario_type is scenario_type for item in scenarios) == 3
    assert all("supported" not in item.scenario_id for item in scenarios)
    assert scenarios == generate_synthetic_scenarios(episodes_per_type=3, seed=42)


def test_gymnasium_env_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(MedicalVerificationEnv(), skip_render_check=True)