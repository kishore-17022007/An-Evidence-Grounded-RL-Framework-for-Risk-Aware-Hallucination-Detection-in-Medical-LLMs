"""Unit and actual-data integration tests for the DQN implementation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.dqn_training import TrainingConfig, train_dqn
from src.experiments.medhallu_adapter import load_medhallu_scenarios
from src.rl.dqn import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    Transition,
    bellman_targets,
)
from src.rl.environment import MedicalVerificationEnv


def small_config(**overrides: object) -> DQNConfig:
    values: dict[str, object] = {
        "learning_rate": 1e-2,
        "batch_size": 2,
        "replay_capacity": 8,
        "gamma": 0.9,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.9,
        "target_update_interval": 1,
        "device": "cpu",
        "hidden_dims": (16,),
    }
    values.update(overrides)
    return DQNConfig(**values)  # type: ignore[arg-type]


def transition(index: int, *, terminated: bool = False, truncated: bool = False) -> Transition:
    state = np.full(15, index / 10, dtype=np.float32)
    next_state = np.full(15, (index + 1) / 10, dtype=np.float32)
    return Transition(state, index % 8, float(index), next_state, terminated, truncated)


def test_network_dimensions_and_input_errors() -> None:
    env = MedicalVerificationEnv()
    agent = DQNAgent.from_env(env, small_config(), seed=3)
    observation, _ = env.reset(seed=3)

    output = agent.policy_network(torch.as_tensor(observation))

    assert agent.policy_network.observation_dim == env.observation_space.shape[0]
    assert agent.policy_network.action_dim == env.action_space.n
    assert output.shape == (env.action_space.n,)
    with pytest.raises(ValueError, match="expected observation shape"):
        agent.select_action(np.zeros(14, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        agent.select_action(np.full(15, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="received 14"):
        agent.policy_network(torch.zeros(14))


def test_action_selection_is_valid_and_reproducible() -> None:
    env = MedicalVerificationEnv()
    observation, _ = env.reset(seed=4)
    first = DQNAgent.from_env(env, small_config(), seed=20)
    second = DQNAgent.from_env(env, small_config(), seed=20)

    actions_a = [first.select_action(observation, explore=True) for _ in range(8)]
    actions_b = [second.select_action(observation, explore=True) for _ in range(8)]

    assert actions_a == actions_b
    assert all(env.action_space.contains(action) for action in actions_a)


def test_masked_dqn_worker_selects_only_configured_actions(tmp_path: Path) -> None:
    env = MedicalVerificationEnv()
    worker = DQNAgent.from_env(env, small_config(), seed=31, valid_actions=(0, 2))
    observation, _ = env.reset(seed=31)

    actions = [worker.select_action(observation, explore=True, epsilon=1.0) for _ in range(20)]

    assert set(actions).issubset({0, 2})
    path = tmp_path / "masked.pt"
    worker.save_checkpoint(path)
    incompatible = DQNAgent.from_env(env, small_config(), seed=31, valid_actions=(1, 2))
    with pytest.raises(ValueError, match="valid action set"):
        incompatible.load_checkpoint(path)


def test_replay_buffer_capacity_sampling_and_errors() -> None:
    replay = ReplayBuffer(capacity=3, seed=8)
    for index in range(4):
        replay.add(transition(index))

    assert len(replay) == 3
    sample = replay.sample(2)
    assert len(sample) == 2
    assert {item.reward for item in sample}.issubset({1.0, 2.0, 3.0})
    with pytest.raises(ValueError, match="cannot sample"):
        replay.sample(4)


def test_bellman_targets_handle_terminated_and_budget_truncated() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0])
    next_q = torch.tensor([10.0, 10.0, 10.0])
    terminated = torch.tensor([False, True, False])
    truncated = torch.tensor([False, False, True])

    targets = bellman_targets(
        rewards, next_q, terminated, truncated, gamma=0.5
    )
    time_limit_targets = bellman_targets(
        rewards,
        next_q,
        terminated,
        truncated,
        gamma=0.5,
        bootstrap_on_truncation=True,
    )

    torch.testing.assert_close(targets, torch.tensor([6.0, 2.0, 3.0]))
    torch.testing.assert_close(time_limit_targets, torch.tensor([6.0, 2.0, 8.0]))


def test_validation_environment_does_not_calculate_target_rewards() -> None:
    episodes = load_medhallu_scenarios(
        Path(__file__).parents[1] / "data/processed/MedHallu/splits",
        configuration="pqa_labeled",
        split="validation",
    )[:1]
    if not episodes:
        pytest.skip("Actual MedHallu validation data is not present")
    opposite = replace(episodes[0], answer_level_target="hallucinated_answer")
    first = MedicalVerificationEnv(scenarios=episodes, training_rewards=False)
    second = MedicalVerificationEnv(scenarios=[opposite], training_rewards=False)
    first.reset(seed=5)
    second.reset(seed=5)
    decision_action = 4

    first_result = first.step(decision_action)
    second_result = second.step(decision_action)

    assert first_result[1] == second_result[1]
    assert first_result[2:4] == second_result[2:4]
    np.testing.assert_array_equal(first_result[0], second_result[0])


def test_optimization_updates_parameters_and_hard_syncs_target() -> None:
    env = MedicalVerificationEnv()
    agent = DQNAgent.from_env(env, small_config(target_update_interval=1), seed=5)
    before = [parameter.detach().clone() for parameter in agent.policy_network.parameters()]
    for index in range(2):
        agent.observe(transition(index, terminated=(index == 1)))

    loss = agent.optimize()

    assert loss is not None and np.isfinite(loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.policy_network.parameters())
    )
    for policy_parameter, target_parameter in zip(
        agent.policy_network.parameters(), agent.target_network.parameters()
    ):
        torch.testing.assert_close(policy_parameter, target_parameter)


def test_soft_target_update_moves_toward_policy() -> None:
    env = MedicalVerificationEnv()
    agent = DQNAgent.from_env(
        env,
        small_config(target_update_mode="soft", target_tau=0.5),
        seed=2,
    )
    with torch.no_grad():
        for parameter in agent.policy_network.parameters():
            parameter.add_(2.0)
    policy_parameter = next(agent.policy_network.parameters()).detach().clone()
    target_before = next(agent.target_network.parameters()).detach().clone()

    agent.optimization_steps = 1
    agent._update_target()
    target_after = next(agent.target_network.parameters()).detach()

    torch.testing.assert_close(target_after, (target_before + policy_parameter) / 2)


def test_checkpoint_round_trip_preserves_q_values(tmp_path: Path) -> None:
    env = MedicalVerificationEnv()
    first = DQNAgent.from_env(env, small_config(), seed=6)
    observation, _ = env.reset(seed=6)
    expected = first.policy_network(torch.as_tensor(observation)).detach().clone()
    checkpoint = tmp_path / "checkpoint.pt"
    first.save_checkpoint(checkpoint, extra={"run": "unit-test"})

    second = DQNAgent.from_env(env, small_config(), seed=99)
    payload = second.load_checkpoint(checkpoint)
    actual = second.policy_network(torch.as_tensor(observation)).detach()

    torch.testing.assert_close(actual, expected)
    assert payload["extra"]["run"] == "unit-test"
    with pytest.raises(ValueError, match="observation dimension"):
        incompatible = DQNAgent(14, 8, small_config(), seed=4)
        incompatible.load_checkpoint(checkpoint)


def test_empty_training_and_validation_scenarios_fail_clearly(tmp_path: Path) -> None:
    config = TrainingConfig(
        episodes=1,
        output_root=tmp_path / "results",
        checkpoint_root=tmp_path / "models",
    )
    with pytest.raises(ValueError, match="training scenarios are empty"):
        train_dqn([], [], config=config, dqn_config=small_config())


def test_train_rejects_source_record_overlap(tmp_path: Path) -> None:
    data_root = Path(__file__).parents[1] / "data/processed/MedHallu/splits"
    if not (data_root / "pqa_labeled/train.jsonl").is_file():
        pytest.skip("Actual MedHallu split files are not present")
    train_item = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="train"
    )[0]
    valid_item = replace(train_item, project_split="validation")
    config = TrainingConfig(
        episodes=1,
        output_root=tmp_path / "results",
        checkpoint_root=tmp_path / "models",
    )
    with pytest.raises(ValueError, match="share source record IDs"):
        train_dqn(
            [train_item],
            [valid_item],
            config=config,
            dqn_config=small_config(),
        )


def test_train_rejects_non_train_training_scenario(tmp_path: Path) -> None:
    data_root = Path(__file__).parents[1] / "data/processed/MedHallu/splits"
    if not (data_root / "pqa_labeled/train.jsonl").is_file():
        pytest.skip("Actual MedHallu split files are not present")
    training = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="train"
    )[0]
    validation = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="validation"
    )[0]
    mislabeled_training = replace(training, project_split="test")
    config = TrainingConfig(
        episodes=1,
        output_root=tmp_path / "results",
        checkpoint_root=tmp_path / "models",
    )

    with pytest.raises(ValueError, match="non-train project_split"):
        train_dqn(
            [mislabeled_training],
            [validation],
            config=config,
            dqn_config=small_config(),
        )


def test_real_medhallu_training_smoke_uses_train_and_validation_only(tmp_path: Path) -> None:
    data_root = Path(__file__).parents[1] / "data/processed/MedHallu/splits"
    if not (data_root / "pqa_labeled/train.jsonl").is_file():
        pytest.skip("Actual MedHallu split files are not present")
    training = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="train"
    )[:4]
    validation = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="validation"
    )[:4]
    initial_agent = DQNAgent.from_env(
        MedicalVerificationEnv(scenarios=training, max_action_budget=3),
        small_config(),
        seed=71,
    )
    initial_parameters = [p.detach().clone() for p in initial_agent.policy_network.parameters()]
    training_config = TrainingConfig(
        episodes=2,
        seed=71,
        max_action_budget=3,
        training_configuration="pqa_labeled",
        training_scenario_limit=4,
        validation_scenario_limit=4,
        output_root=tmp_path / "results",
        checkpoint_root=tmp_path / "models",
    )

    result = train_dqn(
        training,
        validation,
        config=training_config,
        dqn_config=small_config(),
        data_provenance={"test_split_loaded": False},
    )

    checkpoint = Path(result["final_checkpoint"])
    assert checkpoint.is_file()
    trained = DQNAgent.from_env(
        MedicalVerificationEnv(scenarios=training, max_action_budget=3),
        small_config(),
        seed=999,
    )
    trained.load_checkpoint(checkpoint)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, trained.policy_network.parameters())
    )
    assert result["episode_count"] == 2
    assert result["optimization_steps"] > 0
    assert result["parameters_updated"] is True
    assert result["training_split"] == "train"
    assert result["validation_split"] == "validation"
    assert result["test_split_loaded"] is False
    assert result["data_provenance"]["test_split_loaded"] is False
    assert Path(result["result_dir"], "run.json").is_file()