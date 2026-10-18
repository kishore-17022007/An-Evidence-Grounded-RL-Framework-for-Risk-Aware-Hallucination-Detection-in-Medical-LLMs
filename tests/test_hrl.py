"""Options/SMDP HRL tests, including actual MedHallu training integration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.experiments.hrl_training import HRLTrainingConfig, train_hrl
from src.experiments.medhallu_adapter import load_medhallu_scenarios
from src.rl.action_space import MedicalAction
from src.rl.dqn import DQNConfig
from src.rl.environment import MedicalVerificationEnv
from src.rl.hrl import (
    OPTION_SPECS,
    HRLConfig,
    HierarchicalAgent,
    OptionId,
    SMDPReplayBuffer,
    SMDPTransition,
    available_options,
    option_initiable,
    option_termination_reason,
    smdp_target,
    switchable_options,
)


def hrl_config(**overrides: object) -> HRLConfig:
    worker = DQNConfig(
        learning_rate=0.01,
        batch_size=2,
        replay_capacity=16,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        target_update_interval=1,
        hidden_dims=(16,),
        device="cpu",
    )
    values: dict[str, object] = {
        "manager_learning_rate": 0.01,
        "manager_batch_size": 2,
        "manager_replay_capacity": 16,
        "manager_gamma": 0.9,
        "manager_epsilon_start": 1.0,
        "manager_epsilon_end": 0.1,
        "manager_epsilon_decay": 0.9,
        "target_update_interval": 1,
        "hidden_dims": (16,),
        "worker": worker,
        "device": "cpu",
    }
    values.update(overrides)
    return HRLConfig(**values)  # type: ignore[arg-type]


def test_option_initiation_sets_and_state_termination() -> None:
    env = MedicalVerificationEnv()
    observation, _ = env.reset(seed=3)
    assert set(available_options(observation)) == set(OptionId)
    assert option_initiable(OptionId.EVIDENCE_GATHERING, observation)
    assert option_initiable(OptionId.FINALIZATION, observation)
    assert OptionId.EVIDENCE_GATHERING not in switchable_options(
        observation, OptionId.EVIDENCE_GATHERING
    )
    assert switchable_options(observation, OptionId.FINALIZATION)

    verified = observation.copy()
    verified[9] = 0.125
    assert not option_initiable(OptionId.VERIFICATION_ASSESSMENT, verified)
    assert option_termination_reason(
        OptionId.VERIFICATION_ASSESSMENT,
        observation,
        verified,
        duration=1,
        terminated=False,
        truncated=False,
    ) == "verification_completed"


def test_option_termination_reports_budget_terminal_and_duration() -> None:
    state = np.zeros(15, dtype=np.float32)
    assert option_termination_reason(
        OptionId.EVIDENCE_GATHERING,
        state,
        state,
        duration=2,
        terminated=False,
        truncated=False,
    ) == "maximum_option_duration"
    assert option_termination_reason(
        OptionId.EVIDENCE_GATHERING,
        state,
        state,
        duration=1,
        terminated=False,
        truncated=True,
    ) == "environment_truncated_budget"
    assert option_termination_reason(
        OptionId.FINALIZATION,
        state,
        state,
        duration=1,
        terminated=True,
        truncated=False,
    ) == "environment_terminated"


def test_worker_actions_are_masked_and_manager_selects_initiable_options() -> None:
    env = MedicalVerificationEnv()
    agent = HierarchicalAgent.from_env(env, hrl_config(), seed=12)
    observation, _ = env.reset(seed=12)
    for option_id, spec in OPTION_SPECS.items():
        assert len(spec.primitive_actions) > 1
        action = agent.choose_primitive(option_id, observation, explore=True, epsilon=1.0)
        assert action in {int(item) for item in spec.primitive_actions}
    valid = available_options(observation)
    selected = agent.manager.select_option(
        observation, valid, explore=False, epsilon=0.0
    )
    assert selected in valid
    with pytest.raises(ValueError, match="empty option set"):
        agent.manager.select_option(observation, (), explore=False, epsilon=0.0)


def test_smdp_discounted_return_and_terminal_targets() -> None:
    # Two within-option rewards: 1 + gamma*2 = 2; continuation gamma^2*4 = 1.
    discounted_return = torch.tensor([2.0, 2.0])
    next_q = torch.tensor([4.0, 4.0])
    duration = torch.tensor([2.0, 2.0])
    terminated = torch.tensor([False, True])
    truncated = torch.tensor([False, False])
    targets = smdp_target(
        discounted_return,
        next_q,
        duration,
        terminated,
        truncated,
        gamma=0.5,
    )
    torch.testing.assert_close(targets, torch.tensor([3.0, 2.0]))


def test_smdp_replay_and_manager_parameters_update() -> None:
    replay = SMDPReplayBuffer(4, seed=2)
    obs = np.zeros(15, dtype=np.float32)
    next_obs = np.ones(15, dtype=np.float32)
    for index in range(2):
        replay.add(
            SMDPTransition(
                obs,
                int(OptionId.EVIDENCE_GATHERING),
                1.0,
                next_obs,
                2,
                index == 1,
                False,
                (int(OptionId.FINALIZATION),),
            )
        )
    assert len(replay.sample(2)) == 2

    agent = HierarchicalAgent.from_env(MedicalVerificationEnv(), hrl_config(), seed=4)
    before = [p.detach().clone() for p in agent.manager.online.parameters()]
    agent.manager.replay = replay
    loss = agent.manager.optimize()
    assert loss is not None and np.isfinite(loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.manager.online.parameters())
    )


def test_budget_enforces_total_option_duration() -> None:
    episodes = load_medhallu_scenarios(
        Path(__file__).parents[1] / "data/processed/MedHallu/splits",
        configuration="pqa_labeled",
        split="train",
    )[:1]
    if not episodes:
        pytest.skip("Actual MedHallu training data not present")
    env = MedicalVerificationEnv(scenarios=episodes, max_action_budget=2, max_action_count=2)
    agent = HierarchicalAgent.from_env(env, hrl_config(), seed=5)
    observation, _ = env.reset(seed=5)
    actions_taken = 0
    ended = False
    while not ended:
        options = available_options(observation)
        selected = agent.manager.select_option(observation, options, explore=True, epsilon=1.0)
        spec = OPTION_SPECS[selected]
        duration = 0
        while duration < spec.max_duration and not ended:
            action = agent.choose_primitive(selected, observation, explore=True, epsilon=1.0)
            observation, _, terminated, truncated, _ = env.step(action)
            duration += 1
            actions_taken += 1
            ended = terminated or truncated
            reason = option_termination_reason(
                selected,
                observation,
                observation,
                duration=duration,
                terminated=terminated,
                truncated=truncated,
            )
            if reason:
                break
    assert actions_taken <= 2
    assert ended


def test_hrl_checkpoints_round_trip(tmp_path: Path) -> None:
    env = MedicalVerificationEnv()
    first = HierarchicalAgent.from_env(env, hrl_config(), seed=8)
    directory = tmp_path / "checkpoint"
    first.save_checkpoints(directory, extra={"test": True})
    second = HierarchicalAgent.from_env(env, hrl_config(), seed=99)

    payload = second.load_checkpoints(directory)

    assert payload["extra"]["test"] is True
    for option in OptionId:
        expected = first.workers[option].policy_network.state_dict()
        actual = second.workers[option].policy_network.state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        assert second.workers[option].valid_actions == first.workers[option].valid_actions


def test_hrl_reproducible_action_selection_and_target_isolation() -> None:
    data_root = Path(__file__).parents[1] / "data/processed/MedHallu/splits"
    scenarios = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="train"
    )[:1]
    if not scenarios:
        pytest.skip("Actual MedHallu training data not present")
    altered = replace(scenarios[0], answer_level_target="hallucinated_answer")
    env_a = MedicalVerificationEnv(scenarios=scenarios)
    env_b = MedicalVerificationEnv(scenarios=[altered])
    state_a, info_a = env_a.reset(seed=12)
    state_b, info_b = env_b.reset(seed=12)
    np.testing.assert_array_equal(state_a, state_b)
    assert info_a == info_b

    agent_a = HierarchicalAgent.from_env(env_a, hrl_config(), seed=23)
    agent_b = HierarchicalAgent.from_env(env_b, hrl_config(), seed=23)
    options_a = available_options(state_a)
    options_b = available_options(state_b)
    option_a = agent_a.manager.select_option(state_a, options_a, explore=True, epsilon=0.8)
    option_b = agent_b.manager.select_option(state_b, options_b, explore=True, epsilon=0.8)
    action_a = agent_a.choose_primitive(option_a, state_a, explore=True, epsilon=0.8)
    action_b = agent_b.choose_primitive(option_b, state_b, explore=True, epsilon=0.8)
    assert option_a == option_b
    assert action_a == action_b


def test_real_medhallu_hrl_smoke_training_and_checkpoint(tmp_path: Path) -> None:
    data_root = Path(__file__).parents[1] / "data/processed/MedHallu/splits"
    if not (data_root / "pqa_labeled/train.jsonl").is_file():
        pytest.skip("Actual MedHallu files are not present")
    training = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="train"
    )[:8]
    validation = load_medhallu_scenarios(
        data_root, configuration="pqa_labeled", split="validation"
    )[:8]
    run_config = HRLTrainingConfig(
        episodes=10,
        seed=37,
        max_action_budget=4,
        configuration="pqa_labeled",
        train_scenario_limit=8,
        validation_scenario_limit=8,
        validation_interval=10,
        output_root=tmp_path / "results",
        checkpoint_root=tmp_path / "models",
    )
    config = hrl_config()

    result = train_hrl(
        training,
        validation,
        run_config=run_config,
        hrl_config=config,
        data_provenance={"test_split_loaded": False},
    )

    assert result["episode_count"] == 10
    assert result["training_split"] == "train"
    assert result["validation_split"] == "validation"
    assert result["test_split_loaded"] is False
    assert result["manager_parameters_updated"] is True
    assert any(result["worker_parameters_updated"].values())
    assert result["manager_optimization_steps"] > 0
    assert all(item["length"] <= run_config.max_action_budget for item in result["episode_logs"])
    for log in result["episode_logs"]:
        assert all(
            entry["duration"] <= OPTION_SPECS[OptionId[entry["option"].upper()]].max_duration
            for entry in log["options"]
        )
    loaded = HierarchicalAgent.from_env(
        MedicalVerificationEnv(scenarios=training, max_action_budget=4), config, seed=1
    )
    loaded.load_checkpoints(Path(result["final_checkpoint_dir"]))
    assert (Path(result["result_dir"]) / "run.json").is_file()