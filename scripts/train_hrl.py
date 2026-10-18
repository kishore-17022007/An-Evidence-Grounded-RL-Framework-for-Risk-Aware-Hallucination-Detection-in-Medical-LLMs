"""Train the options-based hierarchical SMDP-DQN on MedHallu train data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.hrl_training import HRLTrainingConfig, train_hrl
from src.experiments.medhallu_adapter import load_medhallu_scenarios
from src.rl.dqn import DQNConfig
from src.rl.hrl import HRLConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/MedHallu/splits"))
    parser.add_argument("--configuration", choices=("pqa_labeled", "pqa_artificial"), default="pqa_labeled")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-action-budget", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32, help="Manager batch size")
    parser.add_argument("--worker-batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--worker-replay-capacity", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-scenario-limit", type=int)
    parser.add_argument("--validation-scenario-limit", type=int)
    parser.add_argument("--results-dir", type=Path, default=Path("results/hrl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/hrl"))
    parser.add_argument("--smoke", action="store_true", help="10 episodes, 8 train/validation candidates, small batches, CPU")
    args = parser.parse_args(argv)

    if args.smoke:
        args.episodes = 10
        args.train_scenario_limit = 8
        args.validation_scenario_limit = 8
        args.batch_size = 2
        args.worker_batch_size = 2
        args.replay_capacity = max(2, args.replay_capacity)
        args.worker_replay_capacity = max(2, args.worker_replay_capacity)
        args.device = "cpu"
        args.validation_interval = 10

    try:
        training = load_medhallu_scenarios(
            args.data_dir, configuration=args.configuration, split="train"
        )
        validation = load_medhallu_scenarios(
            args.data_dir, configuration=args.configuration, split="validation"
        )
        if args.train_scenario_limit:
            training = training[: args.train_scenario_limit]
        if args.validation_scenario_limit:
            validation = validation[: args.validation_scenario_limit]
        split_metadata = json.loads((args.data_dir / "split_metadata.json").read_text(encoding="utf-8"))
        run_config = HRLTrainingConfig(
            episodes=args.episodes,
            seed=args.seed,
            max_action_budget=args.max_action_budget,
            configuration=args.configuration,
            train_scenario_limit=args.train_scenario_limit,
            validation_scenario_limit=args.validation_scenario_limit,
            validation_interval=args.validation_interval,
            output_root=args.results_dir,
            checkpoint_root=args.checkpoint_dir,
        )
        worker_config = DQNConfig(
            learning_rate=args.learning_rate,
            batch_size=args.worker_batch_size,
            replay_capacity=max(args.worker_replay_capacity, args.worker_batch_size),
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            device=args.device,
        )
        hrl_config = HRLConfig(
            manager_learning_rate=args.learning_rate,
            manager_batch_size=args.batch_size,
            manager_replay_capacity=max(args.replay_capacity, args.batch_size),
            manager_gamma=args.gamma,
            manager_epsilon_start=args.epsilon_start,
            manager_epsilon_end=args.epsilon_end,
            manager_epsilon_decay=args.epsilon_decay,
            worker=worker_config,
            device=args.device,
        )
        result = train_hrl(
            training,
            validation,
            run_config=run_config,
            hrl_config=hrl_config,
            data_provenance={
                "dataset_id": "UTAustin-AIHealth/MedHallu",
                "configuration": args.configuration,
                "training_split": "train",
                "validation_split": "validation",
                "test_split_loaded": False,
                "split_seed": split_metadata.get("seed"),
            },
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"HRL training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    summary = {
        key: result[key]
        for key in (
            "algorithm",
            "episode_count",
            "training_scenario_count",
            "validation_scenario_count",
            "training_split",
            "validation_split",
            "test_split_loaded",
            "device",
            "manager_optimization_steps",
            "manager_parameters_updated",
            "worker_optimization_steps",
            "worker_parameters_updated",
            "option_selection_frequencies",
            "mean_option_durations",
            "validation",
            "best_checkpoint_dir",
            "final_checkpoint_dir",
            "result_dir",
            "scientific_limitation",
        )
    }
    summary["validation"].pop("predictions", None)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())