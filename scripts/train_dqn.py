"""Train and validate a DQN on MedHallu answer-variant simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.dqn_training import TrainingConfig, train_dqn
from src.experiments.medhallu_adapter import load_medhallu_scenarios
from src.rl.dqn import DQNConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/MedHallu/splits"))
    parser.add_argument("--configuration", choices=("pqa_labeled", "pqa_artificial"), default="pqa_labeled")
    parser.add_argument("--training-split", choices=("train",), default="train")
    parser.add_argument("--validation-split", choices=("validation",), default="validation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-action-budget", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--device", default="auto", help="auto, cpu, or mps (cuda also accepted when available)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/dqn"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/dqn"))
    parser.add_argument("--training-scenario-limit", type=int)
    parser.add_argument("--validation-scenario-limit", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short real-train/validation smoke job (3 episodes, small subsets, CPU).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke:
        args.episodes = 10
        args.training_scenario_limit = args.training_scenario_limit or 8
        args.validation_scenario_limit = args.validation_scenario_limit or 8
        args.batch_size = min(args.batch_size, 2)
        args.replay_capacity = max(args.replay_capacity, args.batch_size)
        args.device = "cpu"

    try:
        training = load_medhallu_scenarios(
            args.data_dir,
            configuration=args.configuration,
            split=args.training_split,
        )
        validation = load_medhallu_scenarios(
            args.data_dir,
            configuration=args.configuration,
            split=args.validation_split,
        )
        if args.training_scenario_limit:
            training = training[: args.training_scenario_limit]
        if args.validation_scenario_limit:
            validation = validation[: args.validation_scenario_limit]
        provenance = json.loads((args.data_dir / "split_metadata.json").read_text(encoding="utf-8"))
        if provenance.get("official_split") is not False:
            raise ValueError("split metadata must identify the project split as non-official")

        training_config = TrainingConfig(
            episodes=args.episodes,
            seed=args.seed,
            max_action_budget=args.max_action_budget,
            training_configuration=args.configuration,
            training_split=args.training_split,
            validation_split=args.validation_split,
            training_scenario_limit=args.training_scenario_limit,
            validation_scenario_limit=args.validation_scenario_limit,
            epsilon_decay=args.epsilon_decay,
            output_root=args.results_dir,
            checkpoint_root=args.checkpoint_dir,
        )
        dqn_config = DQNConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_mode=args.target_update_mode,
            target_update_interval=args.target_update_interval,
            target_tau=args.target_tau,
            device=args.device,
        )
        result = train_dqn(
            training,
            validation,
            config=training_config,
            dqn_config=dqn_config,
            data_provenance={
                "dataset_id": "UTAustin-AIHealth/MedHallu",
                "configuration": args.configuration,
                "training_split": args.training_split,
                "validation_split": args.validation_split,
                "test_split_loaded": False,
                "split_seed": provenance.get("seed"),
                "split_counts_by_configuration": provenance.get(
                    "split_counts_by_configuration", {}
                ),
            },
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"DQN training failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    summary = {
        "episode_count": result["episode_count"],
        "training_scenario_count": result["training_scenario_count"],
        "validation_scenario_count": result["validation_scenario_count"],
        "training_split": result["training_split"],
        "test_split_loaded": result["test_split_loaded"],
        "device": result["device"],
        "environment_steps": result["total_environment_steps"],
        "optimization_steps": result["optimization_steps"],
        "parameters_updated": result["parameters_updated"],
        "validation": {
            key: value
            for key, value in result["validation"].items()
            if key != "predictions"
        },
        "best_checkpoint": result["best_checkpoint"],
        "final_checkpoint": result["final_checkpoint"],
        "result_dir": result["result_dir"],
        "scientific_limitation": result["scientific_limitation"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())