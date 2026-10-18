"""Compare compatible flat-DQN and HRL checkpoints on validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.hrl_comparison import compare_on_validation, write_comparison
from src.experiments.medhallu_adapter import load_medhallu_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/MedHallu/splits"))
    parser.add_argument("--configuration", choices=("pqa_labeled", "pqa_artificial"), default="pqa_labeled")
    parser.add_argument("--dqn-checkpoint", type=Path, required=True)
    parser.add_argument("--hrl-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--max-action-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/comparisons/dqn_vs_hrl.json"))
    args = parser.parse_args(argv)
    try:
        scenarios = load_medhallu_scenarios(
            args.data_dir,
            configuration=args.configuration,
            split="validation",
        )
        if args.validation_limit:
            scenarios = scenarios[: args.validation_limit]
        report = compare_on_validation(
            args.dqn_checkpoint,
            args.hrl_checkpoint_dir,
            scenarios,
            max_action_budget=args.max_action_budget,
            seed=args.seed,
            device=args.device,
        )
        report["configuration"] = args.configuration
        write_comparison(args.output, report)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DQN/HRL comparison failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    public_report = {key: value for key, value in report.items() if key not in {"flat_dqn", "hrl"}}
    public_report["flat_dqn"] = {key: value for key, value in report["flat_dqn"].items() if key != "predictions"}
    public_report["hrl"] = {key: value for key, value in report["hrl"].items() if key != "predictions"}
    print(json.dumps(public_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())