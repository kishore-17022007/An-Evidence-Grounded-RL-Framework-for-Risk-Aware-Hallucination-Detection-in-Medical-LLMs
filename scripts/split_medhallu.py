"""Generate reproducible grouped train/validation/test MedHallu splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _question_key(value: str) -> str:
    """Normalize punctuation/case/spacing to group exact duplicate questions."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def split_medhallu(
    input_root: Path = Path("data/processed/MedHallu"),
    output_root: Path = Path("data/processed/MedHallu/splits"),
    *,
    seed: int = 17,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Group paired variants and normalized duplicate questions before splitting.

    Configurations remain separate in output paths, while normalized duplicate
    question groups spanning configurations receive one shared split. This is
    not an official MedHallu split.
    """
    config_rows: dict[str, list[dict[str, Any]]] = {}
    question_groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for configuration in ("pqa_labeled", "pqa_artificial"):
        source_dir = input_root / configuration
        paths = sorted(source_dir.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No processed input records found under {source_dir}")
        rows = [row for path in paths for row in _read_jsonl(path)]
        config_rows[configuration] = rows
        for row in rows:
            original = row.get("original_fields", {})
            question = original.get("Question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Record {row.get('record_id')} has no usable Question")
            key = _question_key(question)
            question_groups[key].append((configuration, row))

    output_exists = output_root.exists() and any(output_root.iterdir())
    if output_exists and not allow_overwrite:
        raise FileExistsError(
            f"Split output already contains files: {output_root}; pass --allow-overwrite"
        )
    if allow_overwrite and output_exists:
        print(f"WARNING: replacing generated files under {output_root}")

    rng = random.Random(seed)
    groups = list(question_groups.items())
    rng.shuffle(groups)
    target_group_counts = [ratio * len(groups) for ratio in DEFAULT_RATIOS]
    assigned_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
        name: [] for name in SPLIT_NAMES
    }
    assigned_keys: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for group_index, (key, members) in enumerate(groups):
        if group_index < len(SPLIT_NAMES):
            split_index = group_index
        else:
            split_index = min(
                range(len(SPLIT_NAMES)),
                key=lambda index: (
                    len(assigned_keys[SPLIT_NAMES[index]]) + 1
                    - target_group_counts[index]
                )
                / max(1.0, target_group_counts[index]),
            )
        split_name = SPLIT_NAMES[split_index]
        assigned_keys[split_name].append(key)
        assigned_groups[split_name].extend(members)

    split_counts: dict[str, dict[str, int]] = {}
    used_record_ids: set[str] = set()
    for split_name, members in assigned_groups.items():
        split_counts[split_name] = {}
        for configuration in ("pqa_labeled", "pqa_artificial"):
            rows = [row for config_name, row in members if config_name == configuration]
            split_counts[split_name][configuration] = len(rows)
            destination = output_root / configuration / f"{split_name}.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as target:
                for row in rows:
                    record_id = str(row["record_id"])
                    if record_id in used_record_ids:
                        raise ValueError(f"Duplicate source record id encountered: {record_id}")
                    used_record_ids.add(record_id)
                    split_record = {**row, "project_split": split_name}
                    target.write(json.dumps(split_record, ensure_ascii=False) + "\n")

    split_by_question = {
        key: split_name
        for split_name, keys in assigned_keys.items()
        for key in keys
    }
    provenance = {
        "dataset_id": "UTAustin-AIHealth/MedHallu",
        "dataset_url": "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "ratios_requested": dict(zip(SPLIT_NAMES, DEFAULT_RATIOS)),
        "split_counts_by_configuration": split_counts,
        "grouping_strategy": "All answer variants and exact normalized duplicate questions (case/punctuation/spacing normalized), including duplicates across configs, share one split.",
        "official_split": False,
        "limitations": "Near-duplicate questions with materially different wording are not semantically deduplicated. Each configuration is written separately and dataset provenance is retained per record.",
        "question_group_hashes_by_split": {
            split_name: [
                hashlib.sha256(key.encode("utf-8")).hexdigest()
                for key in assigned_keys[split_name]
            ]
            for split_name in SPLIT_NAMES
        },
        "source_record_total": sum(map(len, config_rows.values())),
        "output_root": str(output_root),
    }
    metadata_path = output_root / "split_metadata.json"
    metadata_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/MedHallu"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/MedHallu/splits")
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        split_medhallu(
            args.input_dir,
            args.output_dir,
            seed=args.seed,
            allow_overwrite=args.allow_overwrite,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"MedHallu split generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())