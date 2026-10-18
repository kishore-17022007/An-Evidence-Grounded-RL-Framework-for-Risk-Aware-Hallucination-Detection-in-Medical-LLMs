"""Download and inspect the official MedHallu Hugging Face configurations.

Raw split JSONL preserves each dataset record as loaded. Processed JSONL adds
stable source metadata without deriving labels or changing source text.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_ID = "UTAustin-AIHealth/MedHallu"
CONFIGURATIONS = ("pqa_labeled", "pqa_artificial")
DOCUMENTED_FIELDS = (
    "Question",
    "Knowledge",
    "Ground Truth",
    "Difficulty Level",
    "Hallucinated Answer",
    "Category of Hallucination",
)


def _json_value(value: Any) -> Any:
    """Convert common Hugging Face/Arrow values to JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _label_distributions(records: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, int]]:
    distributions: dict[str, dict[str, int]] = {}
    likely_label_fields = [
        name
        for name in fields
        if any(token in name.casefold() for token in ("label", "category", "difficulty"))
    ]
    for name in likely_label_fields:
        counter = Counter(
            str(row.get(name))
            if isinstance(row.get(name), str)
            else json.dumps(_json_value(row.get(name)), sort_keys=True, ensure_ascii=True)
            for row in records
            if not _missing(row.get(name))
        )
        distributions[name] = dict(counter)
    return distributions


def _inspect_records(configuration: str, split_name: str, records: list[dict[str, Any]], features: Any) -> dict[str, Any]:
    actual_fields = list(features.keys()) if hasattr(features, "keys") else sorted(
        {key for record in records for key in record}
    )
    print(f"\n[{configuration}/{split_name}] Feature schema: {features}")
    print(f"[{configuration}/{split_name}] Record count: {len(records)}")
    print(f"[{configuration}/{split_name}] Available fields: {actual_fields}")
    if records:
        print(f"[{configuration}/{split_name}] First record:")
        print(json.dumps(_json_value(records[0]), indent=2, ensure_ascii=False, default=str))

    missing_counts = {
        name: sum(_missing(row.get(name)) for row in records)
        for name in actual_fields
    }
    distributions = _label_distributions(records, actual_fields)
    print(f"[{configuration}/{split_name}] Missing-value counts: {missing_counts}")
    print(f"[{configuration}/{split_name}] Label/category distributions: {distributions}")
    print(
        f"[{configuration}/{split_name}] Documented fields present: "
        f"{[name for name in DOCUMENTED_FIELDS if name in actual_fields]}"
    )
    return {
        "features": str(features),
        "record_count": len(records),
        "fields": actual_fields,
        "missing_value_counts": missing_counts,
        "label_category_distributions": distributions,
    }


def download_and_save(
    *,
    raw_root: Path,
    processed_root: Path,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Download both configurations and save raw and provenance-wrapped JSONL."""
    planned_paths = [
        path
        for config_name in CONFIGURATIONS
        for path in (
            raw_root / config_name,
            processed_root / config_name,
        )
    ]
    metadata_path = processed_root / "medhallu_download_metadata.json"
    planned_paths.append(metadata_path)
    existing = [path for path in planned_paths if path.exists()]
    if existing and not allow_overwrite:
        listed = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "MedHallu output paths already exist; no files were changed. "
            "Move them or pass --allow-overwrite:\n" + listed
        )

    retrieved_at = datetime.now(UTC).isoformat()
    output: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "retrieved_at_utc": retrieved_at,
        "configurations": {},
    }

    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except Exception as error:
        detail = str(error)
        if isinstance(error, ModuleNotFoundError) and error.name == "_lzma":
            detail += (
                " This Python interpreter lacks the standard-library LZMA extension; "
                "use an interpreter built with liblzma or rebuild Python with xz support."
            )
        raise RuntimeError(
            "Could not import Hugging Face datasets. Install the declared "
            f"dependency first: {type(error).__name__}: {detail}"
        ) from error

    for config_name in CONFIGURATIONS:
        print(f"\nLoading {DATASET_ID}, configuration={config_name!r} ...", flush=True)
        try:
            loaded = load_dataset(DATASET_ID, name=config_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load MedHallu configuration {config_name!r} from "
                f"Hugging Face: {type(error).__name__}: {error}"
            ) from error

        splits = loaded if hasattr(loaded, "items") else {"train": loaded}
        config_raw = raw_root / config_name
        config_processed = processed_root / config_name
        config_raw.mkdir(parents=True, exist_ok=True)
        config_processed.mkdir(parents=True, exist_ok=True)
        config_details: dict[str, Any] = {"splits": {}}
        total_records = 0

        for split_name, dataset in splits.items():
            rows = [_json_value(item) for item in dataset]
            inspection = _inspect_records(
                config_name, split_name, rows, dataset.features
            )
            total_records += len(rows)
            raw_path = config_raw / f"{split_name}.jsonl"
            processed_path = config_processed / f"{split_name}.jsonl"
            with raw_path.open("w", encoding="utf-8") as raw_file, processed_path.open(
                "w", encoding="utf-8"
            ) as processed_file:
                for index, row in enumerate(rows):
                    raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    wrapper = {
                        "dataset_id": DATASET_ID,
                        "configuration": config_name,
                        "source_split": split_name,
                        "source_record_index": index,
                        "record_id": f"{config_name}:{split_name}:{index}",
                        "original_fields": row,
                        "preprocessing": "Raw source fields preserved without target remapping.",
                    }
                    processed_file.write(
                        json.dumps(wrapper, ensure_ascii=False) + "\n"
                    )
            inspection["raw_file"] = str(raw_path)
            inspection["processed_file"] = str(processed_path)
            config_details["splits"][split_name] = inspection

        config_details["total_records"] = total_records
        output["configurations"][config_name] = config_details
        print(f"Downloaded total for {config_name}: {total_records}")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output["preprocessing"] = {
        "raw_format": "JSON Lines; original fields and values retained",
        "processed_format": "JSON Lines wrapper preserving original_fields",
        "target_mapping": None,
        "split_policy": "Source splits retained; project train/validation/test splits are generated separately.",
    }
    metadata_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nProvenance metadata saved: {metadata_path}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/MedHallu"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/MedHallu")
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing MedHallu output paths after warning in the console.",
    )
    args = parser.parse_args(argv)
    if args.allow_overwrite:
        print("WARNING: existing MedHallu raw/processed outputs may be replaced.")
    try:
        result = download_and_save(
            raw_root=args.raw_dir,
            processed_root=args.processed_dir,
            allow_overwrite=args.allow_overwrite,
        )
    except (FileExistsError, RuntimeError, OSError) as error:
        print(f"MedHallu download failed: {error}", file=sys.stderr)
        return 1

    for config_name, details in result["configurations"].items():
        print(f"Verified downloaded record count [{config_name}]: {details['total_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())