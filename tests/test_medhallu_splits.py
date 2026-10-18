"""Tests for reproducible question-grouped MedHallu splitting."""

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.split_medhallu import split_medhallu


def write_source_records(root: Path, configuration: str, records: list[dict[str, Any]]) -> None:
    config_dir = root / configuration
    config_dir.mkdir(parents=True)
    with (config_dir / "train.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record) + "\n")


def wrapper(configuration: str, index: int, question: str) -> dict[str, Any]:
    return {
        "dataset_id": "UTAustin-AIHealth/MedHallu",
        "configuration": configuration,
        "source_split": "train",
        "record_id": f"{configuration}:train:{index}",
        "original_fields": {"Question": question},
    }


def assignment_map(output_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for configuration in ("pqa_labeled", "pqa_artificial"):
        for split in ("train", "validation", "test"):
            path = output_root / configuration / f"{split}.jsonl"
            with path.open(encoding="utf-8") as source:
                for line in source:
                    mapping[json.loads(line)["record_id"]] = split
    return mapping


def test_split_groups_duplicates_and_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    labeled = [
        wrapper("pqa_labeled", 0, "Duplicate question?"),
        wrapper("pqa_labeled", 1, "A distinct question."),
        wrapper("pqa_labeled", 2, "Third unrelated question?"),
        wrapper("pqa_labeled", 3, "Fourth unrelated question?"),
    ]
    artificial = [
        wrapper("pqa_artificial", 0, "duplicate QUESTION !"),
        wrapper("pqa_artificial", 1, "Fifth unrelated question?"),
        wrapper("pqa_artificial", 2, "Sixth unrelated question?"),
    ]
    write_source_records(source, "pqa_labeled", labeled)
    write_source_records(source, "pqa_artificial", artificial)
    out_a = tmp_path / "split-a"
    out_b = tmp_path / "split-b"

    metadata_a = split_medhallu(source, out_a, seed=91)
    split_medhallu(source, out_b, seed=91)

    assignments = assignment_map(out_a)
    assert assignments == assignment_map(out_b)
    assert assignments["pqa_labeled:train:0"] == assignments["pqa_artificial:train:0"]
    assert metadata_a["official_split"] is False
    assert metadata_a["seed"] == 91
    assert metadata_a["split_counts_by_configuration"]["train"]["pqa_labeled"] > 0
    assert metadata_a["split_counts_by_configuration"]["train"]["pqa_artificial"] > 0


def test_split_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source_records(source, "pqa_labeled", [wrapper("pqa_labeled", 0, "Q")])
    write_source_records(source, "pqa_artificial", [wrapper("pqa_artificial", 0, "R")])
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains files"):
        split_medhallu(source, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve"