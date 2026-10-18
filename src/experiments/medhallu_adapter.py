"""Convert MedHallu source rows into answer-level environment scenarios.

MedHallu provides Ground Truth and Hallucinated Answer text fields, not a
four-class verification label. Targets below identify which source field the
candidate came from; this is an explicitly derived answer-level target.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.rl.synthetic_scenarios import ObservableScenario


DATASET_ID = "UTAustin-AIHealth/MedHallu"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
REQUIRED_FIELDS = (
    "Question",
    "Knowledge",
    "Ground Truth",
    "Difficulty Level",
    "Hallucinated Answer",
    "Category of Hallucination",
)


@dataclass(frozen=True, slots=True)
class MedHalluEpisode:
    """One candidate answer; hidden answer-origin target is never policy context."""

    scenario_id: str
    dataset_id: str
    configuration: str
    source_split: str
    project_split: str | None
    record_id: str
    question: str
    candidate_answer: str
    knowledge_passages: tuple[str, ...]
    original_ground_truth: str
    hallucinated_answer: str
    difficulty: str
    hallucination_category: str
    answer_level_target: str = field(repr=False)
    target_meaning: str = field(repr=False)
    signals: ObservableScenario
    metadata: Mapping[str, str | bool] = field(repr=False)
    training_label: None = field(default=None, repr=False)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _normalized_complexity(question: str, answer: str) -> float:
    word_count = len(_tokens(question)) + len(_tokens(answer))
    return min(1.0, word_count / 240.0)


def _scenario_id(configuration: str, source_split: str, record_id: str, role: str) -> str:
    source_key = f"{DATASET_ID}|{configuration}|{source_split}|{record_id}|{role}"
    return "medhallu-" + hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def scenario_from_record(
    wrapper: Mapping[str, Any], *, configuration: str | None = None
) -> list[MedHalluEpisode]:
    """Validate one preserved processed row and produce its two answer variants."""
    original = wrapper.get("original_fields", wrapper)
    missing = [name for name in REQUIRED_FIELDS if name not in original]
    if missing:
        raise ValueError(f"MedHallu record is missing required actual fields: {missing}")

    question = original["Question"]
    knowledge = original["Knowledge"]
    ground_truth = original["Ground Truth"]
    hallucinated = original["Hallucinated Answer"]
    if not isinstance(question, str) or not isinstance(ground_truth, str) or not isinstance(hallucinated, str):
        raise ValueError("Question, Ground Truth, and Hallucinated Answer must be strings")
    if not isinstance(knowledge, list) or not all(isinstance(item, str) for item in knowledge):
        raise ValueError("Knowledge must be a list of passage strings")

    config_name = configuration or wrapper.get("configuration")
    source_split = wrapper.get("source_split", "unknown")
    record_id = str(wrapper.get("record_id", ""))
    if not config_name or not record_id:
        raise ValueError("processed rows must retain configuration and record_id")

    common_metadata: dict[str, str | bool] = {
        "synthetic": False,
        "dataset_id": DATASET_ID,
        "configuration": str(config_name),
        "source_split": str(source_split),
        "record_id": record_id,
        "source_url": DATASET_URL,
        "knowledge_source_note": "Dataset Knowledge text; underlying publication identifiers are not provided here.",
    }
    variants = (
        (
            "ground_truth_answer",
            ground_truth,
            "ground_truth_answer",
            "Candidate is copied from the dataset's Ground Truth field; this is derived from field origin, not an explicit target column.",
        ),
        (
            "hallucinated_answer",
            hallucinated,
            "hallucinated_answer",
            "Candidate is copied from the dataset's Hallucinated Answer field; this is derived from field origin, not an explicit target column.",
        ),
    )
    episodes: list[MedHalluEpisode] = []
    for role, candidate, target, target_meaning in variants:
        # No evidence or answer label is present in initial numeric state.
        signals = ObservableScenario(
            claim_complexity=_normalized_complexity(question, candidate),
            initial_relevance=0.0,
            initial_coverage=0.0,
            prior_confidence=0.0,
            contradiction_signal=0.0,
            retrieval_relevance_gain=0.0,
            retrieval_coverage_gain=0.0,
            investigation_contradiction_gain=0.0,
            independent_coverage_gain=0.0,
            independent_contradiction_signal=0.0,
        )
        episodes.append(
            MedHalluEpisode(
                scenario_id=_scenario_id(str(config_name), str(source_split), record_id, role),
                dataset_id=DATASET_ID,
                configuration=str(config_name),
                source_split=str(source_split),
                project_split=(
                    str(wrapper["project_split"])
                    if wrapper.get("project_split") is not None
                    else None
                ),
                record_id=record_id,
                question=question,
                candidate_answer=candidate,
                knowledge_passages=tuple(knowledge),
                original_ground_truth=ground_truth,
                hallucinated_answer=hallucinated,
                difficulty=str(original["Difficulty Level"]),
                hallucination_category=str(original["Category of Hallucination"]),
                answer_level_target=target,
                target_meaning=target_meaning,
                signals=signals,
                metadata={**common_metadata, "candidate_role": role},
            )
        )
    return episodes


def load_medhallu_scenarios(
    data_root: Path = Path("data/processed/MedHallu"),
    *,
    configuration: str | None = None,
    split: str | None = "train",
) -> list[MedHalluEpisode]:
    """Load processed files, defaulting to train to avoid accidental test use."""
    if configuration is not None and configuration not in {"pqa_labeled", "pqa_artificial"}:
        raise ValueError(f"unsupported MedHallu configuration: {configuration}")
    configurations = [configuration] if configuration else ["pqa_labeled", "pqa_artificial"]
    scenarios: list[MedHalluEpisode] = []
    for config_name in configurations:
        config_dir = data_root / config_name
        split_paths = [config_dir / f"{split}.jsonl"] if split else sorted(config_dir.glob("*.jsonl"))
        if not split_paths or any(not path.is_file() for path in split_paths):
            raise FileNotFoundError(f"No processed MedHallu split found under {config_dir}")
        for path in split_paths:
            for wrapper in _rows_from_file(path):
                scenarios.extend(scenario_from_record(wrapper, configuration=config_name))
    return scenarios