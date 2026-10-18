"""Integration checks against the actually downloaded MedHallu files."""

from dataclasses import replace
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.experiments.medhallu_adapter import (
    DATASET_ID,
    REQUIRED_FIELDS,
    load_medhallu_scenarios,
)
from src.rl.action_space import MedicalAction
from src.rl.environment import MedicalVerificationEnv


PROJECT_ROOT = Path(__file__).parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "MedHallu"
SPLIT_ROOT = PROCESSED_ROOT / "splits"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "MedHallu"
METADATA_PATH = PROCESSED_ROOT / "medhallu_download_metadata.json"


@pytest.mark.parametrize("configuration", ["pqa_labeled", "pqa_artificial"])
def test_downloaded_dataset_schema_and_record_count(
    configuration: str
) -> None:
    if not METADATA_PATH.is_file():
        pytest.skip("Real MedHallu download is not present in this workspace")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["dataset_id"] == DATASET_ID
    detail = metadata["configurations"][configuration]
    split = detail["splits"]["train"]
    assert set(split["fields"]) == set(REQUIRED_FIELDS)
    assert all(count == 0 for count in split["missing_value_counts"].values())

    raw_path = RAW_ROOT / configuration / "train.jsonl"
    with raw_path.open(encoding="utf-8") as source:
        actual_count = sum(1 for _ in source)
    assert split["record_count"] == actual_count
    assert detail["total_records"] == actual_count
    with raw_path.open(encoding="utf-8") as source:
        record = json.loads(next(source))
    assert set(record) == set(REQUIRED_FIELDS)
    assert isinstance(record["Knowledge"], list)
    assert isinstance(record["Ground Truth"], str)
    assert isinstance(record["Hallucinated Answer"], str)


def test_real_record_adapts_and_environment_reveals_knowledge_on_retrieval() -> None:
    if not (PROCESSED_ROOT / "pqa_labeled" / "train.jsonl").is_file():
        pytest.skip("Real MedHallu download is not present in this workspace")

    episodes = load_medhallu_scenarios(
        PROCESSED_ROOT, configuration="pqa_labeled", split="train"
    )
    episode = episodes[0]
    env = MedicalVerificationEnv(scenarios=[episode], max_action_budget=4)
    observation, info = env.reset(seed=31)

    assert info["policy_context"]["question"] == episode.question
    assert info["policy_context"]["candidate_answer"] == episode.candidate_answer
    assert "Ground Truth" not in info["policy_context"]
    assert "record_id" not in info
    assert "configuration" not in info
    assert "source_split" not in info
    assert observation[1] == 0.0
    assert observation[2] == 0.0

    next_observation, _, terminated, truncated, step_info = env.step(
        MedicalAction.RETRIEVE_EVIDENCE
    )

    assert not terminated and not truncated
    assert next_observation[2] > 0.0
    evidence = step_info["newly_retrieved_evidence"]
    assert evidence["text"] in episode.knowledge_passages
    assert evidence["source_identifier"] is None
    assert evidence["metadata"]["lexical_relevance_is_not_verification"] is True
    assert "answer_level_target" not in step_info
    assert "Ground Truth" not in step_info


def test_real_answer_target_only_changes_terminal_reward_not_observation() -> None:
    if not (PROCESSED_ROOT / "pqa_labeled" / "train.jsonl").is_file():
        pytest.skip("Real MedHallu download is not present in this workspace")

    episode = load_medhallu_scenarios(
        PROCESSED_ROOT, configuration="pqa_labeled", split="train"
    )[0]
    changed_target = replace(episode, answer_level_target="hallucinated_answer")
    first_env = MedicalVerificationEnv(scenarios=[episode])
    second_env = MedicalVerificationEnv(scenarios=[changed_target])
    first_observation, first_info = first_env.reset(seed=8)
    second_observation, second_info = second_env.reset(seed=8)

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info
    assert "answer_level_target" not in first_info

    first_after_retrieval, _, _, _, first_step_info = first_env.step(
        MedicalAction.RETRIEVE_EVIDENCE
    )
    second_after_retrieval, _, _, _, second_step_info = second_env.step(
        MedicalAction.RETRIEVE_EVIDENCE
    )
    np.testing.assert_array_equal(first_after_retrieval, second_after_retrieval)
    assert first_step_info == second_step_info


def test_actual_split_files_keep_duplicate_questions_together() -> None:
    if not (SPLIT_ROOT / "split_metadata.json").is_file():
        pytest.skip("Generated MedHallu splits are not present in this workspace")

    group_to_split: dict[str, str] = {}
    for configuration in ("pqa_labeled", "pqa_artificial"):
        for split_name in ("train", "validation", "test"):
            path = SPLIT_ROOT / configuration / f"{split_name}.jsonl"
            with path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    question = row["original_fields"]["Question"]
                    normalized = " ".join(re.findall(r"[a-z0-9]+", question.casefold()))
                    previous_split = group_to_split.setdefault(normalized, split_name)
                    assert previous_split == split_name
                    assert row["project_split"] == split_name