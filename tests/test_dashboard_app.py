from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import load_checkpoint_status, load_run_catalog, run_verification_request

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def real_run_catalog() -> dict:
    return load_run_catalog()


def test_load_run_catalog_reads_real_experiment_results() -> None:
    catalog = load_run_catalog()

    assert catalog["dqn"]
    assert catalog["hrl"]
    assert catalog["dqn"][0]["status"] in {"available", "not_yet_evaluated"}
    assert catalog["hrl"][0]["status"] in {"available", "not_yet_evaluated"}


def test_policy_bridge_status_uses_existing_checkpoints() -> None:
    status = load_checkpoint_status()

    assert status["dqn"]["status"] == "loaded"
    assert status["hrl"]["status"] == "loaded"
    assert status["dqn"]["path"].endswith(".pt")
    assert status["hrl"]["path"].endswith("final")


def test_offline_verification_request_uses_service_and_abstains() -> None:
    result = run_verification_request(
        question="Does compound X affect outcome Y?",
        candidate_answer="Compound X changes outcome Y in adult patients.",
        mode="offline",
    )

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["stance"] == "UNCLEAR"
    assert result["claim_count"] == 1
    assert "warning" in result["summary"].lower()


def test_app_renders_dashboard_and_sidebar() -> None:
    at = AppTest.from_file(str(APP_PATH))
    at.run()

    radio = at.radio[0]
    assert radio.label == "Go to"
    assert radio.options == [
        "Overview",
        "Live Verification",
        "Evidence Explorer",
        "Agent Control Center",
        "RL Laboratory",
        "Research Report",
    ]
    assert at.title[0].value == "Overview"


def test_dashboard_live_form_runs_verification() -> None:
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    at.radio[0].set_value("Live Verification")
    at.run()
    at.text_area[0].input("Does aspirin reduce cardiovascular events in adults?")
    at.text_area[1].input("Aspirin reduces cardiovascular events in some adults.")
    at.radio[0].set_value("offline")
    at.form_submit_button[0].click()
    at.run()

    assert len(at.markdown) >= 1
    assert any("WARNING" in str(item.value) for item in at.markdown) or any("INSUFFICIENT_EVIDENCE" in str(item.value) for item in at.markdown)
