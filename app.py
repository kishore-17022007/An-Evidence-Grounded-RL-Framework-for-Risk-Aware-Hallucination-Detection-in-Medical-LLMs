"""Streamlit dashboard for the MedVerify-RL research prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import Settings
from scripts.verify_answer import OfflineFixtureClient
from src.common.logging_config import configure_logging
from src.verification.config import VerificationConfig
from src.verification.europe_pmc import EuropePMCClient
from src.verification.models import VerificationRequest
from src.verification.policy_bridge import SimulatorPolicyBridge
from src.verification.workflow import VerificationWorkflow

ROOT = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_run_catalog() -> dict[str, list[dict[str, Any]]]:
    """Read the repo's real DQN/HRL run metadata into a simple dashboard catalog."""
    catalog: dict[str, list[dict[str, Any]]] = {"dqn": [], "hrl": []}
    for key, root in {"dqn": ROOT / "results" / "dqn", "hrl": ROOT / "results" / "hrl"}.items():
        if not root.exists():
            catalog[key] = [{"name": "Not yet evaluated", "status": "not_yet_evaluated", "path": str(root)}]
            continue
        run_files = sorted(root.glob("**/run.json"))
        if not run_files:
            catalog[key] = [{"name": "Not yet evaluated", "status": "not_yet_evaluated", "path": str(root)}]
            continue
        for run_file in run_files:
            run = json.loads(run_file.read_text(encoding="utf-8"))
            entry = {
                "name": run_file.parent.name,
                "status": "available",
                "path": str(run_file),
                "seed": run.get("seed") or run.get("run_config", {}).get("seed"),
                "episode_count": run.get("episode_count") or run.get("training_config", {}).get("episodes"),
                "training_configuration": run.get("training_configuration") or run.get("run_config", {}).get("configuration"),
                "validation_accuracy": run.get("best_validation_accuracy") or run.get("validation", {}).get("answer_origin_decision_accuracy"),
                "mean_reward": run.get("mean_reward") or run.get("mean_return") or (run.get("episode_logs", [{}])[0].get("reward") if run.get("episode_logs") else None),
                "runtime_seconds": run.get("runtime_seconds"),
                "device": run.get("device") or run.get("dqn_config", {}).get("device") or run.get("hrl_config", {}).get("device"),
            }
            catalog[key].append(entry)
    return catalog


@st.cache_data(show_spinner=False)
def load_checkpoint_status() -> dict[str, dict[str, Any]]:
    """Validate actual checkpoint compatibility using the safe SimulatorPolicyBridge."""
    status: dict[str, dict[str, Any]] = {}

    dqn_paths = sorted((ROOT / "models" / "dqn").glob("*.pt"))
    if dqn_paths:
        path = max(dqn_paths, key=lambda item: item.stat().st_mtime)
        try:
            SimulatorPolicyBridge.from_checkpoints(dqn_checkpoint=path, seed=17)
            status["dqn"] = {"status": "loaded", "path": str(path), "notes": "Compatible with the simulator 15-feature observation contract."}
        except Exception as exc:  # pragma: no cover - UI safety path
            status["dqn"] = {"status": "failed_to_load", "path": str(path), "notes": str(exc)}
    else:
        status["dqn"] = {"status": "not_found", "path": "models/dqn", "notes": "No DQN checkpoint present."}

    hrl_dirs = sorted((ROOT / "models" / "hrl").glob("*"))
    final_dirs = [item for item in hrl_dirs if item.is_dir() and (item / "final").exists()]
    if final_dirs:
        directory = max(final_dirs, key=lambda item: item.stat().st_mtime)
        try:
            SimulatorPolicyBridge.from_checkpoints(hrl_checkpoint_dir=directory / "final", seed=17)
            status["hrl"] = {"status": "loaded", "path": str(directory / "final"), "notes": "Compatible with the simulator 15-feature observation contract."}
        except Exception as exc:  # pragma: no cover - UI safety path
            status["hrl"] = {"status": "failed_to_load", "path": str(directory / "final"), "notes": str(exc)}
    else:
        status["hrl"] = {"status": "not_found", "path": "models/hrl", "notes": "No HRL checkpoint directory present."}
    return status


def run_verification_request(
    *,
    question: str,
    candidate_answer: str,
    mode: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Run the existing verification workflow and return JSON-friendly UI data."""
    config = VerificationConfig.from_env()
    if mode == "offline":
        client = OfflineFixtureClient(config)
    elif mode == "live":
        client = EuropePMCClient(config)
    else:
        raise ValueError("mode must be 'offline' or 'live'")

    workflow = VerificationWorkflow(config, literature_client=client)
    report = workflow.verify(VerificationRequest(question, candidate_answer, context))
    payload = report.to_dict()
    claim = payload["claims"][0] if payload.get("claims") else {}
    status = payload.get("overall_status", "INSUFFICIENT_EVIDENCE")
    stance = claim.get("stance", "UNCLEAR")
    rationale = claim.get('rationale', 'No rationale recorded.') if claim else 'No claim extracted.'
    summary = f"WARNING: {status} / {stance}. {rationale}"
    return {
        "status": status,
        "stance": stance,
        "claim_count": len(payload.get("claims", [])),
        "confidence": float(claim.get("confidence", 0.0)),
        "summary": summary,
        "report": payload,
        "warnings": payload.get("limitations", []),
    }


def render_header() -> None:
    st.set_page_config(page_title="MedVerify-RL", page_icon="M", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #0b1020; color: #edf2ff; }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
        .stSidebar { background: #111827; }
        .metric-container { background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.2); border-radius: 10px; padding: 0.8rem; }
        div[data-testid="stMetricLabel"] { color: #a5b4fc; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("MedVerify-RL")
    st.sidebar.caption("Research prototype · not clinical decision support")


def render_overview() -> None:
    settings = Settings.from_env()
    st.title("Overview")
    st.caption("Evidence-linked medical claim verification with simulator RL and Europe PMC checks.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Offline mode", str(settings.offline_mode))
    with col_b:
        st.metric("DQN runs", len(load_run_catalog()["dqn"]))
    with col_c:
        st.metric("HRL runs", len(load_run_catalog()["hrl"]))

    st.subheader("Project architecture")
    st.markdown(
        """
        - Verification service: deterministic answer validation, exact-source evidence, Europe PMC integration, JSONL audit ledger.
        - RL simulator: structured MedHallu tasks inside the project Gymnasium environment.
        - DQN/HRL: checkpoint-safe simulator policy loading, not invoked for live evidence.
        - Dashboard: interactive project overview, verification workflow, evidence explorer, and research summary.
        """
    )

    st.subheader("Repository facts")
    run_catalog = load_run_catalog()
    dqn_rows = run_catalog["dqn"]
    hrl_rows = run_catalog["hrl"]
    dqn_experiment_count = sum(1 for row in dqn_rows if row.get("status") == "available")
    hrl_experiment_count = sum(1 for row in hrl_rows if row.get("status") == "available")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"DQN experiment runs: {dqn_experiment_count}")
        st.write(f"HRL experiment runs: {hrl_experiment_count}")
        st.write(f"Checkpoint status: {load_checkpoint_status()['dqn']['status']} / {load_checkpoint_status()['hrl']['status']}")
    with col2:
        st.write("Main workflow files: app.py, src/verification/workflow.py, scripts/verify_answer.py, src/rl/dqn.py, src/rl/hrl.py")
        st.write("Data and splits: data/processed/MedHallu, data/raw/MedHallu, results/dqn, results/hrl")

    st.subheader("Validated and unvalidated components")
    st.info(
        "Validated in the current workspace: verification service, offline CLI, checkpoint compatibility checks, and the full existing pytest suite. "
        "Experimental RL training artifacts are present, but they remain simulator-only and are not clinical evidence."
    )

    st.subheader("Main workflows")
    for label, path in {
        "Verification workflow": "src/verification/workflow.py",
        "Offline/live verification CLI": "scripts/verify_answer.py",
        "Europe PMC client": "src/verification/europe_pmc.py",
        "DQN policy": "src/rl/dqn.py",
        "HRL policy": "src/rl/hrl.py",
    }.items():
        st.markdown(f"- **{label}:** `{path}`")


def render_live_verification() -> None:
    st.title("Live Verification")
    if "verification_result" not in st.session_state:
        st.session_state.verification_result = None

    with st.form("verification_form"):
        question = st.text_area("Medical question", value="Does aspirin reduce cardiovascular events in adults?", height=120)
        candidate_answer = st.text_area("Candidate answer", value="Aspirin reduces cardiovascular events in some adults.", height=120)
        context = st.text_area("Optional context (not sent to the API or audit)", value="", height=80)
        mode = st.radio("Verification mode", ["offline", "live"], horizontal=True)
        submitted = st.form_submit_button("Run verification")

    if submitted:
        try:
            result = run_verification_request(
                question=question,
                candidate_answer=candidate_answer,
                mode=mode,
                context=context or None,
            )
            st.session_state.verification_result = result
        except Exception as exc:  # pragma: no cover - UI safety path
            st.error(f"Verification failed: {exc}")
            st.session_state.verification_result = {"status": "ERROR", "summary": str(exc), "report": {"claims": []}}

    result = st.session_state.verification_result
    if result is None:
        st.info("Submit a question and answer to run the existing verification workflow.")
        return

    st.subheader("Result")
    st.write(result["summary"])
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Status", result.get("status", "UNKNOWN"))
    with metric_cols[1]:
        st.metric("Stance", result.get("stance", "UNCLEAR"))
    with metric_cols[2]:
        st.metric("Claims", result.get("claim_count", 0))
    with metric_cols[3]:
        st.metric("Confidence", round(float(result.get("confidence", 0.0)), 3))

    if result.get("warnings"):
        for warning in result["warnings"]:
            st.warning(warning)

    report = result.get("report", {})
    claims = report.get("claims", [])
    if not claims:
        st.info("No claim assessment was produced.")
        return
    for index, claim in enumerate(claims, start=1):
        with st.expander(f"Claim {index}: {claim['claim']['text']}", expanded=True):
            st.write(f"Status: {claim['status']}")
            st.write(f"Stance: {claim['stance']}")
            st.write(f"Rationale: {claim['rationale']}")
            st.write(f"Coverage: {claim['evidence_coverage']}")
            st.write(f"Uncertainty: {claim['uncertainty']}")
            if claim.get("missing_information"):
                st.write("Missing information:")
                for item in claim["missing_information"]:
                    st.write(f"- {item}")
            if claim.get("evidence"):
                for evidence in claim["evidence"]:
                    publication = evidence.get("publication", {})
                    st.markdown(f"**Excerpt:** {evidence.get('excerpt')}")
                    st.write(f"Citation: {evidence.get('citation_identifier', 'not available')}")
                    if publication.get("source_url"):
                        st.markdown(f"Source: [{publication['source_url']}]({publication['source_url']})")
                    st.caption(f"Publication: {publication.get('title', 'Unknown title')}")
            else:
                st.caption("No supporting or contradictory evidence excerpt was ranked for this claim.")

    if report:
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="medverify_report.json",
            mime="application/json",
        )


def render_evidence_explorer() -> None:
    st.title("Evidence Explorer")
    result = st.session_state.get("verification_result")
    if not result or not result.get("report", {}).get("claims"):
        st.info("Run a verification to populate the evidence explorer.")
        return

    report = result["report"]
    for claim_index, claim in enumerate(report["claims"], start=1):
        st.subheader(f"Claim {claim_index}: {claim['claim']['text']}")
        if not claim.get("evidence"):
            st.caption("No retrieved abstracts survived ranking for this claim.")
            continue
        for evidence in claim["evidence"]:
            publication = evidence.get("publication", {})
            source_url = publication.get("source_url")
            title = publication.get("title") or "Unknown publication"
            excerpt = evidence.get("excerpt")
            identifier = evidence.get("citation_identifier") or "No/public identifier returned"
            st.markdown(f"### {title}")
            st.write(f"Identifier(s): {identifier}")
            if source_url:
                st.markdown(f"Source URL: [{source_url}]({source_url})")
            st.write(f"Excerpt: \"{excerpt}\"")
            st.caption(f"Query provenance: {claim['claim']['text']}")
            st.caption("Retrieved source text is shown separately from any generated explanation.")


def render_agent_control_center() -> None:
    st.title("Agent Control Center")
    result = st.session_state.get("verification_result")
    if not result or not result.get("report", {}).get("events"):
        st.info("Run a verification to load the workflow event ledger.")
        return

    report = result["report"]
    events = report.get("events", [])
    for event in events:
        status = event.get("assessment_status") or event.get("retrieval_status") or "—"
        with st.expander(f"{event.get('event_type')} · {status}"):
            st.write(f"Run ID: {event.get('run_id')}")
            st.write(f"Source API: {event.get('source_api')}")
            st.write(f"Claim ID: {event.get('claim_id')}")
            st.write(f"Selected action: {event.get('selected_action')}")
            st.write(f"Tool outcome: {event.get('tool_outcome')}")
            st.write(f"Rationale: {event.get('rationale')}")

    st.subheader("Action budget summary")
    st.write(f"Actions used: {report.get('actions_used')} / {report.get('action_budget')}")
    st.write(f"Stopping reason: {report.get('stopped_reason')}")


def render_rl_lab() -> None:
    st.title("RL Laboratory")
    checkpoint_status = load_checkpoint_status()
    catalog = load_run_catalog()

    st.subheader("Checkpoint compatibility")
    dqn_status = checkpoint_status.get("dqn", {})
    hrl_status = checkpoint_status.get("hrl", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("DQN policy", dqn_status.get("status", "not_found"))
        st.write(dqn_status.get("path"))
    with c2:
        st.metric("HRL policy", hrl_status.get("status", "not_found"))
        st.write(hrl_status.get("path"))

    st.subheader("Available training runs")
    dqn_frame = pd.DataFrame(catalog.get("dqn", []))
    hrl_frame = pd.DataFrame(catalog.get("hrl", []))
    if dqn_frame.empty:
        st.caption("DQN runs: Not yet evaluated")
    else:
        st.dataframe(dqn_frame[["name", "episode_count", "seed", "validation_accuracy", "runtime_seconds"]], use_container_width=True)
    if hrl_frame.empty:
        st.caption("HRL runs: Not yet evaluated")
    else:
        st.dataframe(hrl_frame[["name", "episode_count", "seed", "validation_accuracy", "runtime_seconds"]], use_container_width=True)

    st.subheader("Actual experiment metrics")
    for run_group, label in ((catalog.get("dqn", []), "DQN"), (catalog.get("hrl", []), "HRL")):
        if not run_group:
            st.caption(f"{label}: Not yet evaluated")
            continue
        selected = run_group[0]
        run_data_path = ROOT / selected["path"] if "path" in selected else None
        if run_data_path and run_data_path.exists():
            data = json.loads(run_data_path.read_text(encoding="utf-8"))
            st.write(f"**{label} latest run**: {selected['name']}")
            relevant = {
                "seed": data.get("seed") or data.get("run_config", {}).get("seed"),
                "episode_count": data.get("episode_count") or data.get("run_config", {}).get("episodes"),
                "mean_option_duration": data.get("mean_option_durations"),
                "primitive_action_frequencies": data.get("primitive_action_frequencies"),
                "validation_accuracy": data.get("best_validation_accuracy") or data.get("validation", {}).get("answer_origin_decision_accuracy"),
            }
            st.json(relevant)


def render_research_report() -> None:
    st.title("Research Report")
    st.subheader("Actual experiment summary")
    catalog = load_run_catalog()
    metrics: list[dict[str, Any]] = []
    for group_name, runs in catalog.items():
        for run in runs:
            if run.get("status") == "available":
                path = ROOT / run["path"]
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    metrics.append({
                        "group": group_name.upper(),
                        "run": run["name"],
                        "seed": payload.get("seed") or payload.get("run_config", {}).get("seed"),
                        "episode_count": payload.get("episode_count") or payload.get("run_config", {}).get("episodes"),
                        "validation_accuracy": payload.get("best_validation_accuracy") or payload.get("validation", {}).get("answer_origin_decision_accuracy"),
                        "runtime_seconds": payload.get("runtime_seconds"),
                    })
    if not metrics:
        st.caption("Not yet evaluated: no real DQN or HRL runs are present.")
        return
    report_df = pd.DataFrame(metrics)
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Limitations and reproducibility")
    st.markdown(
        """
        - These runs are simulator experiments on the project MedHallu task, not clinical validation.
        - The verification service is conservative and will abstain when evidence is weak, mixed, or not directly grounded.
        - Checkpoint compatibility uses the actual 15-feature simulator contract; live evidence never gets passed into a simulator observation vector.
        - Seed, split, and configuration metadata are recorded in the run JSON; the test split is intentionally not used for training.
        """
    )

    export_data = json.dumps({"summary": metrics}, indent=2)
    st.download_button("Export experiment summary JSON", export_data, file_name="medverify_research_summary.json", mime="application/json")


def main() -> None:
    render_header()
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    pages = {
        "Overview": render_overview,
        "Live Verification": render_live_verification,
        "Evidence Explorer": render_evidence_explorer,
        "Agent Control Center": render_agent_control_center,
        "RL Laboratory": render_rl_lab,
        "Research Report": render_research_report,
    }

    st.sidebar.title("Navigation")
    selected = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selected]()


if __name__ == "__main__":
    main()