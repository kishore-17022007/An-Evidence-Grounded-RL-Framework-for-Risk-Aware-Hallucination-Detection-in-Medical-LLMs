"""Verify one candidate answer against Europe PMC abstracts or an explicit offline fixture."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging_config import configure_logging
from src.verification.config import VerificationConfig
from src.verification.europe_pmc import EuropePMCClient
from src.verification.models import VerificationRequest
from src.verification.workflow import VerificationWorkflow


OFFLINE_FIXTURE = {
    "resultList": {
        "result": [
            {
                "source": "OFFLINE_FIXTURE",
                "id": "fixture-001",
                "title": "Synthetic offline fixture (not a publication)",
                "authorString": "",
                "pubYear": "2024",
                "abstractText": "OFFLINE TEST FIXTURE ONLY: evidence content is intentionally not biomedical data.",
                "isOpenAccess": "N",
            }
        ]
    }
}


class OfflineFixtureClient:
    """Explicitly synthetic response for offline CLI demonstration only."""

    api_name = "OFFLINE_SYNTHETIC_FIXTURE"
    live_api = False

    def __init__(self, config: VerificationConfig) -> None:
        self.config = config

    def search(self, query: str, *, max_results: int | None = None):
        from src.verification.europe_pmc import parse_search_response

        return parse_search_response(OFFLINE_FIXTURE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--context")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Send sanitized query terms to Europe PMC over HTTPS.")
    mode.add_argument("--offline", action="store_true", help="Use a clearly marked non-biomedical local fixture.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--max-results", type=int, default=8)
    args = parser.parse_args(argv)
    configure_logging()
    env_config = VerificationConfig.from_env()
    config = replace(
        env_config,
        max_results_per_claim=args.max_results,
        audit_path=args.audit_path or env_config.audit_path,
    )
    client = OfflineFixtureClient(config) if args.offline else EuropePMCClient(config)
    workflow = VerificationWorkflow(config, literature_client=client)
    try:
        report = workflow.verify(
            VerificationRequest(args.question, args.answer, args.context)
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Verification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    payload = report.to_dict()
    if args.offline:
        payload["limitations"] = [
            *payload["limitations"],
            "OFFLINE MODE: this run used a synthetic fixture, not biomedical evidence.",
        ]
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())