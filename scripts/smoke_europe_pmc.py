"""Explicitly invoked live connectivity/schema smoke test for Europe PMC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.verification.config import VerificationConfig
from src.verification.europe_pmc import EuropePMCClient


def main() -> int:
    config = VerificationConfig(max_results_per_claim=2, max_retries=1)
    client = EuropePMCClient(config)
    try:
        records = client.search('TITLE_ABS:("aspirin" AND "cardiovascular")', max_results=2)
    except Exception as error:
        print(
            json.dumps(
                {"live_api": True, "status": "FAILED", "error_type": type(error).__name__}
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "live_api": True,
                "status": "SUCCESS" if records else "EMPTY",
                "record_count": len(records),
                "records": [
                    {
                        "title": record.title,
                        "pmid": record.pmid,
                        "pmcid": record.pmcid,
                        "doi": record.doi,
                        "publication_year": record.publication_year,
                        "abstract_present": bool(record.abstract),
                        "source_url": record.source_url,
                        "is_open_access": record.is_open_access,
                        "license": record.license,
                    }
                    for record in records
                ],
            },
            indent=2,
        )
    )
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())