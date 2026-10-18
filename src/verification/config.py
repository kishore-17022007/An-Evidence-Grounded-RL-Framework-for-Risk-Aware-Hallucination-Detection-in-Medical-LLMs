"""Configuration for live claim verification and Europe PMC access."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.verification.errors import VerificationInputError

DEFAULT_EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """Service limits and conservative baseline thresholds."""

    europe_pmc_base_url: str = DEFAULT_EUROPE_PMC_BASE_URL
    request_timeout_seconds: float = 12.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    min_request_interval_seconds: float = 0.25
    max_results_per_claim: int = 8
    max_claims: int = 12
    action_budget: int = 24
    max_question_characters: int = 4000
    max_answer_characters: int = 12000
    max_context_characters: int = 4000
    minimum_coverage: float = 0.75
    max_recency_years: int = 12
    audit_path: Path = Path("results/verification/audit.jsonl")
    cache_size: int = 256

    def __post_init__(self) -> None:
        if not self.europe_pmc_base_url.startswith("https://"):
            raise ValueError("Europe PMC API URL must use HTTPS")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0:
            raise ValueError("retry settings must not be negative")
        if self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must not be negative")
        for name in ("max_results_per_claim", "max_claims", "action_budget", "cache_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in [0, 1]")
        if self.max_recency_years < 1:
            raise ValueError("max_recency_years must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> VerificationConfig:
        values = os.environ if environ is None else environ
        try:
            return cls(
                europe_pmc_base_url=values.get(
                    "MEDVERIFY_EPMC_BASE_URL", DEFAULT_EUROPE_PMC_BASE_URL
                ),
                request_timeout_seconds=float(values.get("MEDVERIFY_EPMC_TIMEOUT", "12")),
                max_retries=int(values.get("MEDVERIFY_EPMC_RETRIES", "2")),
                retry_backoff_seconds=float(values.get("MEDVERIFY_EPMC_BACKOFF", "0.5")),
                min_request_interval_seconds=float(values.get("MEDVERIFY_EPMC_INTERVAL", "0.25")),
                max_results_per_claim=int(values.get("MEDVERIFY_MAX_RESULTS_PER_CLAIM", "8")),
                max_claims=int(values.get("MEDVERIFY_MAX_CLAIMS", "12")),
                action_budget=int(values.get("MEDVERIFY_ACTION_BUDGET", "24")),
                max_question_characters=int(values.get("MEDVERIFY_MAX_QUESTION_CHARACTERS", "4000")),
                max_answer_characters=int(values.get("MEDVERIFY_MAX_ANSWER_CHARACTERS", "12000")),
                max_context_characters=int(values.get("MEDVERIFY_MAX_CONTEXT_CHARACTERS", "4000")),
                minimum_coverage=float(values.get("MEDVERIFY_MIN_COVERAGE", "0.75")),
                max_recency_years=int(values.get("MEDVERIFY_MAX_RECENCY_YEARS", "12")),
                audit_path=Path(values.get("MEDVERIFY_AUDIT_PATH", "results/verification/audit.jsonl")),
                cache_size=int(values.get("MEDVERIFY_EPMC_CACHE_SIZE", "256")),
            )
        except (TypeError, ValueError) as error:
            raise VerificationInputError(f"Invalid verification configuration: {error}") from error