"""Low-volume Europe PMC REST search client for metadata and abstracts."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import quote

import requests

from src.verification.config import VerificationConfig
from src.verification.errors import EvidenceRetrievalError
from src.verification.models import EuropePMCRecord


LOGGER = logging.getLogger(__name__)
API_NAME = "Europe PMC REST API"
_SEARCH_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_STOP_WORDS = {
    "about", "after", "also", "among", "because", "been", "being", "between",
    "could", "does", "during", "from", "have", "into", "more", "most", "often",
    "over", "such", "than", "that", "their", "there", "these", "they", "this",
    "those", "through", "under", "using", "very", "were", "which", "while", "with",
    "question", "candidate", "answer",
}


class ResponseLike(Protocol):
    status_code: int

    def json(self) -> Any: ...

    def raise_for_status(self) -> None: ...


class SessionLike(Protocol):
    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> ResponseLike: ...


def build_search_query(claim: str, question: str, *, max_terms: int = 10) -> str:
    """Build a transparent Europe PMC TITLE_ABS query from salient source words."""
    # Claim terms have priority; question terms fill remaining slots only.
    words = _SEARCH_WORD.findall(claim) + _SEARCH_WORD.findall(question)
    selected: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.casefold()
        if key in _STOP_WORDS or len(key) < 3 or key in seen:
            continue
        seen.add(key)
        selected.append(word)
        if len(selected) >= max_terms:
            break
    if not selected:
        raise ValueError("claim and question contain no searchable biomedical terms")
    terms = " AND ".join(f'"{word}"' for word in selected)
    return f"TITLE_ABS:({terms})"


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_search_response(payload: Any, *, retrieved_at: datetime | None = None) -> tuple[EuropePMCRecord, ...]:
    """Parse actual Europe PMC JSON core result records; ignore malformed entries."""
    if not isinstance(payload, dict):
        raise EvidenceRetrievalError("Europe PMC response must be a JSON object")
    result_list = payload.get("resultList")
    results = result_list.get("result") if isinstance(result_list, dict) else None
    if results is None:
        raise EvidenceRetrievalError("Europe PMC response omitted resultList.result")
    if not isinstance(results, list):
        raise EvidenceRetrievalError("Europe PMC resultList.result must be a list")

    timestamp = retrieved_at or datetime.now(UTC)
    records: list[EuropePMCRecord] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        source = _as_optional_str(item.get("source"))
        api_record_id = _as_optional_str(item.get("id"))
        title = _as_optional_str(item.get("title"))
        if not source or not api_record_id or not title:
            continue
        pmid = _as_optional_str(item.get("pmid"))
        pmcid = _as_optional_str(item.get("pmcid"))
        doi = _as_optional_str(item.get("doi"))
        url = _publication_url(pmid=pmid, pmcid=pmcid, doi=doi)
        raw_year = _as_optional_str(item.get("pubYear")) or _publication_year(item.get("firstPublicationDate"))
        try:
            year = int(raw_year) if raw_year else None
        except ValueError:
            year = None
        authors_raw = _as_optional_str(item.get("authorString"))
        authors = tuple(part.strip() for part in re.split(r";\s*", authors_raw) if part.strip()) if authors_raw else ()
        is_oa_raw = item.get("isOpenAccess")
        if is_oa_raw is None:
            is_open_access = None
        else:
            is_open_access = str(is_oa_raw).casefold() in {"y", "yes", "true", "1"}
        records.append(
            EuropePMCRecord(
                api_source=source,
                api_record_id=api_record_id,
                title=title,
                authors=authors,
                publication_year=year,
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                abstract=_as_optional_str(item.get("abstractText")) or "",
                journal=_as_optional_str(item.get("journalTitle")),
                is_open_access=is_open_access,
                license=_as_optional_str(item.get("license")),
                source_url=url,
                retrieved_at=timestamp,
            )
        )
    return tuple(records)


def _publication_year(value: Any) -> str | None:
    text = _as_optional_str(value)
    match = re.match(r"(\d{4})", text or "")
    return match.group(1) if match else None


def _publication_url(*, pmid: str | None, pmcid: str | None, doi: str | None) -> str | None:
    if pmid:
        return f"https://europepmc.org/article/MED/{quote(pmid)}"
    if pmcid:
        normalized = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        return f"https://europepmc.org/article/PMC/{quote(normalized)}"
    if doi:
        return f"https://doi.org/{quote(doi, safe='/') }"
    return None


class EuropePMCClient:
    """Retrying, rate-limited, in-memory cached client for Europe PMC abstracts."""

    def __init__(
        self,
        config: VerificationConfig | None = None,
        *,
        session: SessionLike | None = None,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config or VerificationConfig()
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request = 0.0
        self._rate_lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[EuropePMCRecord, ...]] = OrderedDict()

    def search(self, query: str, *, max_results: int | None = None) -> tuple[EuropePMCRecord, ...]:
        """Retrieve core metadata and abstracts from the official search endpoint."""
        if not query.strip():
            raise ValueError("Europe PMC query must not be empty")
        result_limit = self.config.max_results_per_claim if max_results is None else max_results
        if not 1 <= result_limit <= self.config.max_results_per_claim:
            raise ValueError("max_results exceeds configured Europe PMC limit")
        cache_key = hashlib.sha256(f"{query}|{result_limit}".encode()).hexdigest()
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        url = f"{self.config.europe_pmc_base_url.rstrip('/')}/search"
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": result_limit,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.get(
                    url, params=params, timeout=self.config.request_timeout_seconds
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.config.max_retries:
                        retry_after = _retry_after(response)
                        self._sleep(
                            retry_after
                            if retry_after is not None
                            else self.config.retry_backoff_seconds * (2**attempt)
                        )
                        continue
                response.raise_for_status()
                records = parse_search_response(response.json())
                unique_records = deduplicate_publications(records)
                # Process-local bounded cache only; no abstracts are persisted to disk.
                self._cache[cache_key] = unique_records
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.config.cache_size:
                    self._cache.popitem(last=False)
                LOGGER.info(
                    "Europe PMC search completed",
                    extra={"source_api": API_NAME, "result_count": len(unique_records)},
                )
                return unique_records
            except (requests.RequestException, EvidenceRetrievalError, ValueError) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    self._sleep(self.config.retry_backoff_seconds * (2**attempt))
                    continue
        # Do not include request URLs/params or user-derived queries in exception logs.
        LOGGER.warning(
            "Europe PMC retrieval failed",
            extra={"source_api": API_NAME, "error_type": type(last_error).__name__ if last_error else "Unknown"},
        )
        raise EvidenceRetrievalError(
            f"Europe PMC retrieval failed after {self.config.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__ if last_error else 'unknown error'}"
        ) from last_error


    def _wait_for_rate_limit(self) -> None:
        with self._rate_lock:
            now = self._clock()
            delay = self.config.min_request_interval_seconds - (now - self._last_request)
            if delay > 0:
                self._sleep(delay)
            self._last_request = self._clock()


def deduplicate_publications(records: tuple[EuropePMCRecord, ...] | list[EuropePMCRecord]) -> tuple[EuropePMCRecord, ...]:
    """Deduplicate on any reliable PMID/PMCID/DOI alias; keep unknowns."""
    deduplicated: list[EuropePMCRecord] = []
    seen: set[str] = set()
    for record in records:
        identifiers = {
            f"PMID:{record.pmid}".casefold() if record.pmid else None,
            (
                record.pmcid if record.pmcid.startswith("PMC") else f"PMC{record.pmcid}"
            ).casefold() if record.pmcid else None,
            f"DOI:{record.doi}".casefold() if record.doi else None,
        } - {None}
        if identifiers & seen:
            continue
        seen.update(identifiers)
        deduplicated.append(record)
    return tuple(deduplicated)


def _retry_after(response: ResponseLike) -> float | None:
    headers = getattr(response, "headers", {})
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return min(30.0, max(0.0, float(raw))) if raw is not None else None
    except (TypeError, ValueError):
        return None