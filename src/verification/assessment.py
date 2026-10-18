"""Conservative, transparent lexical baseline for claim/evidence assessment."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Iterable

from src.verification.config import VerificationConfig
from src.verification.models import (
    ClaimAssessment,
    EvidenceExcerpt,
    EvidenceStance,
    EuropePMCRecord,
    ExtractedClaim,
    RetrievalStatus,
    VerificationStatus,
)


_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.DOTALL)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with", "after", "before", "during", "may", "might", "can", "could", "should",
}
_NEGATION = {"no", "not", "never", "without", "lack", "lacks", "failed", "neither"}
_SPECIALIZED = re.compile(
    r"\b(dosage|dose|prescrib|medication|should (?:take|stop|start)|diagnos|emergency|pregnan|infant|child patient)\b",
    re.IGNORECASE,
)


def content_tokens(text: str) -> set[str]:
    return {token.casefold() for token in _WORD.findall(text) if token.casefold() not in _STOP}


def rank_evidence(
    claim: ExtractedClaim, records: Iterable[EuropePMCRecord], *, limit: int
) -> tuple[EvidenceExcerpt, ...]:
    """Select verbatim abstract sentences ordered by lexical retrieval relevance.

    This ranks passages for review; it is not a support score or entailment model.
    """
    terms = content_tokens(claim.text)
    if not terms:
        return ()
    ranked: list[tuple[float, int, EvidenceExcerpt]] = []
    for record_index, record in enumerate(records):
        if not record.abstract:
            continue
        for match in _SENTENCE.finditer(record.abstract):
            sentence = match.group(0)
            left_trim = len(sentence) - len(sentence.lstrip())
            right_trim = len(sentence.rstrip())
            start = match.start() + left_trim
            end = match.start() + right_trim
            excerpt = record.abstract[start:end]
            sentence_terms = content_tokens(excerpt)
            if not sentence_terms:
                continue
            overlap = len(terms & sentence_terms)
            relevance = overlap / max(1, len(terms))
            if overlap == 0:
                continue
            evidence_id = f"epmc-{record.api_source}-{record.api_record_id}-{start}"
            ranked.append(
                (
                    relevance,
                    record_index,
                    EvidenceExcerpt(
                        evidence_id=evidence_id,
                        excerpt=excerpt,
                        source_start=start,
                        source_end=end,
                        publication=record,
                        lexical_relevance=relevance,
                    ),
                )
            )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2].source_start))
    selected: list[EvidenceExcerpt] = []
    seen_publications: set[str] = set()
    for _, _, excerpt in ranked:
        publication = excerpt.publication
        publication_key = publication.primary_identifier or (
            f"{publication.api_source}:{publication.api_record_id}"
        )
        if publication_key in seen_publications:
            continue
        seen_publications.add(publication_key)
        selected.append(excerpt)
        if len(selected) >= limit:
            break
    return tuple(selected)


def assess_claim(
    claim: ExtractedClaim,
    evidence: tuple[EvidenceExcerpt, ...],
    *,
    retrieval_status: RetrievalStatus,
    retrieval_error: str | None,
    config: VerificationConfig,
    now: datetime | None = None,
) -> ClaimAssessment:
    """Apply a conservative exact-text/polarity baseline and explicit mapping.

    Confidence is a heuristic ranking signal, not a calibrated probability.
    Similarity alone never yields SUPPORTS or CONTRADICTS.
    """
    if not claim.checkable:
        return _assessment(
            claim,
            evidence,
            EvidenceStance.UNCLEAR,
            VerificationStatus.UNVERIFIABLE,
            0.0,
            1.0,
            0.0,
            (claim.uncheckable_reason or "Claim could not be treated as factual.",),
            claim.uncheckable_reason or "The claim is not objectively checkable by this service.",
            "No biomedical query was issued for this non-checkable text.",
        )
    if _SPECIALIZED.search(claim.text):
        return _assessment(
            claim,
            evidence,
            EvidenceStance.UNCLEAR,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            0.0,
            1.0,
            0.0,
            ("Claim requires specialized or individualized clinical judgment.",),
            "The workflow does not assess personalized diagnosis, medication, dosage, emergency, pregnancy, infant, or pediatric decisions.",
            "Do not use this result to start, stop, or change treatment. Consult a qualified clinician.",
        )
    if retrieval_status is RetrievalStatus.ERROR:
        reason = "Europe PMC retrieval failed; no evidence-based decision can be made."
        if retrieval_error:
            reason += f" Retrieval error type: {retrieval_error}."
        return _assessment(
            claim,
            evidence,
            EvidenceStance.UNCLEAR,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            0.0,
            1.0,
            0.0,
            ("External literature retrieval failed.",),
            reason,
            "No retrieved literature was available to assess this claim.",
        )
    if retrieval_status in {RetrievalStatus.EMPTY, RetrievalStatus.NO_RELEVANT_EVIDENCE, RetrievalStatus.SKIPPED_BUDGET} or not evidence:
        if retrieval_status is RetrievalStatus.EMPTY:
            reason = (
                "The configured literature source returned no publication records for this query. A no-result search "
                "does not establish that the claim is false."
            )
        elif retrieval_status is RetrievalStatus.NO_RELEVANT_EVIDENCE:
            reason = (
                "The configured literature source returned publication records, but no sufficiently relevant abstract "
                "excerpt could be ranked by this baseline."
            )
        else:
            reason = "The action budget did not allow literature retrieval for this claim."
        return _assessment(
            claim,
            evidence,
            EvidenceStance.UNCLEAR,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            0.0,
            1.0,
            0.0,
            (
                "No publication records returned."
                if retrieval_status is RetrievalStatus.EMPTY
                else "Publication records returned, but no relevant abstract excerpt was identified."
                if retrieval_status is RetrievalStatus.NO_RELEVANT_EVIDENCE
                else "No relevant abstract available."
            ,),
            reason,
            "Search failure/no results cannot distinguish a false claim from an unindexed or unsearched claim.",
        )

    claim_tokens = content_tokens(claim.text)
    all_evidence_tokens = set().union(*(content_tokens(item.excerpt) for item in evidence))
    coverage = len(claim_tokens & all_evidence_tokens) / max(1, len(claim_tokens))
    exact_support = any(_normalize(claim.text) in _normalize(item.excerpt) for item in evidence)
    claim_negated = bool(content_tokens(claim.text) & _NEGATION)
    explicit_polarity_conflict = any(
        item.lexical_relevance >= 0.70
        and bool(content_tokens(item.excerpt) & _NEGATION) != claim_negated
        for item in evidence
    )
    if exact_support and explicit_polarity_conflict:
        stance = EvidenceStance.MIXED
    elif exact_support:
        stance = EvidenceStance.SUPPORTS
    elif explicit_polarity_conflict:
        stance = EvidenceStance.CONTRADICTS
    else:
        stance = EvidenceStance.UNCLEAR

    now_value = now or datetime.now(UTC)
    years = [
        max(0, now_value.year - item.publication.publication_year)
        for item in evidence
        if item.publication.publication_year is not None
    ]
    missing: list[str] = []
    if coverage < config.minimum_coverage:
        missing.append("Low lexical coverage of claim terms in retrieved abstracts.")
    if not years:
        missing.append("Publication year unavailable for recency assessment.")
    elif min(years) > config.max_recency_years:
        missing.append("All selected abstracts exceed the configured recency window.")
    if any(not item.publication.abstract for item in evidence):
        missing.append("Some search records have no abstract text.")
    if any(item.publication.is_open_access is None for item in evidence):
        missing.append("Article-specific access/license metadata is unavailable for some records.")

    current_enough = bool(years) and min(years) <= config.max_recency_years
    enough_coverage = coverage >= config.minimum_coverage
    citation_available = any(item.publication.primary_identifier for item in evidence)
    if not citation_available:
        missing.append("No PMID, PMCID, or DOI was returned; source citation cannot be verified.")
    if (
        stance is EvidenceStance.SUPPORTS
        and exact_support
        and enough_coverage
        and current_enough
        and citation_available
    ):
        status = VerificationStatus.SUPPORTED
        rationale = (
            "The exact normalized claim text appears in a retrieved abstract, with sufficient "
            "lexical coverage and at least one publication inside the configured recency window. "
            "This is a conservative string-match baseline, not semantic entailment."
        )
    elif stance is EvidenceStance.CONTRADICTS and enough_coverage and current_enough and citation_available:
        status = VerificationStatus.CONTRADICTED
        rationale = (
            "A high-overlap abstract contains an explicit polarity marker inconsistent with the "
            "claim. This is a heuristic contradiction flag requiring human review, not NLI."
        )
    elif stance is EvidenceStance.MIXED:
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        rationale = "Retrieved excerpts include both matching and polarity-conflicting text; evidence is mixed."
    else:
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        rationale = (
            "Retrieved abstracts were lexically ranked, but the transparent baseline found no "
            "sufficient exact-text or explicit-polarity pattern to support a confident decision."
        )

    confidence = _heuristic_confidence(stance, coverage, len(evidence), years, config)
    if stance is EvidenceStance.MIXED:
        confidence = min(confidence, 0.35)
    elif stance is EvidenceStance.UNCLEAR:
        confidence = min(confidence, 0.4)
    uncertainty = 1.0 - confidence
    limitation = (
        "Confidence is an uncalibrated deterministic heuristic. Abstract-level lexical rules "
        "do not establish causality, clinical applicability, study quality, or entailment."
    )
    return _assessment(
        claim,
        evidence,
        stance,
        status,
        confidence,
        uncertainty,
        coverage,
        tuple(missing),
        rationale,
        limitation,
    )


def _assessment(
    claim: ExtractedClaim,
    evidence: tuple[EvidenceExcerpt, ...],
    stance: EvidenceStance,
    status: VerificationStatus,
    confidence: float,
    uncertainty: float,
    coverage: float,
    missing: tuple[str, ...],
    rationale: str,
    limitation: str | None,
) -> ClaimAssessment:
    return ClaimAssessment(
        claim=claim,
        evidence=evidence,
        stance=stance,
        status=status,
        confidence=confidence,
        uncertainty=uncertainty,
        confidence_method="uncalibrated deterministic heuristic; not a probability",
        evidence_coverage=coverage,
        missing_information=missing,
        rationale=rationale,
        limitation=limitation,
    )


def _heuristic_confidence(
    stance: EvidenceStance,
    coverage: float,
    evidence_count: int,
    publication_ages: list[int],
    config: VerificationConfig,
) -> float:
    stance_factor = 0.65 if stance in {EvidenceStance.SUPPORTS, EvidenceStance.CONTRADICTS} else 0.25
    count_factor = min(1.0, evidence_count / 3.0)
    recency_factor = 1.0 if publication_ages and min(publication_ages) <= config.max_recency_years else 0.5
    return round(min(0.75, 0.15 + 0.45 * coverage + 0.1 * count_factor + 0.1 * recency_factor + 0.1 * stance_factor), 3)


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)