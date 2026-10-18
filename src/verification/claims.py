"""Deterministic sentence claim extraction with exact source-text offsets."""

from __future__ import annotations

import hashlib
import re

from src.verification.models import ExtractedClaim


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_NON_FACTUAL_PREFIXES = (
    "in my opinion",
    "i think",
    "i feel",
    "i believe",
)


def extract_claims(answer: str, *, run_id: str) -> tuple[ExtractedClaim, ...]:
    """Segment answer sentences and preserve exact corresponding source spans.

    This is deterministic sentence segmentation, not a learned claim extractor.
    Every emitted claim's text is a substring of the original answer.
    """
    if not isinstance(answer, str) or not answer.strip():
        return ()
    segments: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(answer):
        end = boundary.start()
        segments.append((start, end))
        start = boundary.end()
    segments.append((start, len(answer)))

    claims: list[ExtractedClaim] = []
    for segment_start, segment_end in segments:
        while segment_start < segment_end and answer[segment_start].isspace():
            segment_start += 1
        while segment_end > segment_start and answer[segment_end - 1].isspace():
            segment_end -= 1
        text = answer[segment_start:segment_end]
        if not text:
            continue
        words = _WORD.findall(text)
        normalized = text.casefold()
        checkable = len(words) >= 4 and not normalized.startswith(_NON_FACTUAL_PREFIXES)
        reason = None
        if len(words) < 4:
            reason = "Too little proposition text for this baseline to construct a biomedical query."
        elif not checkable:
            reason = "First-person opinion is not an objectively checkable claim under this baseline."
        claim_id = hashlib.sha256(
            f"{run_id}:{segment_start}:{segment_end}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        claims.append(
            ExtractedClaim(
                claim_id=f"claim-{claim_id}",
                text=text,
                source_start=segment_start,
                source_end=segment_end,
                checkable=checkable,
                uncheckable_reason=reason,
            )
        )
    return tuple(claims)


def normalize_search_text(text: str) -> str:
    """Normalize whitespace for outbound queries without changing source offsets."""
    return re.sub(r"\s+", " ", text).strip()