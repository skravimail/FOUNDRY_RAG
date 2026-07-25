"""T²-RAGBench scoring: Number Match (ε=1e−2), MRR@k, R@k."""

from __future__ import annotations

import re
from typing import Iterable

# Relative tolerance from T²-RAGBench / paper contract.
DEFAULT_EPS = 1e-2

# Numbers with optional thousands separators, decimals, scientific notation, %.
_NUM_RE = re.compile(
    r"(?<![\w./])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(%)?"
)


def parse_floats(text: str) -> list[float]:
    """Extract numeric candidates from free-form model text.

    Percentages are converted to fractions (``9.3%`` → ``0.093``).
    """
    if not text:
        return []
    values: list[float] = []
    for match in _NUM_RE.finditer(text):
        raw, pct = match.group(1), match.group(2)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if pct:
            value /= 100.0
        values.append(value)
    return values


def number_match(
    prediction: str | float | None,
    gold: str | float,
    *,
    eps: float = DEFAULT_EPS,
) -> bool:
    """True if any parsed prediction number matches ``gold`` within relative ``eps``."""
    try:
        gold_f = float(gold)
    except (TypeError, ValueError):
        return False

    if prediction is None:
        return False
    if isinstance(prediction, (int, float)):
        candidates = [float(prediction)]
    else:
        candidates = parse_floats(str(prediction))
    if not candidates:
        return False

    denom = abs(gold_f) if abs(gold_f) > 1e-12 else 1.0
    for pred in candidates:
        if abs(pred - gold_f) / denom <= eps:
            return True
        # Also accept percent-form when gold is a fraction (or vice versa).
        if abs(pred / 100.0 - gold_f) / denom <= eps:
            return True
        if abs(pred * 100.0 - gold_f) / denom <= eps:
            return True
    return False


def mrr_at_k(ranked_ids: Iterable[str], gold_id: str, *, k: int = 3) -> float:
    """Mean Reciprocal Rank@k for a single question (1/r or 0)."""
    for rank, doc_id in enumerate(list(ranked_ids)[:k], start=1):
        if doc_id == gold_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Iterable[str], gold_id: str, *, k: int = 3) -> float:
    """Recall@k for a single question (1 if gold in top-k else 0)."""
    return 1.0 if gold_id in list(ranked_ids)[:k] else 0.0
