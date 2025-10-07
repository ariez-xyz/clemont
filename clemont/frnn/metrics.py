"""Canonical metric names and helper utilities for FRNN backends."""

from __future__ import annotations

from typing import Tuple

_CANONICAL_METRICS: Tuple[str, ...] = ("linf", "l2", "l1", "cosine")


def canonical_metrics() -> Tuple[str, ...]:
    """Return the tuple of supported canonical metric names."""

    return _CANONICAL_METRICS


def ensure_canonical_metric(metric: str) -> str:
    """Validate and normalise the metric name.

    Only canonical names are accepted (case-insensitive). Synonyms such as
    "infinity" must be translated by the caller before reaching this layer.
    """

    if not isinstance(metric, str):
        raise TypeError("metric must be a string")

    normalized = metric.strip().lower()
    if normalized not in _CANONICAL_METRICS:
        raise ValueError(
            f"unsupported metric '{metric}'. Supported metrics: {_CANONICAL_METRICS}"
        )
    return normalized

