"""Naive fixed-radius nearest neighbour implementation.

This backend is intended for testing and fallback scenarios. It maintains an
in-memory list of points and answers queries via a brute-force search.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from .base import FRNNBackend, FRNNResult


class NaiveFRNN(FRNNBackend):
    """Brute-force FRNN backend supporting Linf, L2, and L1 distances."""

    _SUPPORTED: Tuple[str, ...] = ("linf", "l2", "l1")

    @classmethod
    def supported_metrics(cls) -> Tuple[str, ...]:
        return cls._SUPPORTED

    @property
    def supports_knn(self) -> bool:
        return True

    def __init__(self, *, epsilon: float, metric: str = "linf") -> None:
        super().__init__(
            epsilon=epsilon,
            metric=metric,
            is_sound=True,
            is_complete=True,
        )
        self._points: list[tuple[np.ndarray, int]] = []

    def add(self, point: Iterable[float], point_id: int) -> None:
        coords = np.asarray(tuple(point), dtype=float)
        if coords.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")
        self._points.append((coords, int(point_id)))

    def _compute_distance(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        if self.metric == "linf":
            return float(np.max(np.abs(lhs - rhs)))
        if self.metric == "l1":
            return float(np.sum(np.abs(lhs - rhs)))
        return float(np.linalg.norm(lhs - rhs))

    def query(self, point: Iterable[float], *, radius=None) -> FRNNResult:
        if not self._points:
            return FRNNResult(ids=())

        coords = np.asarray(tuple(point), dtype=float)
        if coords.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")

        epsilon = self.resolve_radius(radius)
        matches = []
        distances = []

        for stored_point, stored_id in self._points:
            if stored_point.shape != coords.shape:
                raise ValueError("all points must share the same dimensionality")
            distance = self._compute_distance(stored_point, coords)
            if distance <= epsilon:
                matches.append(stored_id)
                distances.append(distance)

        return FRNNResult.from_iterables(matches, distances)

    def query_knn(
        self,
        point: Iterable[float],
        *,
        k: int,
        radius: Optional[float] = None,
    ) -> FRNNResult:
        if k <= 0:
            raise ValueError("k must be positive")

        if not self._points:
            return FRNNResult(ids=())

        coords = np.asarray(tuple(point), dtype=float)
        if coords.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")

        candidates = []
        for stored_point, stored_id in self._points:
            if stored_point.shape != coords.shape:
                raise ValueError("all points must share the same dimensionality")
            distance = self._compute_distance(stored_point, coords)
            candidates.append((stored_id, distance))

        if radius is not None:
            epsilon = self.resolve_radius(radius)
            tol = epsilon * 1e-9 if epsilon > 1 else 1e-9
            candidates = [
                (pid, dist) for pid, dist in candidates if dist <= (epsilon + tol)
            ]

        if not candidates:
            return FRNNResult(ids=())

        candidates.sort(key=lambda item: item[1])

        top = candidates[:k]
        ids, distances = zip(*top)
        return FRNNResult(ids=tuple(ids), distances=tuple(distances))
