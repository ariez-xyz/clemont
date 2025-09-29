"""KDTree-based fixed-radius nearest neighbour backend."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.neighbors import KDTree

from .base import FRNNBackend, FRNNResult


class KdTreeFRNN(FRNNBackend):
    """Simple KDTree-backed FRNN backend.

    Notes
    -----
    This implementation rebuilds the KDTree on every ``add`` call. While not
    suitable for high-throughput scenarios, it provides a straightforward
    baseline that matches the behaviour of the legacy monitor for correctness
    testing purposes.
    """

    _METRIC_MAP = {
        "l2": "euclidean",
        "l1": "manhattan",
        "linf": "chebyshev",
    }

    @classmethod
    def supported_metrics(cls) -> Tuple[str, ...]:
        return tuple(cls._METRIC_MAP.keys())

    def __init__(self, *, epsilon: float, metric: str = "linf") -> None:
        super().__init__(
            epsilon=epsilon,
            metric=metric,
            is_sound=True,
            is_complete=True,
        )
        self._points: list[np.ndarray] = []
        self._ids: list[int] = []
        self._tree: Optional[KDTree] = None
        self._sk_metric = self._METRIC_MAP[self.metric]

    def _rebuild_tree(self) -> None:
        if not self._points:
            self._tree = None
            return
        data = np.vstack(self._points)
        self._tree = KDTree(data, metric=self._sk_metric)

    def add(self, point: Iterable[float], point_id: int) -> None:
        arr = np.asarray(tuple(point), dtype=float)
        if arr.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")
        self._points.append(arr)
        self._ids.append(int(point_id))
        self._rebuild_tree()

    def query(self, point: Iterable[float], *, radius: Optional[float] = None) -> FRNNResult:
        if self._tree is None:
            return FRNNResult(ids=())

        arr = np.asarray(tuple(point), dtype=float)
        if arr.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")

        r = self.resolve_radius(radius)
        indices, distances = self._tree.query_radius(arr.reshape(1, -1), r, return_distance=True)

        idx_row = indices[0]
        dist_row = distances[0]
        if len(idx_row) == 0:
            return FRNNResult(ids=())

        unique: dict[int, float] = {}
        for idx, dist in zip(idx_row, dist_row):
            pid = self._ids[int(idx)]
            fd = float(dist)
            if pid not in unique or fd < unique[pid]:
                unique[pid] = fd

        ordered = sorted(unique.items(), key=lambda item: (item[1], item[0]))
        ordered_ids = tuple(pid for pid, _ in ordered)
        ordered_dists = tuple(dist for _, dist in ordered)
        return FRNNResult(ids=ordered_ids, distances=ordered_dists)
