"""KDTree-based fixed-radius nearest neighbour backend."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from numpy._core.fromnumeric import reshape, shape
from sklearn.neighbors import KDTree

from clemont.frnn.faiss import FaissFRNN

from .base import FRNNBackend, FRNNResult


class KdTreeFRNN(FRNNBackend):
    """KDTree-backed FRNN backend. Dynamic indexing is implemented with a
    long-term/short-term memory split, where the KDTree serves as long-term
    memory that is reindexed periodically (controllable via batchsize parameter).
    Points that arrive between reindexing cycles are stored in a brute-force 
    short-term memory. Query results are merged from both memories."""

    _METRIC_MAP = {
        "l2": "euclidean",
        "l1": "manhattan",
        "linf": "chebyshev",
    }

    @classmethod
    def supported_metrics(cls) -> Tuple[str, ...]:
        return tuple(cls._METRIC_MAP.keys())

    @property
    def supports_knn(self) -> bool:
        return True


    def __init__(
        self,
        *,
        epsilon: float | None = None,
        metric: str = "linf",
        batchsize: int = 500,
        bf_threads: int = 1,
    ) -> None:
        super().__init__(
            epsilon=epsilon,
            metric=metric,
            is_sound=True,
            is_complete=True,
        )
        self._batchsize: int = batchsize
        self._bf_threads: int = bf_threads
        self._current_batch_length: int = 0
        self._points: list[np.ndarray] = []
        self._ids: list[int] = []
        self._ltmemory: Optional[KDTree] = None
        self._stmemory: FaissFRNN = FaissFRNN(
            epsilon=self.epsilon,
            metric=self.metric,
            nthreads=bf_threads,
        )
        self._sk_metric = self._METRIC_MAP[self.metric]


    def _point2np(self, point: Iterable[float]) -> np.ndarray:
        arr = np.asarray(tuple(point), dtype=float)
        if arr.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")
        return arr.reshape(1, -1)


    def _build_ltmemory(self) -> None:
        if not self._points:
            self._ltmemory = None
            return
        data = np.vstack(self._points)
        self._ltmemory = KDTree(data, metric=self._sk_metric)


    def _ltmemory_query(self, point: Iterable[float], *, radius: Optional[float] = None) -> FRNNResult:
        if not self._ltmemory: return FRNNResult(ids=())

        r = self.resolve_radius(radius)
        arr = self._point2np(point)
        
        indices, distances = self._ltmemory.query_radius(arr, r, return_distance=True)

        idx_row = indices[0]
        dist_row = distances[0]

        if len(idx_row) == 0:
            return FRNNResult(ids=())

        # sklearn.KDTree uses sequential ids, need to map back to actual point_ids
        mapped_idxs: Tuple[int] = tuple([self._ids[idx] for idx in idx_row])

        return FRNNResult(ids=mapped_idxs, distances=dist_row)

    def _ltmemory_knn(self, point: Iterable[float], k: int) -> FRNNResult:
        if not self._ltmemory or k <= 0:
            return FRNNResult(ids=())

        arr = self._point2np(point)
        data = getattr(self._ltmemory, "data", None)
        available = data.shape[0] if data is not None else k
        max_k = min(k, available)
        if max_k <= 0:
            return FRNNResult(ids=())

        distances, indices = self._ltmemory.query(arr, k=max_k, return_distance=True)

        idx_row = np.asarray(indices[0], dtype=int)
        dist_row = np.asarray(distances[0], dtype=float)

        if idx_row.size == 0:
            return FRNNResult(ids=())

        mapped = [self._ids[idx] for idx in idx_row]

        return FRNNResult.from_iterables(mapped, dist_row)


    def _clear_stmemory(self) -> None:
        self._stmemory = FaissFRNN(epsilon=self.epsilon, metric=self.metric, nthreads=self._bf_threads)


    def add(self, point: Iterable[float], point_id: int) -> None:
        self._points.append(self._point2np(point))
        self._ids.append(int(point_id))
        self._stmemory.add(point, point_id)
        self._current_batch_length += 1

        if self._current_batch_length >= self._batchsize: # Rebuild
            self._build_ltmemory()
            self._clear_stmemory()
            self._current_batch_length = 0


    def query(self, point: Iterable[float], *, radius: Optional[float] = None) -> FRNNResult:
        if len(self._points) == 0:
            return FRNNResult(ids=())

        lt_results = self._ltmemory_query(point, radius=radius)
        st_results = self._stmemory.query(point, radius=radius)

        return FRNNResult.merging([lt_results, st_results])

    def query_knn(
        self,
        point: Iterable[float],
        *,
        k: int,
        radius: Optional[float] = None,
    ) -> FRNNResult:
        if k <= 0:
            raise ValueError("k must be positive")

        if len(self._points) == 0:
            return FRNNResult(ids=())

        lt_result = self._ltmemory_knn(point, k)
        st_result = self._stmemory.query_knn(point, k=k)

        if radius is not None:
            raise NotImplementedError("unsupported")

        return FRNNResult.merging([lt_result, st_result])
