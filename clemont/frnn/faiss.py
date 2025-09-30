"""FAISS-based fixed-radius nearest neighbour backend."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .base import FRNNBackend, FRNNResult

import faiss  

class FaissFRNN(FRNNBackend):
    """FAISS IndexFlat-based FRNN backend.

    This implementation stores all points in a single FAISS index and performs
    range-style queries by progressively increasing the number of neighbours
    requested from FAISS until the epsilon-cut is satisfied.
    """

    _METRIC_MAP: Dict[str, int] = {
        "linf": faiss.METRIC_Linf,
        "l2": faiss.METRIC_L2,
        "l1": faiss.METRIC_L1,
        "cosine": faiss.METRIC_INNER_PRODUCT,
    }

    @classmethod
    def supported_metrics(cls) -> Tuple[str, ...]:
        return tuple(
            metric for metric, metric_id in cls._METRIC_MAP.items() if metric_id is not None
        )

    @property
    def supports_knn(self) -> bool:
        return True

    def __init__(self, *, epsilon: float, metric: str = "linf", nthreads: int = 0) -> None:
        super().__init__(
            epsilon=epsilon,
            metric=metric,
            is_sound=True,
            is_complete=True,
        )

        metric_id = self._METRIC_MAP[self.metric]
        if metric_id is None:
            raise RuntimeError(f"FAISS does not provide metric '{self.metric}' on this platform")

        if nthreads > 0:
            faiss.omp_set_num_threads(nthreads)

        self._dim: Optional[int] = None
        self._metric_id = metric_id
        self._index: Optional[faiss.IndexIDMap] = None  # type: ignore[name-defined]

    def _ensure_index(self, dim: int) -> None:
        if self._dim is None: # create index of dimension dim
            self._dim = dim
            if self._index is None:
                flat = faiss.IndexFlat(dim, self._metric_id)
                self._index = faiss.IndexIDMap(flat)
        if dim != self._dim:
            raise ValueError("all points must share the same dimensionality")

    def _compute_faiss_epsilon(self, epsilon: float) -> float:
        if self.metric == "l2":
            return epsilon ** 2
        return epsilon

    def _transform_distances_for_output(self, distances: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return 1.0 - distances
        if self.metric == "l2":
            return np.sqrt(np.maximum(distances, 0.0))
        return distances

    def _prepare_query_vector(self, point: Iterable[float]) -> np.ndarray:
        vector = np.asarray(tuple(point), dtype="float32")
        if vector.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")

        if self._dim is None or vector.shape[0] != self._dim:
            raise ValueError("query dimensionality does not match indexed points")

        norm = np.linalg.norm(vector)
        if self.metric == "cosine" and not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("points must have unit L2 norm when using cosine metric")

        return vector

    def add(self, point: Iterable[float], point_id: int) -> None:
        vector = np.asarray(tuple(point), dtype="float32")
        if vector.ndim != 1:
            raise ValueError("points must be one-dimensional sequences")

        norm = np.linalg.norm(vector)
        if self.metric == "cosine" and not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("points must have unit L2 norm when using cosine metric")

        dim = vector.shape[0]
        self._ensure_index(dim)

        self._index.add_with_ids(vector.reshape(1, -1), np.array([int(point_id)], dtype="int64")) # type: ignore

    def query(self, point: Iterable[float], *, radius: Optional[float] = None) -> FRNNResult:
        if self._index is None or self._index.ntotal == 0:
            return FRNNResult(ids=())

        vector = self._prepare_query_vector(point)
        epsilon = self.resolve_radius(radius)
        faiss_epsilon = self._compute_faiss_epsilon(epsilon)

        query_vec = vector.reshape(1, -1)

        total = self._index.ntotal
        initial_k = min(max(1, total), 4)

        def _search(k: int):
            distances_raw, ids_raw = self._index.search(query_vec, k)  # type: ignore
            return distances_raw[0], ids_raw[0]

        raw_distances, raw_ids = self.emulate_range_query(
            _search,
            faiss_epsilon,
            initial_k=initial_k,
            max_k=total,
        )

        if raw_ids.size == 0:
            return FRNNResult(ids=())

        distances = self._transform_distances_for_output(raw_distances)

        return FRNNResult.from_iterables(raw_ids, distances)

    def query_knn(
        self,
        point: Iterable[float],
        *,
        k: int,
        radius: Optional[float] = None,
    ) -> FRNNResult:
        if k <= 0:
            raise ValueError("k must be positive")

        if self._index is None or self._index.ntotal == 0:
            return FRNNResult(ids=())

        vector = self._prepare_query_vector(point)
        query_vec = vector.reshape(1, -1)

        k = min(k, self._index.ntotal)
        distances_raw, ids_raw = self._index.search(query_vec, k)  # type: ignore

        ids = ids_raw[0]
        distances = self._transform_distances_for_output(distances_raw[0])

        valid_mask = ids != -1
        if not np.any(valid_mask):
            return FRNNResult(ids=())

        ids = ids[valid_mask]
        distances = distances[valid_mask]

        if radius is not None:
            epsilon = self.resolve_radius(radius)
            tol = epsilon * 1e-9 if epsilon > 1 else 1e-9
            within_mask = distances <= (epsilon + tol)
            ids = ids[within_mask]
            distances = distances[within_mask]

        if ids.size == 0:
            return FRNNResult(ids=())

        return FRNNResult.from_iterables(ids, distances)
