"""Foundational classes for Clemont's Fixed-Radius Nearest Neighbour backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np


class RadiusOverrideNotSupported(RuntimeError):
    """Raised when a backend cannot honour a runtime radius override."""


@dataclass(frozen=True)
class FRNNResult:
    """Container for neighbour query results."""

    ids: Tuple[int, ...]
    distances: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.distances is not None and len(self.ids) != len(self.distances):
            raise ValueError("ids and distances must have matching lengths")
        
        # Sort ids and distances by distance if distances are provided
        if len(self.ids) > 0 and self.distances is not None:
            sorted_pairs = sorted(zip(self.ids, self.distances), key=lambda x: x[1])
            sorted_ids, sorted_distances = zip(*sorted_pairs)
            object.__setattr__(self, 'ids', sorted_ids)
            object.__setattr__(self, 'distances', sorted_distances)

    @classmethod
    def from_iterables(
        cls,
        ids: Iterable[int],
        distances: Optional[Iterable[float]] = None,
    ) -> "FRNNResult":
        """Build a result from generic iterables while enforcing tuple storage."""
        ids_tuple = tuple(int(i) for i in ids)
        distances_tuple: Optional[Tuple[float, ...]] = None
        if distances is not None:
            distances_tuple = tuple(float(d) for d in distances)
        return cls(ids=ids_tuple, distances=distances_tuple)

    @classmethod
    def merging(
        cls,
        results: Iterable[FRNNResult]
    ) -> "FRNNResult":
        """Merge compatible FRNNResult instances into a single result."""
        if not results:
            return cls(ids=())
        
        all_ids = []
        all_distances = []
        has_distances: bool = all([res.distances is not None for res in results])
        
        for result in results:
            for id_val in result.ids:
                if id_val in all_ids:
                    raise RuntimeError(f"Incompatible results: duplicate ID {id_val}.")

            current_has_distances = result.distances is not None
            if not has_distances and current_has_distances:
                raise RuntimeError("Incompatible results: inconsistent distance availability.")

            all_ids.extend(result.ids)
            if has_distances:
                all_distances.extend(result.distances)
        
        return cls(
            ids=tuple(all_ids),
            distances=tuple(all_distances) if has_distances else None
        )


class FRNNBackend(ABC):
    """Abstract interface for fixed-radius nearest neighbour search backends."""

    def __init__(
        self,
        *,
        epsilon: float,
        metric: str,
        is_sound: bool,
        is_complete: bool,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        from .metrics import ensure_canonical_metric  # Local import to avoid cycles.

        self._epsilon = float(epsilon)
        self._metric = ensure_canonical_metric(metric)
        self._is_sound = bool(is_sound)
        self._is_complete = bool(is_complete)
        self._range_query_ks: list[int] = []

        supported = {m.lower() for m in self.supported_metrics()}
        if supported and self._metric not in supported:
            raise ValueError(
                f"metric '{self._metric}' not supported by {self.__class__.__name__}"
            )

    @property
    def epsilon(self) -> float:
        """Default epsilon configured for the backend."""
        return self._epsilon

    @property
    def metric(self) -> str:
        """Distance metric used by the backend."""
        return self._metric

    @property
    def is_sound(self) -> bool:
        """Whether all reported violations are true positives."""
        return self._is_sound

    @property
    def is_complete(self) -> bool:
        """Whether all true violations are reported."""
        return self._is_complete

    @classmethod
    @abstractmethod
    def supported_metrics(cls) -> Tuple[str, ...]:
        """Return the set of canonical metrics supported by the backend."""

    @property
    def range_query_ks(self) -> Tuple[int, ...]:
        """Sequence of `k` values used by emulate_range_query for diagnostics."""

        return tuple(self._range_query_ks)

    def resolve_radius(self, radius: Optional[float]) -> float:
        """Return the radius to use for a query, enforcing backend constraints."""
        if radius is None:
            return self._epsilon
        if radius <= 0:
            raise ValueError("radius override must be positive")
        return float(radius)

    def emulate_range_query(
        self,
        query_fn: Callable[[int], Tuple[Iterable[float], Iterable[int]]],
        epsilon: float,
        *,
        initial_k: int = 4,
        max_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate a range query using adaptive k-NN searches.

        Parameters
        ----------
        query_fn:
            Callable accepting ``k`` and returning a pair ``(distances, ids)``
            for the requested neighbours.
        epsilon:
            Distance threshold in the same metric reported by ``query_fn``.
        initial_k:
            Starting value for ``k``.
        max_k:
            Upper bound for ``k``; defaults to the current index size when the
            backend knows it.
        """

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if initial_k <= 0:
            raise ValueError("initial_k must be positive")

        if max_k is not None and max_k < initial_k:
            raise ValueError("max_k must be >= initial_k")

        k = initial_k
        tol = epsilon * 1e-9 if epsilon > 1 else 1e-9

        while True:
            distances_raw, ids_raw = query_fn(k)
            distances = np.asarray(distances_raw, dtype=float).reshape(-1)
            ids = np.asarray(ids_raw, dtype=int).reshape(-1)

            valid_mask = ids != -1
            if not np.any(valid_mask):
                self._range_query_ks.append(0)
                return np.empty(0, dtype=float), np.empty(0, dtype=int)

            distances = distances[valid_mask]
            ids = ids[valid_mask]

            within_mask = distances <= (epsilon + tol)

            if np.all(within_mask):
                if max_k is not None and k >= max_k:
                    self._range_query_ks.append(k)
                    return distances, ids

                new_k = k * 2
                if max_k is not None:
                    new_k = min(new_k, max_k)

                if new_k == k:
                    self._range_query_ks.append(k)
                    return distances, ids

                k = new_k
                continue

            self._range_query_ks.append(k)
            return distances[within_mask], ids[within_mask]

    @abstractmethod
    def add(self, point, point_id: int) -> None:
        """Insert a new observation into the backend index."""

    @abstractmethod
    def query(
        self,
        point,
        *,
        radius: Optional[float] = None,
    ) -> FRNNResult:
        """Return neighbours for the given point within the requested radius."""
