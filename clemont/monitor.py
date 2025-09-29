"""Monitoring utilities built on top of FRNN backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Optional, Tuple

from clemont.frnn import FRNNBackend, FRNNResult


@dataclass(frozen=True)
class ObservationResult:
    """Return value for a single `Monitor.observe` call."""

    point_id: int
    counterexamples: FRNNResult


class Monitor:
    """Manage per-decision FRNN indices and surface monitor-friendly semantics."""

    def __init__(self, backend_factory: Callable[[], FRNNBackend]) -> None:
        """Create a monitor backed by per-decision FRNN indices.

        Parameters
        ----------
        backend_factory:
            Callable returning a new ``FRNNBackend``. The callable should
            capture any configuration (e.g., metric, epsilon) internally via a
            closure, ``functools.partial`` or other mechanism.
        """

        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")

        self._backend_factory = backend_factory
        self._backends: Dict[Hashable, FRNNBackend] = {}
        self._next_point_id = 0

    @property
    def decisions(self) -> Tuple[Hashable, ...]:
        """Return the decisions currently tracked by the monitor."""

        return tuple(self._backends.keys())

    def _make_backend(self) -> FRNNBackend:
        """Instantiate a backend for the configured metric and epsilon."""

        try:
            backend = self._backend_factory()
        except Exception as exc:
            raise RuntimeError(
                "failed to construct backend via monitor factory"
            ) from exc
        if not isinstance(backend, FRNNBackend):
            raise TypeError("backend_factory must return an instance of FRNNBackend")
        return backend

    def _combine_results(self, results: Iterable[FRNNResult]) -> FRNNResult:
        """Merge neighbour results from multiple decision-specific backends."""

        merged: Dict[int, Optional[float]] = {}
        has_distances: bool = all([x.distances is not None for x in results])

        for result in results:
            if not result.ids:
                continue

            current_has_distances = result.distances is not None

            assert has_distances == current_has_distances, "Inconsistent distance availability across per-decision backends"

            if current_has_distances:
                assert result.distances is not None  # for type-checkers
                dists_raw = result.distances
            else:
                dists_raw = [0] * len(result.ids)

            for pid_raw, dist_raw in zip(result.ids, dists_raw):
                pid = int(pid_raw)
                assert pid not in merged, "Duplicate neighbour id encountered across per-decision backends"
                merged[pid] = float(dist_raw)

        if not merged:
            return FRNNResult(ids=())

        if has_distances:
            ordered = sorted(
                ((pid, dist) for pid, dist in merged.items() if dist is not None),
                key=lambda item: (item[1], item[0]),
            )
            ordered_ids = tuple(pid for pid, _ in ordered)
            ordered_distances = tuple(dist for _, dist in ordered)
            return FRNNResult(ids=ordered_ids, distances=ordered_distances)

        return FRNNResult(ids=tuple(sorted(merged.keys())))

    def observe(
        self,
        point: Iterable[float],
        decision: Hashable,
        *,
        point_id: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> ObservationResult:
        """Record an observation and return counterexamples from other decisions.

        Parameters
        ----------
        point:
            Feature vector for the observation.
        decision:
            Decision label associated with the point.
        point_id:
            Unique identifier for the point. If omitted, an incremental ID is
            assigned automatically.
        radius:
            Optional override for the FRNN radius when querying other decision
            buckets.
        """
        if point_id is None:
            point_id = self._next_point_id
            self._next_point_id += 1

        queries = []
        for dec_value, backend in self._backends.items():
            if dec_value == decision:
                continue
            queries.append(backend.query(point, radius=radius))

        backend = self._backends.get(decision)
        if backend is None:
            backend = self._make_backend()
            self._backends[decision] = backend

        backend.add(point, point_id)

        combined = self._combine_results(queries)
        return ObservationResult(point_id=point_id, counterexamples=combined)
