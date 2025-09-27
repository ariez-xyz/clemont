"""Monitoring utilities built on top of FRNN backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Optional, Tuple

from clemont.frnn import FRNNBackend, FRNNResult
from clemont.frnn.metrics import ensure_canonical_metric


@dataclass(frozen=True)
class ObservationResult:
    """Return value for a single `Monitor.observe` call."""

    point_id: int
    counterexamples: FRNNResult


class Monitor:
    """Manage per-decision FRNN indices and surface monitor-friendly semantics."""

    def __init__(
        self,
        backend_factory: Callable[..., FRNNBackend],
        *,
        epsilon: float,
        metric: str,
        backend_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        """Create a monitor backed by per-decision FRNN indices.

        Parameters
        ----------
        backend_factory:
            Callable that returns a new ``FRNNBackend`` when invoked with
            ``epsilon`` and ``metric`` keyword arguments. This can be the class
            itself or any factory function/partial.
        epsilon:
            Default radius to configure for each backend instance.
        metric:
            Canonical metric name (e.g., ``"linf"``) that the backend should
            support.
        backend_kwargs:
            Additional keyword arguments forwarded to the factory.
        """
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")

        canonical_metric = ensure_canonical_metric(metric)

        self._backend_factory = backend_factory
        self._backend_kwargs = dict(backend_kwargs or {})
        self._epsilon = float(epsilon)
        self._metric = canonical_metric
        self._backends: Dict[Hashable, FRNNBackend] = {}
        self._next_point_id = 0

    @property
    def decisions(self) -> Tuple[Hashable, ...]:
        """Return the decisions currently tracked by the monitor."""

        return tuple(self._backends.keys())

    def _make_backend(self) -> FRNNBackend:
        """Instantiate a backend for the configured metric and epsilon."""

        try:
            return self._backend_factory(
                epsilon=self._epsilon,
                metric=self._metric,
                **self._backend_kwargs,
            )
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("backend_factory must accept epsilon= and metric=") from exc
        except Exception as exc:
            raise RuntimeError(
                f"failed to construct backend for metric '{self._metric}'"
            ) from exc

    def _combine_results(self, results: Iterable[FRNNResult]) -> FRNNResult:
        """Merge neighbour results from multiple decision-specific backends."""

        ids_with_dist: Dict[int, float] = {}
        all_distances_available = True
        seen_ids = set()

        for result in results:
            if not result.ids:
                continue

            if result.distances is None:
                all_distances_available = False
                seen_ids.update(result.ids)
                continue

            for pid, dist in zip(result.ids, result.distances):
                seen_ids.add(pid)
                if pid not in ids_with_dist or dist < ids_with_dist[pid]:
                    ids_with_dist[pid] = dist

        if not seen_ids:
            return FRNNResult(ids=())

        if not all_distances_available:
            return FRNNResult(ids=tuple(sorted(seen_ids)))

        ordered = sorted(ids_with_dist.items(), key=lambda item: (item[1], item[0]))
        ordered_ids = tuple(pid for pid, _ in ordered)
        ordered_dists = tuple(dist for _, dist in ordered)
        return FRNNResult(ids=ordered_ids, distances=ordered_dists)

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
