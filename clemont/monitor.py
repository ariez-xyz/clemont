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

        results = []
        for dec_value, backend in self._backends.items():
            if dec_value == decision:
                continue
            results.append(backend.query(point, radius=radius))

        backend = self._backends.get(decision)
        if backend is None:
            backend = self._make_backend()
            self._backends[decision] = backend

        backend.add(point, point_id)

        combined = FRNNResult.merging(results)
        return ObservationResult(point_id=point_id, counterexamples=combined)
