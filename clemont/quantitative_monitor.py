from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Literal, Set, cast

import math
import time
import numpy as np

from clemont.frnn import FRNNBackend, FRNNResult


# ---------- Utilities for output-space distances & bounds ----------

def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))

def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=2))

def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))

def _tv(a: np.ndarray, b: np.ndarray) -> float:
    # Total variation distance = 0.5 * L1
    return 0.5 * _l1(a, b)

# Cosine distance in [0, 2], but for nonnegative probability vectors (sum to 1) it stays in [0, 1]
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos_sim = float(np.dot(a, b) / denom)
    return 1.0 - max(min(cos_sim, 1.0), -1.0)

_DOUTS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "l1": _l1,
    "l2": _l2,
    "linf": _linf,
    "tv": _tv,
    "cosine": _cosine_distance,
}

# Known safe global bounds b for y in the probability simplex (dimension k >= 2)
# - L∞: <= 1 (e.g., e_i vs e_j differs by 1 in two coords)
# - L1: <= 2
# - TV = 0.5 * L1: <= 1
# - L2: <= sqrt(2) (||e_i - e_j||_2)
# - Cosine distance: <= 1 for nonnegative vectors
_BOUNDS: Dict[str, float] = {
    "linf": 1.0,
    "l1": 2.0,
    "tv": 1.0,
    "l2": math.sqrt(2.0),
    "cosine": 1.0,
}


@dataclass(frozen=True)
class QuantitativeResult:
    """Return value for a single QuantitativeMonitor.observe call."""
    point_id: int
    max_ratio: float
    witness_id: Optional[int]
    witness_in_distance: Optional[float]
    witness_out_distance: Optional[float]
    compared_count: int                   # number of historical points actually compared
    k_progression: Tuple[int, ...]        # ks we used (e.g., 16, 32, 64, ...)
    ratio_progression: Tuple[float, ...]  # how the max ratio develops as a function of k
    bound_progression: Tuple[float, ...]  # how the bound develops as a function of k
    time_progression: Tuple[float, ...]   # time for each iteration
    stopped_by_bound: bool                # whether we early-stopped via b / d_k
    stopped_by_maxk: bool                # whether we early-stopped via b / d_k
    note: Optional[str] = None            # any extra diagnostic note

    @classmethod
    def empty(cls, pid) -> 'QuantitativeResult':
        """Return an empty QuantitativeResult with default values."""
        return cls(
            point_id=pid,
            max_ratio=0.0,
            witness_id=None,
            witness_in_distance=None,
            witness_out_distance=None,
            compared_count=0,
            k_progression=(),
            ratio_progression=(),
            bound_progression=(),
            time_progression=(),
            stopped_by_bound=False,
            stopped_by_maxk=False,
            note="Empty result",
        )


class QuantitativeMonitor:
    """
    Quantitative monitor over (x, y) pairs using a single global FRNN backend
    for inputs x, plus an external store for outputs y.

    For a new (x, y), computes:
        max_{(x', y') in H} d_out(y, y') / d_in(x, x')
    by querying k-NN with geometrically increasing k and early-stopping via a
    known bound b on d_out for probability vectors.
    """

    def __init__(
        self,
        backend_factory: Callable[[], FRNNBackend],
        *,
        out_metric: Literal["linf", "l1", "l2", "tv", "cosine"] = "linf",
        initial_k: int = 16,
        max_k: Optional[int] = None,
        k_grow_factor: Optional[float] = None,
        tol: float = 1e-6,
        input_exponent: float = 1,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self._backend = backend_factory()
        if not isinstance(self._backend, FRNNBackend):
            raise TypeError("backend_factory must return an FRNNBackend")
        if not self._backend.supports_knn:
            raise RuntimeError("QuantitativeMonitor requires a backend with native k-NN support")

        if initial_k <= 0:
            raise ValueError("initial_k must be positive")
        if max_k is not None and max_k < initial_k:
            raise ValueError("max_k must be >= initial_k")
        if tol is not None and tol < 0:
            raise ValueError("tol must be >= 0")
        if k_grow_factor is not None and k_grow_factor <= 0:
            raise ValueError("k_grow_factor must be > 0")

        if out_metric not in _DOUTS:
            raise ValueError(f"Unsupported out_metric: {out_metric}")

        self._dout = _DOUTS[out_metric]
        self._b = _BOUNDS[out_metric]      # finite bound enables early-stopping
        self._initial_k: int = int(initial_k)
        self._max_k = None if (max_k is None) else int(max_k)
        self._k_grow_factor = 2 if not k_grow_factor else k_grow_factor
        self._tol = float(tol)
        self._input_exponent = input_exponent

        self._next_point_id: int = 0
        self._ys: Dict[int, np.ndarray] = {}                # id -> y (probability vector)

    @property
    def size(self) -> int:
        """Number of points currently indexed (== len(self._ys))."""
        return len(self._ys)

    def batch_add(
        self,
        items: Iterable[Tuple[Iterable[float], Iterable[float]] | Tuple[Iterable[float], Iterable[float], Optional[int]]],
    ) -> None:
        """Preload the monitor with multiple (point, probability) pairs.

        Parameters
        ----------
        items:
            Iterable yielding either ``(point, probability)`` or
            ``(point, probability, point_id)`` tuples. When ``point_id`` is omitted,
            the monitor assigns incremental identifiers just like ``observe``.

        Raises
        ------
        ValueError
            If any supplied point identifier collides with existing entries or
            appears multiple times within the batch.
        """

        pending: list[Tuple[int, np.ndarray, np.ndarray]] = []
        existing_ids = set(self._ys.keys())
        seen_ids: Set[int] = set()
        next_auto_id = self._next_point_id

        for raw in items:
            entry = tuple(raw)
            if len(entry) == 2:
                entry = cast(Tuple[Iterable[float], Iterable[float]], entry)
                point_raw, prob_raw = entry
                point_id_raw: Optional[int] = None
            elif len(entry) == 3:
                entry = cast(Tuple[Iterable[float], Iterable[float], Optional[int]], entry)
                point_raw, prob_raw, point_id_raw = entry
            else:
                raise ValueError(
                    "batch_add expects entries of the form (point, prob) or (point, prob, point_id)"
                )

            x_vec = np.asarray(list(point_raw), dtype=float).reshape(-1)
            if x_vec.size == 0:
                raise ValueError("points must be non-empty sequences")
            y_vec = np.asarray(list(prob_raw), dtype=float).reshape(-1)

            if point_id_raw is None:
                pid = next_auto_id
                next_auto_id += 1
            else:
                pid = int(point_id_raw)

            if pid in existing_ids or pid in seen_ids:
                raise ValueError(f"duplicate point_id {pid} in batch_add")
            seen_ids.add(pid)

            pending.append((pid, x_vec, y_vec))

        if not pending:
            return

        # Ensure the next automatically assigned id exceeds all provided ids.
        max_pid = max(pid for pid, _, _ in pending)
        next_auto_id = max(next_auto_id, max_pid + 1)

        self._backend.batch_add((x, pid) for pid, x, _ in pending)
        for pid, _, y_vec in pending:
            self._ys[pid] = y_vec

        self._next_point_id = next_auto_id

    def _resolve_point_id(self, point_id: Optional[int]) -> int:
        if point_id is None:
            pid = self._next_point_id
            self._next_point_id += 1
            return pid
        return int(point_id)

    def observe(
        self,
        point: Iterable[float],
        y: Iterable[float],
        *,
        point_id: Optional[int] = None,
        dry_run: bool = False,
    ) -> QuantitativeResult:
        """
        Compute the quantitative ratio for (point, y) against existing history,
        then insert the point into the index.

        Important: we compute against *history only* (the new point is added after).
        """
        x_vec = np.asarray(list(point), dtype=float).reshape(-1)
        y_vec = np.asarray(list(y), dtype=float).reshape(-1)

        pid = self._resolve_point_id(point_id)
        k: int = self._initial_k

        # Short-circuit: if history empty, return 0.0 (no comparator)
        if self.size == 0:
            self._backend.add(x_vec, pid)
            self._ys[pid] = y_vec
            return QuantitativeResult.empty(pid)

        # Main loop: double k, accumulate unique neighbors, early-stop via b / d_k
        seen: Set[int] = set()
        compared_count = 0
        k_progression: list[int] = []
        ratio_progression: list[float] = []
        bound_progression: list[float] = []
        time_progression: list[float] = []
        max_ratio = -math.inf
        witness = None
        witness_in = None
        witness_out = None
        stopped_by_bound = False
        stopped_by_maxk = False
        note = None

        # We keep track of the current furthest *considered* input distance.
        # Any unseen neighbor must be at least this far away.
        furthest_seen_din = 0.0

        terminate_outer = False

        while not terminate_outer: # repeat kNN queries
            iter_start = time.time()

            res: FRNNResult = self._backend.query_knn(x_vec, k=k)

            if res.distances is None:
                raise RuntimeError("QuantitativeMonitor requires distances from query_knn")

            # FRNNResult is always sorted by distance, so distances are non-decreasing.
            ids_batch = list(res.ids)
            dists_batch = list([d ** self._input_exponent for d in res.distances])

            # If fewer than k returned, we've exhausted the index.
            exhausted = (len(ids_batch) < k)

            # Update furthest seen distance using the *largest* distance from this batch (if any).
            if dists_batch:
                furthest_seen_din = max(furthest_seen_din, float(dists_batch[-1]))

            for j, nid in enumerate(ids_batch):
                # Compute ratio only for new ids (avoid re-compute across growing k)
                if nid in seen:
                    continue
                seen.add(nid)

                din = float(dists_batch[j])
                y_hist = self._ys.get(nid)
                assert y_hist is not None, f"failed sanity check: missing prediction for past point {nid}"
                dout = self._dout(y_vec, y_hist)

                # Guard small denominators; treat exact-zero carefully:
                #  - if din == 0 and dout > 0 -> ratio = +inf (definite violation)
                #  - if din == 0 and dout == 0 -> define ratio = 0 (doesn't raise the max)
                if din == 0.0:
                    if dout > self._tol:
                        ratio = math.inf
                        note = "Model assigns distinct outputs to identical inputs?"
                    else:
                        ratio = 0.0
                else:
                    ratio = dout / (din + self._tol)

                compared_count += 1

                if ratio > max_ratio:
                    max_ratio = ratio
                    witness = nid
                    witness_in = din
                    witness_out = dout

                if ratio == math.inf:
                    terminate_outer = True

            # Early stopping via bound b / furthest_seen_din:
            # Any unseen neighbor must have d_in >= furthest_seen_din,
            # hence max possible unseen ratio <= b / furthest_seen_din.
            bound = (self._b / (furthest_seen_din + self._tol))
            if max_ratio >= bound:
                stopped_by_bound = True
                terminate_outer = True

            if exhausted:
                terminate_outer = True

            if self._max_k is not None and k >= self._max_k:
                stopped_by_maxk = True
                terminate_outer = True

            k_progression.append(k)
            ratio_progression.append(max_ratio)
            bound_progression.append(bound)
            time_progression.append(time.time() - iter_start)

            # Otherwise, grow k.
            k = math.ceil(min(k * self._k_grow_factor, self._max_k) if self._max_k is not None else (k * self._k_grow_factor))

        # Done comparing; index the new point.
        if not dry_run:
            self._backend.add(x_vec, pid)
            self._ys[pid] = y_vec

        return QuantitativeResult(
            point_id=pid,
            max_ratio=max_ratio if max_ratio != -math.inf else 0.0,
            witness_id=witness,
            witness_in_distance=witness_in,
            witness_out_distance=witness_out,
            compared_count=compared_count,
            k_progression=tuple(k_progression),
            ratio_progression=tuple(ratio_progression),
            bound_progression=tuple(bound_progression),
            time_progression=tuple(time_progression),
            stopped_by_bound=stopped_by_bound,
            stopped_by_maxk=stopped_by_maxk,
            note=note,
        )
