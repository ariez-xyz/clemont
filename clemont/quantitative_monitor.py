from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Literal, Set

import math
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
    k_progression: Tuple[int, ...]        # ks we used (e.g., 10, 20, 40, ...)
    stopped_by_bound: bool                # whether we early-stopped via b / d_k
    note: Optional[str] = None            # any extra diagnostic note


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
        initial_k: int = 10,
        max_k: Optional[int] = None,
        tol: float = 1e-12,
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self._backend = backend_factory()
        if not isinstance(self._backend, FRNNBackend):
            raise TypeError("backend_factory must return an FRNNBackend")
        if not self._backend.supports_knn:
            # You *can* still implement this by binary searching a radius and using range queries,
            # but the cleanest/fastest path is a backend with native k-NN.
            raise RuntimeError("QuantitativeMonitor requires a backend with native k-NN support")

        if initial_k <= 0:
            raise ValueError("initial_k must be positive")
        if max_k is not None and max_k < initial_k:
            raise ValueError("max_k must be >= initial_k")

        if out_metric not in _DOUTS:
            raise ValueError(f"Unsupported out_metric: {out_metric}")

        self._dout = _DOUTS[out_metric]
        self._b = _BOUNDS[out_metric]      # finite bound enables early-stopping
        self._initial_k = int(initial_k)
        self._max_k = None if (max_k is None) else int(max_k)
        self._tol = float(tol)

        self._next_point_id: int = 0
        self._ys: Dict[int, np.ndarray] = {}                # id -> y (probability vector)

    @property
    def size(self) -> int:
        """Number of points currently indexed (== len(self._ys))."""
        return len(self._ys)

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
        initial_k: Optional[int] = None,
        max_k: Optional[int] = None,
    ) -> QuantitativeResult:
        """
        Compute the quantitative ratio for (point, y) against existing history,
        then insert the point into the index.

        Important: we compute against *history only* (the new point is added after).
        """
        pid = self._resolve_point_id(point_id)
        x_vec = np.asarray(list(point), dtype=float).reshape(-1)
        y_vec = np.asarray(list(y), dtype=float).reshape(-1)

        # Short-circuit: if history empty, return 0.0 (no comparator)
        if self.size == 0:
            # Add and return
            self._backend.add(x_vec, pid)
            self._ys[pid] = y_vec
            return QuantitativeResult(
                point_id=pid,
                max_ratio=0.0,
                witness_id=None,
                witness_in_distance=None,
                witness_out_distance=None,
                compared_count=0,
                k_progression=(),
                stopped_by_bound=False,
                note="No history",
            )

        # Config overrides
        k = int(initial_k if initial_k is not None else self._initial_k)
        k_cap = int(max_k) if max_k is not None else self._max_k

        # Main loop: double k, accumulate unique neighbors, early-stop via b / d_k
        seen: Set[int] = set()
        compared_count = 0
        k_progression: list[int] = []
        max_ratio = -math.inf
        witness = None
        witness_in = None
        witness_out = None
        stopped_by_bound = False

        # We keep track of the current furthest *considered* input distance.
        # Any unseen neighbor must be at least this far away.
        furthest_seen_din = 0.0

        while True:
            res: FRNNResult = self._backend.query_knn(x_vec, k=k)

            if res.distances is None:
                raise RuntimeError("QuantitativeMonitor requires distances from query_knn")

            # FRNNResult sorts by distance in __post_init__, so distances are non-decreasing.
            ids_batch = list(res.ids)
            dists_batch = list(res.distances)

            # If fewer than k returned, we've exhausted the index.
            exhausted = (len(ids_batch) < k)

            # Update furthest seen distance using the *largest* distance from this batch (if any).
            if dists_batch:
                furthest_seen_din = max(furthest_seen_din, float(dists_batch[-1]))

            # Compare only new ids (avoid re-compute across growing k)
            for j, nid in enumerate(ids_batch):
                if nid in seen:
                    continue
                seen.add(nid)

                din = float(dists_batch[j])
                # Guard small denominators; treat exact-zero carefully:
                #  - if din == 0 and dout > 0 -> ratio = +inf (definite violation)
                #  - if din == 0 and dout == 0 -> define ratio = 0 (doesn't raise the max)
                y_hist = self._ys.get(nid)
                if y_hist is None:
                    # Should not happen; indicates stale index vs store.
                    continue
                dout = self._dout(y_vec, y_hist)

                if din == 0.0:
                    if dout > 0.0:
                        # maximal possible ratio; we can finish immediately
                        max_ratio = math.inf
                        witness = nid
                        witness_in = 0.0
                        witness_out = dout
                        compared_count += 1
                        stopped_by_bound = True
                        k_progression.append(k)
                        # Do NOT add the point yet; add after returning
                        result = QuantitativeResult(
                            point_id=pid,
                            max_ratio=max_ratio,
                            witness_id=witness,
                            witness_in_distance=witness_in,
                            witness_out_distance=witness_out,
                            compared_count=compared_count,
                            k_progression=tuple(k_progression),
                            stopped_by_bound=stopped_by_bound,
                            note="Zero input distance with nonzero output difference",
                        )
                        # insert afterwards
                        self._backend.add(x_vec, pid)
                        self._ys[pid] = y_vec
                        return result
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

            k_progression.append(k)

            # Early stopping via bound b / furthest_seen_din:
            # Any unseen neighbor must have d_in >= furthest_seen_din,
            # hence max possible unseen ratio <= b / furthest_seen_din.
            if furthest_seen_din > 0.0 and max_ratio >= (self._b / (furthest_seen_din + self._tol)):
                stopped_by_bound = True
                break

            # If we've exhausted the index, nothing more to see.
            if exhausted:
                break

            # If we reached k_cap, stop.
            if k_cap is not None and k >= k_cap:
                break

            # Otherwise, grow k geometrically.
            k = min(k * 2, k_cap) if k_cap is not None else (k * 2)

        # Done comparing; now insert the new point.
        self._backend.add(x_vec, pid)
        self._ys[pid] = y_vec

        note = None

        return QuantitativeResult(
            point_id=pid,
            max_ratio=max_ratio if max_ratio != -math.inf else 0.0,
            witness_id=witness,
            witness_in_distance=witness_in,
            witness_out_distance=witness_out,
            compared_count=compared_count,
            k_progression=tuple(k_progression),
            stopped_by_bound=stopped_by_bound,
            note=note,
        )
