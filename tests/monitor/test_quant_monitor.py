# tests/test_quantitative_monitor.py
from __future__ import annotations

import math
import numpy as np
import pytest

from clemont.quantitative_monitor import QuantitativeMonitor
from clemont.frnn import FRNNBackend, FRNNResult


# =========================
# Exact, in-memory backend
# =========================

class ExactBackend(FRNNBackend):
    """
    A simple, exact FRNN backend used only for tests.
    - Stores all points in-memory
    - Computes distances exactly with numpy
    - Supports both range queries and k-NN queries
    """

    def __init__(self, *, epsilon: float = 1.0, metric: str = "l2"):
        super().__init__(epsilon=epsilon, metric=metric, is_sound=True, is_complete=True)
        self._xs: list[np.ndarray] = []
        self._ids: list[int] = []

    @classmethod
    def supported_metrics(cls) -> tuple[str, ...]:
        return ("l2", "l1", "linf", "cosine")

    @property
    def supports_knn(self) -> bool:
        return True

    def add(self, point, point_id: int) -> None:
        x = np.asarray(point, dtype=float).reshape(-1)
        self._xs.append(x)
        self._ids.append(int(point_id))

    # Distance helpers
    def _d(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "l2":
            return float(np.linalg.norm(a - b, ord=2))
        if self.metric == "l1":
            return float(np.linalg.norm(a - b, ord=1))
        if self.metric == "linf":
            return float(np.linalg.norm(a - b, ord=np.inf))
        if self.metric == "cosine":
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0.0 or nb == 0.0:
                return 0.0
            cos_sim = float(np.dot(a, b) / (na * nb))
            cos_sim = max(min(cos_sim, 1.0), -1.0)
            return 1.0 - cos_sim
        raise ValueError(f"unsupported metric: {self.metric}")

    def _all_dists(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._xs:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=int)
        X = np.vstack(self._xs)
        if self.metric == "l2":
            d = np.linalg.norm(X - q, axis=1)
        elif self.metric == "l1":
            d = np.linalg.norm(X - q, ord=1, axis=1)
        elif self.metric == "linf":
            d = np.linalg.norm(X - q, ord=np.inf, axis=1)
        elif self.metric == "cosine":
            qn = np.linalg.norm(q)
            Xn = np.linalg.norm(X, axis=1)
            denom = Xn * qn
            with np.errstate(invalid="ignore", divide="ignore"):
                cos = (X @ q) / denom
            cos = np.clip(np.nan_to_num(cos, nan=1.0), -1.0, 1.0)
            d = 1.0 - cos
        else:
            raise ValueError(f"unsupported metric: {self.metric}")
        ids = np.asarray(self._ids, dtype=int)
        return d, ids

    def query(self, point, *, radius: float | None = None) -> FRNNResult:
        eps = self.resolve_radius(radius)
        q = np.asarray(point, dtype=float).reshape(-1)
        d, ids = self._all_dists(q)
        if d.size == 0:
            return FRNNResult(ids=())
        mask = d <= eps + (1e-9 if eps > 1 else 1e-9)
        if not np.any(mask):
            return FRNNResult(ids=())
        order = np.argsort(d[mask], kind="stable")
        return FRNNResult.from_iterables(ids[mask][order], d[mask][order])

    def query_knn(self, point, *, k: int, radius: float | None = None) -> FRNNResult:
        q = np.asarray(point, dtype=float).reshape(-1)
        d, ids = self._all_dists(q)
        if d.size == 0:
            return FRNNResult(ids=())
        order = np.argsort(d, kind="stable")
        order = order[: min(k, order.size)]
        # Optional post-filter by radius if provided
        if radius is not None:
            eps = float(radius)
            mask = d[order] <= eps + (1e-9 if eps > 1 else 1e-9)
            order = order[mask]
        return FRNNResult.from_iterables(ids[order], d[order])


class NoKNNBackend(FRNNBackend):
    """Backend without native k-NN support (to test constructor checks)."""
    def __init__(self, *, epsilon: float = 1.0, metric: str = "l2"):
        super().__init__(epsilon=epsilon, metric=metric, is_sound=True, is_complete=True)

    @classmethod
    def supported_metrics(cls) -> tuple[str, ...]:
        return ("l2",)

    # supports_knn remains False

    def add(self, point, point_id: int) -> None:
        pass

    def query(self, point, *, radius: float | None = None) -> FRNNResult:
        return FRNNResult(ids=())


# =========================
# Output-space distances
# =========================

def dout_fn(name: str):
    def _l1(a, b): return float(np.sum(np.abs(a - b)))
    def _l2(a, b): return float(np.linalg.norm(a - b, ord=2))
    def _linf(a, b): return float(np.max(np.abs(a - b)))
    def _tv(a, b): return 0.5 * _l1(a, b)
    def _cos(a, b):
        na = np.linalg.norm(a) * np.linalg.norm(b)
        if na == 0.0:
            return 0.0
        cs = float(np.dot(a, b) / na)
        cs = max(min(cs, 1.0), -1.0)
        return 1.0 - cs
    m = {
        "l1": _l1,
        "l2": _l2,
        "linf": _linf,
        "tv": _tv,
        "cosine": _cos,
    }
    return m[name]


# =========================
# Fixtures
# =========================

@pytest.fixture
def backend_factory_l2():
    return lambda: ExactBackend(epsilon=1.0, metric="l2")

@pytest.fixture
def backend_factory_linf():
    return lambda: ExactBackend(epsilon=1.0, metric="linf")

@pytest.fixture
def rng():
    return np.random.default_rng(seed=12345)


# =========================
# Hand-crafted tests
# =========================

def test_constructor_requires_knn():
    with pytest.raises(RuntimeError):
        _ = QuantitativeMonitor(lambda: NoKNNBackend(), out_metric="linf")


def test_empty_history_returns_zero(backend_factory_l2):
    qm = QuantitativeMonitor(backend_factory_l2, out_metric="linf", initial_k=4)
    res = qm.observe([0.0, 0.0], [1.0, 0.0, 0.0])
    assert res.max_ratio == 0.0
    assert res.witness_id is None
    assert res.compared_count == 0
    assert res.k_progression == ()
    assert res.stopped_by_bound is False
    assert "No history" in (res.note or "")


def test_known_geometry_and_outputs_expected_result(backend_factory_l2):
    """
    History:
      p0: x=[0,0],   y=[1,0,0]
      p1: x=[1,0],   y=[0,1,0]
      p2: x=[0,1],   y=[0,0,1]
    New:
      q:  x=[0,0.1], y=[0.9,0.1,0.0]
    Input metric: L2, Output metric: L∞
    Expected:
      d_in to p0 = 0.1,   d_out = 0.1, ratio = 1.0
      d_in to p1 ~ 1.005, d_out = 0.9, ratio ~ 0.8955
      d_in to p2 = 0.9,   d_out = 1.0, ratio = 1.111...
      => max at p2 with ~1.111...
    """
    qm = QuantitativeMonitor(backend_factory_l2, out_metric="linf", initial_k=10)

    # Seed history
    qm.observe([0.0, 0.0], [1.0, 0.0, 0.0])
    qm.observe([1.0, 0.0], [0.0, 1.0, 0.0])
    qm.observe([0.0, 1.0], [0.0, 0.0, 1.0])

    res = qm.observe([0.0, 0.1], [0.9, 0.1, 0.0])

    assert pytest.approx(res.max_ratio, rel=1e-10, abs=1e-10) == 1.0 / 0.9
    assert res.witness_in_distance is not None
    assert pytest.approx(res.witness_in_distance, rel=1e-12, abs=1e-12) == 0.9
    assert pytest.approx(res.witness_out_distance, rel=1e-12, abs=1e-12) == 1.0
    # Early stop by bound is likely true here because furthest_seen_din ~1.005 -> 1/1.005 < 1.111
    assert res.stopped_by_bound is True
    assert len(res.k_progression) >= 1
    assert res.k_progression[0] == 10  # exhausted index, but first k is recorded


def test_zero_input_distance_infinite_ratio(backend_factory_l2):
    qm = QuantitativeMonitor(backend_factory_l2, out_metric="l1", initial_k=2)

    # Seed one point
    qm.observe([2.0, 2.0], [0.4, 0.6, 0.0])

    # Same x, different y -> din = 0, dout > 0 => ratio = +inf and immediate stop
    res = qm.observe([2.0, 2.0], [0.5, 0.5, 0.0])

    assert math.isinf(res.max_ratio)
    assert res.witness_id is not None
    assert res.witness_in_distance == 0.0
    assert res.witness_out_distance is not None and res.witness_out_distance > 0.0
    assert res.stopped_by_bound is True
    assert "Zero input distance" in (res.note or "")

# =========================
# Randomized validation vs naive O(n^2)
# =========================

def _naive_max_ratio(
    X: np.ndarray,
    Y: np.ndarray,
    i: int,
    dout_name: str,
    din_name: str,
    tol: float = 1e-12,
):
    """Return (max_ratio, witness_id, witness_din, witness_dout, count) for point i vs 0..i-1."""
    if i == 0:
        return 0.0, None, None, None, 0

    def din(a, b):
        if din_name == "l2":
            return float(np.linalg.norm(a - b, ord=2))
        if din_name == "l1":
            return float(np.linalg.norm(a - b, ord=1))
        if din_name == "linf":
            return float(np.linalg.norm(a - b, ord=np.inf))
        if din_name == "cosine":
            na = np.linalg.norm(a) * np.linalg.norm(b)
            if na == 0.0:
                return 0.0
            cs = float(np.dot(a, b) / na)
            cs = max(min(cs, 1.0), -1.0)
            return 1.0 - cs
        raise ValueError

    dout = dout_fn(dout_name)
    xi = X[i]
    yi = Y[i]
    max_ratio = -math.inf
    w_id = None
    w_din = None
    w_dout = None
    count = 0

    for j in range(i):
        d_in = din(xi, X[j])
        d_out = dout(yi, Y[j])
        if d_in == 0.0:
            if d_out > 0.0:
                return math.inf, j, 0.0, d_out, count + 1
            ratio = 0.0
        else:
            ratio = d_out / (d_in + tol)
        count += 1
        if ratio > max_ratio:
            max_ratio = ratio
            w_id = j
            w_din = d_in
            w_dout = d_out

    if max_ratio == -math.inf:
        return 0.0, None, None, None, 0
    return max_ratio, w_id, w_din, w_dout, count


@pytest.mark.parametrize("din_metric", ["l2"])  # keep runtime reasonable; expand if desired
@pytest.mark.parametrize("dout_metric", ["linf"])  # primary bound-aware metric for monitor
def test_random_stream_matches_naive_all(backend_factory_l2, rng, din_metric, dout_metric):
    """
    Stream ~1000 points; for each point, compare monitor's result to naive O(n^2)
    on the prefix history (0..i-1). This tests exactness despite early stopping.
    """
    N = 1000
    d_in = 5
    k_out = 4

    # Generate random inputs X and probability outputs Y (Dirichlet)
    X = rng.normal(size=(N, d_in))
    Y = rng.dirichlet(alpha=np.ones(k_out), size=N)

    qm = QuantitativeMonitor(backend_factory_l2, out_metric=dout_metric, initial_k=10)

    ratios_mon = []
    ratios_naive = []

    for i in range(N):
        res = qm.observe(X[i], Y[i])
        ratios_mon.append(res.max_ratio)

        r, wid, dinv, doutv, cnt = _naive_max_ratio(X, Y, i, dout_metric, din_metric)
        ratios_naive.append(r)

        # Compare with tight tolerance
        if math.isinf(r):
            assert math.isinf(res.max_ratio)
        else:
            assert pytest.approx(res.max_ratio, rel=1e-10, abs=1e-10) == r

    # sanity
    assert len(ratios_mon) == N
    assert len(ratios_naive) == N


def test_k_progression_is_monotone_and_starts_at_initial(backend_factory_l2, rng):
    qm = QuantitativeMonitor(backend_factory_l2, out_metric="linf", initial_k=6)
    # Seed a few points spread out so early-stop likely triggers before exhausting
    for x, y in [
        ([0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.1, 0.0], [0.9, 0.1, 0.0]),
        ([5.0, 0.0], [0.0, 1.0, 0.0]),
        ([10.0, 0.0], [0.0, 0.0, 1.0]),
        ([20.0, 0.0], [0.2, 0.8, 0.0]),
    ]:
        qm.observe(x, y)

    res = qm.observe([0.05, 0.0], [0.92, 0.08, 0.0])
    if res.k_progression:
        assert res.k_progression[0] == 6
        assert all(res.k_progression[i] <= res.k_progression[i + 1]
                   for i in range(len(res.k_progression) - 1))
