import functools

import pytest

from clemont.frnn import FRNNBackend, FRNNResult
from clemont.monitor import Monitor


class DummyFRNN(FRNNBackend):
    _SUPPORTED = ("linf",)

    @classmethod
    def supported_metrics(cls):
        return cls._SUPPORTED

    def __init__(self, *, epsilon: float, metric: str) -> None:
        super().__init__(epsilon=epsilon, metric=metric, is_sound=True, is_complete=True)
        self.points = {}
        self._init_epsilon = epsilon
        self._init_metric = metric

    def add(self, point, point_id: int) -> None:  # pragma: no cover - unused for combine tests
        self.points[point_id] = tuple(point)

    def query(self, point, *, radius=None):  # pragma: no cover - not used here
        return FRNNResult(ids=())


def _build_monitor():
    factory = lambda: DummyFRNN(epsilon=0.1, metric="linf")
    return Monitor(factory)


def test_frnn_merge_basic():
    monitor = _build_monitor()

    results = [
        FRNNResult(ids=(1, 2), distances=(0.3, 0.05)),
        FRNNResult(ids=(3,), distances=(0.02,)),
    ]

    combined = FRNNResult.merging(results)

    assert set(combined.ids) == set((3, 2, 1))
    assert sorted(combined.distances) == pytest.approx(sorted((0.02, 0.05, 0.3)))


def test_frnn_merge_handles_missing_distances():
    monitor = _build_monitor()

    results = [
        FRNNResult(ids=(4,), distances=None),
        FRNNResult(ids=(5,), distances=(0.1,)),
    ]

    with pytest.raises(RuntimeError):
        FRNNResult.merging(results)


def test_combine_results_all_without_distances():
    monitor = _build_monitor()

    results = [
        FRNNResult(ids=(4,), distances=None),
        FRNNResult(ids=(5,), distances=None),
    ]

    combined = FRNNResult.merging(results)

    assert combined.ids == (4, 5)
    assert combined.distances is None


def test_result_merge_raises_on_duplicate_ids():
    monitor = _build_monitor()

    results = [
        FRNNResult(ids=(1,), distances=(0.1,)),
        FRNNResult(ids=(1,), distances=(0.2,)),
    ]

    with pytest.raises(RuntimeError):
        FRNNResult.merging(results)


def test_monitor_factory_without_parameters():
    """Factory can ignore epsilon/metric if it captures them via closure."""

    def factory():
        return DummyFRNN(epsilon=0.42, metric="linf")

    monitor = Monitor(factory)
    backend = monitor._make_backend()
    assert isinstance(backend, DummyFRNN)
    assert backend._init_epsilon == 0.42


def test_monitor_factory_with_partial_kwargs():
    factory = functools.partial(DummyFRNN, epsilon=0.3, metric="linf")
    monitor = Monitor(factory)
    backend = monitor._make_backend()
    assert backend._init_epsilon == pytest.approx(0.3)


def test_monitor_with_deferred_metric():
    factory = functools.partial(DummyFRNN, epsilon=0.25, metric="linf")
    monitor = Monitor(factory)
    backend = monitor._make_backend()
    assert backend._init_epsilon == pytest.approx(0.25)
