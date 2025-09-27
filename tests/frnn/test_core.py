import pytest

from clemont.frnn import FRNNBackend, FRNNResult
from clemont.frnn.metrics import canonical_metrics, ensure_canonical_metric


def test_frnn_result_matching_lengths():
    result = FRNNResult.from_iterables([1, 2], [0.1, 0.2])
    assert result.ids == (1, 2)
    assert result.distances == (0.1, 0.2)


def test_frnn_result_length_mismatch_raises():
    with pytest.raises(ValueError):
        FRNNResult(ids=(1,), distances=(0.2, 0.3))


def test_frnn_backend_rejects_non_positive_epsilon():
    class Broken(FRNNBackend):
        @classmethod
        def supported_metrics(cls):
            return ("linf",)

        def add(self, point, point_id):  # pragma: no cover - never called
            raise NotImplementedError

        def query(self, point, *, radius=None):  # pragma: no cover - never called
            raise NotImplementedError

    with pytest.raises(ValueError):
        Broken(epsilon=0, metric="linf", is_sound=True, is_complete=True)


def test_metric_normalization_strictness():
    assert ensure_canonical_metric("linf") == "linf"
    assert ensure_canonical_metric("L2") == "l2"
    assert "linf" in canonical_metrics()

    with pytest.raises(ValueError):
        ensure_canonical_metric("infinity")

    with pytest.raises(TypeError):
        ensure_canonical_metric(123)  # type: ignore[arg-type]

