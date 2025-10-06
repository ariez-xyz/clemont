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


def test_frnn_backend_knn_default_not_supported():
    class NoKnn(FRNNBackend):
        @classmethod
        def supported_metrics(cls):
            return ("linf",)

        def add(self, point, point_id):  # pragma: no cover - unused
            raise NotImplementedError

        def query(self, point, *, radius=None):  # pragma: no cover - unused
            return FRNNResult(ids=())

    backend = NoKnn(epsilon=0.1, metric="linf", is_sound=True, is_complete=True)

    assert backend.supports_knn is False

    with pytest.raises(NotImplementedError):
        backend.query_knn((0.0,), k=1)


def test_metric_normalization_strictness():
    assert ensure_canonical_metric("linf") == "linf"
    assert ensure_canonical_metric("L2") == "l2"
    assert "linf" in canonical_metrics()

    with pytest.raises(ValueError):
        ensure_canonical_metric("infinity")

    with pytest.raises(TypeError):
        ensure_canonical_metric(123)  # type: ignore[arg-type]


def test_merging_empty_results():
    """Test merging an empty iterable returns empty result."""
    result = FRNNResult.merging([])
    assert result.ids == ()
    assert result.distances is None


def test_merging_single_result():
    """Test merging a single result returns that result."""
    original = FRNNResult(ids=(1, 2), distances=(0.1, 0.2))
    merged = FRNNResult.merging([original])
    assert merged.ids == (1, 2)
    assert merged.distances == (0.1, 0.2)


def test_merging_multiple_results_with_distances():
    """Test merging multiple results with distances."""
    result1 = FRNNResult(ids=(1, 2), distances=(0.1, 0.2))
    result2 = FRNNResult(ids=(3, 4), distances=(0.3, 0.4))
    merged = FRNNResult.merging([result1, result2])
    assert set(merged.ids) == {1, 2, 3, 4}
    assert merged.distances is not None
    assert len(merged.distances) == 4


def test_merging_multiple_results_without_distances():
    """Test merging multiple results without distances."""
    result1 = FRNNResult(ids=(1, 2))
    result2 = FRNNResult(ids=(3, 4))
    merged = FRNNResult.merging([result1, result2])
    assert set(merged.ids) == {1, 2, 3, 4}
    assert merged.distances is None


def test_merging_with_empty_results():
    """Test merging with some empty results included."""
    result1 = FRNNResult(ids=(1, 2), distances=(0.1, 0.2))
    empty_result = FRNNResult(ids=())
    result2 = FRNNResult(ids=(3, 4), distances=(0.3, 0.4))
    merged = FRNNResult.merging([result1, empty_result, result2])
    assert set(merged.ids) == {1, 2, 3, 4}
    assert merged.distances is not None
    assert len(merged.distances) == 4


def test_merging_duplicate_ids_raises():
    """Test that merging results with duplicate IDs raises RuntimeError."""
    result1 = FRNNResult(ids=(1, 2), distances=(0.1, 0.2))
    result2 = FRNNResult(ids=(2, 3), distances=(0.3, 0.4))
    with pytest.raises(RuntimeError, match="duplicate ID 2"):
        FRNNResult.merging([result1, result2])


def test_merging_inconsistent_distance_availability_raises():
    """Test that merging results with inconsistent distance availability raises RuntimeError."""
    result_with_distances = FRNNResult(ids=(1, 2), distances=(0.1, 0.2))
    result_without_distances = FRNNResult(ids=(3, 4))
    with pytest.raises(RuntimeError, match="inconsistent distance availability"):
        FRNNResult.merging([result_with_distances, result_without_distances])


def test_merging_preserves_sorting():
    """Test that merged results are properly sorted by distance."""
    result1 = FRNNResult(ids=(1, 2), distances=(0.2, 0.1))
    result2 = FRNNResult(ids=(3, 4), distances=(0.4, 0.3))
    merged = FRNNResult.merging([result1, result2])
    # Results should be sorted by distance in ascending order
    assert merged.ids == (2, 1, 4, 3)
    assert merged.distances == (0.1, 0.2, 0.3, 0.4)

def test_frnnresult_top_k():
    result = FRNNResult.from_iterables(
        ids=[3, 1, 2],
        distances=[0.3, 0.1, 0.2],
    )

    # __post_init__ sorts by distance
    assert result.ids == (1, 2, 3)

    top_two = result.top_k(2)
    assert top_two.ids == (1, 2)
    assert top_two.distances == (0.1, 0.2)

    same = result.top_k(10)
    assert same.ids == result.ids

    empty = result.top_k(0)
    assert empty.ids == ()

    via_classmethod = FRNNResult.nearest(1, result)
    assert via_classmethod.ids == (1,)
