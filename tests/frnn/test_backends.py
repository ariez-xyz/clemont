import pathlib

import numpy as np
import pandas as pd
import pytest

from clemont.frnn import NaiveFRNN, KdTreeFRNN

try:  # Optional dependency
    from clemont.frnn import FaissFRNN
except ImportError:  # pragma: no cover - import-time failure already handled by module
    FaissFRNN = None  # type: ignore

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
DEFAULT_EPSILON = {"linf": 0.1, "l2": 0.05, "l1": 0.1}

BACKEND_CLASSES = [KdTreeFRNN]
if FaissFRNN is not None:
    BACKEND_CLASSES.append(FaissFRNN)


def _load_points(limit: int | None = 128):
    df = pd.read_csv(DATA_PATH, dtype=float)
    features = df.drop(columns=["prediction"]).to_numpy(dtype=float)
    if limit is not None:
        return features[:limit]
    return features


def _result_to_dict(result):
    if result.distances is None:
        raise AssertionError("Distances should be populated for comparison")
    return dict(zip(result.ids, result.distances))


@pytest.mark.parametrize("backend_cls", BACKEND_CLASSES)
def test_backends_match_naive(backend_cls):
    metrics = sorted(
        set(NaiveFRNN.supported_metrics()) & set(backend_cls.supported_metrics())
    )
    if not metrics:
        pytest.skip("backend shares no metrics with NaiveFRNN")

    points = _load_points()

    for metric in metrics:
        epsilon = DEFAULT_EPSILON.get(metric, 0.1)
        naive = NaiveFRNN(epsilon=epsilon, metric=metric)
        backend = backend_cls(epsilon=epsilon, metric=metric)

        for idx, point in enumerate(points):
            naive.add(point, point_id=idx)
            backend.add(point, point_id=idx)

        for point in points:
            expected = naive.query(point)
            actual = backend.query(point)

            assert set(actual.ids) == set(expected.ids)

            expected_map = _result_to_dict(expected)
            actual_map = _result_to_dict(actual)

            for pid, exp_dist in expected_map.items():
                assert pid in actual_map, f"missing neighbour {pid} for metric {metric}"
                assert np.isclose(actual_map[pid], exp_dist, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend_cls", BACKEND_CLASSES)
def test_backends_respect_radius_override(backend_cls):
    metrics = sorted(
        set(NaiveFRNN.supported_metrics()) & set(backend_cls.supported_metrics())
    )
    if not metrics:
        pytest.skip("backend shares no metrics with NaiveFRNN")

    points = _load_points(limit=32)

    for metric in metrics:
        epsilon = DEFAULT_EPSILON.get(metric, 0.1)
        override = max(epsilon / 2, epsilon * 0.1)

        naive = NaiveFRNN(epsilon=epsilon, metric=metric)
        backend = backend_cls(epsilon=epsilon, metric=metric)

        for idx, point in enumerate(points):
            naive.add(point, point_id=idx)
            backend.add(point, point_id=idx)

        for point in points:
            expected = naive.query(point, radius=override)
            actual = backend.query(point, radius=override)

            assert set(actual.ids) == set(expected.ids)

            expected_map = _result_to_dict(expected)
            actual_map = _result_to_dict(actual)

            for pid, exp_dist in expected_map.items():
                assert pid in actual_map
                assert np.isclose(actual_map[pid], exp_dist, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend_cls", BACKEND_CLASSES)
def test_backends_report_knn_support(backend_cls):
    metrics = sorted(
        set(NaiveFRNN.supported_metrics()) & set(backend_cls.supported_metrics())
    )
    if not metrics:
        pytest.skip("backend shares no metrics with NaiveFRNN")

    epsilon = DEFAULT_EPSILON.get(metrics[0], 0.1)
    backend = backend_cls(epsilon=epsilon, metric=metrics[0])

    assert backend.supports_knn


@pytest.mark.parametrize("backend_cls", BACKEND_CLASSES)
def test_backends_reject_invalid_k(backend_cls):
    metrics = sorted(
        set(NaiveFRNN.supported_metrics()) & set(backend_cls.supported_metrics())
    )
    if not metrics:
        pytest.skip("backend shares no metrics with NaiveFRNN")

    epsilon = DEFAULT_EPSILON.get(metrics[0], 0.1)
    backend = backend_cls(epsilon=epsilon, metric=metrics[0])

    with pytest.raises(ValueError):
        backend.query_knn((0.0,), k=0)


@pytest.mark.parametrize("backend_cls", BACKEND_CLASSES)
def test_backends_knn_matches_naive(backend_cls):
    print(backend_cls)
    metrics = sorted(
        set(NaiveFRNN.supported_metrics()) & set(backend_cls.supported_metrics())
    )
    if not metrics:
        pytest.skip("backend shares no metrics with NaiveFRNN")

    points = _load_points(limit=64)
    k = 5

    for metric in metrics:
        epsilon = DEFAULT_EPSILON.get(metric, 0.1)
        naive = NaiveFRNN(epsilon=epsilon, metric=metric)
        backend = backend_cls(epsilon=epsilon, metric=metric)

        for idx, point in enumerate(points):
            naive.add(point, point_id=idx)
            backend.add(point, point_id=idx)

        sample_points = points[: min(len(points), 8)]

        for point in sample_points:
            expected = naive.query_knn(point, k=k)
            actual = backend.query_knn(point, k=k)

            assert len(actual.ids) <= k
            assert set(actual.ids) == set(expected.ids)

            if expected.ids:
                expected_map = _result_to_dict(expected)
                actual_map = _result_to_dict(actual)
                for pid, exp_dist in expected_map.items():
                    assert pid in actual_map
                    assert np.isclose(actual_map[pid], exp_dist, rtol=1e-5, atol=1e-6)

            override = max(epsilon / 2, epsilon * 0.1)
            expected_limited = naive.query_knn(point, k=k, radius=override)
            actual_limited = backend.query_knn(point, k=k, radius=override)

            assert len(actual_limited.ids) <= k
            assert set(actual_limited.ids) == set(expected_limited.ids)

            if expected_limited.ids:
                expected_map = _result_to_dict(expected_limited)
                actual_map = _result_to_dict(actual_limited)
                for pid, exp_dist in expected_map.items():
                    assert pid in actual_map
                    assert np.isclose(actual_map[pid], exp_dist, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(FaissFRNN is None, reason="FAISS backend unavailable")
def test_faiss_rejects_non_unit_vectors_for_cosine():
    backend = FaissFRNN(epsilon=0.1, metric="cosine")

    backend.add([1.0, 0.0], point_id=0)

    with pytest.raises(ValueError):
        backend.add([0.5, 0.5], point_id=1)

    with pytest.raises(ValueError):
        backend.query([0.5, 0.5])
