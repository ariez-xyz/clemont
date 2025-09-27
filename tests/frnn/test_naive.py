import numpy as np
import pytest

from clemont.frnn import NaiveFRNN


def test_naive_frnn_linf_radius_handling_and_properties():
    backend = NaiveFRNN(epsilon=0.5, metric="linf")
    backend.add((0.0, 0.0), point_id=42)

    assert backend.epsilon == pytest.approx(0.5)
    assert backend.metric == "linf"
    assert backend.is_sound
    assert backend.is_complete

    default_result = backend.query((0.1, 0.1))
    assert default_result.ids == (42,)
    assert default_result.distances == pytest.approx((0.1,))

    override_result = backend.query((0.4, 0.4), radius=0.3)
    assert override_result.ids == ()

    with pytest.raises(ValueError):
        backend.query((0.0, 0.0), radius=-1.0)


def test_naive_frnn_l2_support():
    backend = NaiveFRNN(epsilon=0.25, metric="l2")
    for idx, point in enumerate([(0.0, 0.0), (0.2, 0.2), (0.3, 0.3)]):
        backend.add(point, point_id=idx)

    result = backend.query((0.1, 0.1))
    assert result.ids == (0, 1)

    distances = np.array(result.distances)
    expected = np.array([np.sqrt(0.02), np.sqrt(0.02)])
    assert np.allclose(distances, expected)


def test_naive_frnn_l1_support():
    backend = NaiveFRNN(epsilon=0.3, metric="l1")
    backend.add((0.0, 0.0), point_id=1)
    backend.add((0.2, 0.1), point_id=2)

    result = backend.query((0.1, 0.05))
    assert set(result.ids) == {1, 2}


def test_supported_metrics_reported_by_naive_backend():
    assert set(NaiveFRNN.supported_metrics()) == {"linf", "l2", "l1"}

