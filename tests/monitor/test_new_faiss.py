import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN
from clemont.backends.faiss import BruteForce

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
LEGACY_METRICS = {"linf": "infinity", "l2": "l2", "l1": "l1"}
EPSILONS = [round(0.1 * i, 1) for i in range(1, 6)]


def _load_dataframe(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=float)
    if limit is not None:
        df = df.head(limit)
    df["prediction"] = df["prediction"].astype(int)
    return df


def _generate_random_dataframe(n_rows: int, n_features: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.random((n_rows, n_features))
    predictions = rng.integers(0, 2, size=n_rows)
    columns = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "prediction", predictions)
    return df


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("limit", [256, 1000])
def test_monitor_matches_legacy_faiss_bruteforce(metric_frnn, epsilon, limit):
    metric_legacy = LEGACY_METRICS[metric_frnn]

    df = _load_dataframe(limit=limit)

    legacy_backend = BruteForce(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric=metric_legacy,
    )

    factory = lambda epsilon=epsilon, metric_frnn=metric_frnn: FaissFRNN(
        epsilon=epsilon,
        metric=metric_frnn,
    )
    monitor = Monitor(factory)

    for idx, row in df.iterrows():
        row_id = int(idx)
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        legacy_cexs = legacy_backend.observe(row.copy(), row_id=row_id)
        new_result = monitor.observe(point, decision, point_id=row_id)

        legacy_set = {int(x) for x in legacy_cexs}
        assert set(new_result.counterexamples.ids) == legacy_set


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_matches_legacy_random_dataset(metric_frnn, epsilon):
    metric_legacy = LEGACY_METRICS[metric_frnn]

    df = _generate_random_dataframe(n_rows=1000, n_features=10, seed=42)

    legacy_backend = BruteForce(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric=metric_legacy,
    )

    factory = lambda epsilon=epsilon, metric_frnn=metric_frnn: FaissFRNN(
        epsilon=epsilon,
        metric=metric_frnn,
    )
    monitor = Monitor(factory)

    for idx, row in df.iterrows():
        row_id = int(idx)
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        legacy_cexs = legacy_backend.observe(row.copy(), row_id=row_id)
        new_result = monitor.observe(point, decision, point_id=row_id)

        legacy_set = {int(x) for x in legacy_cexs}
        assert set(new_result.counterexamples.ids) == legacy_set
