import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from clemont.monitor import Monitor
from clemont.frnn import KdTreeFRNN, NaiveFRNN

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
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


def _stream_monitor(df: pd.DataFrame, factory):
    monitor = Monitor(factory)
    for idx, row in df.iterrows():
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])
        result = monitor.observe(point, decision, point_id=int(idx))
        yield result.counterexamples.ids


@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("limit", [256, 1000])
def test_monitor_kdtree_matches_naive_canned(epsilon, limit):
    df = _load_dataframe(limit=limit)

    monitor_kd = _stream_monitor(df, lambda: KdTreeFRNN(epsilon=epsilon, metric="linf"))
    monitor_naive = _stream_monitor(df, lambda: NaiveFRNN(epsilon=epsilon, metric="linf"))

    for kd_ids, naive_ids in zip(monitor_kd, monitor_naive):
        assert set(map(int, kd_ids)) == set(map(int, naive_ids))


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_kdtree_matches_naive_random(epsilon):
    df = _generate_random_dataframe(n_rows=512, n_features=8, seed=123)

    monitor_kd = _stream_monitor(df, lambda: KdTreeFRNN(epsilon=epsilon, metric="linf"))
    monitor_naive = _stream_monitor(df, lambda: NaiveFRNN(epsilon=epsilon, metric="linf"))

    for kd_ids, naive_ids in zip(monitor_kd, monitor_naive):
        assert set(map(int, kd_ids)) == set(map(int, naive_ids))
