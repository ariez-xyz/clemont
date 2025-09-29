import pathlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from clemont.monitor import Monitor
from clemont.frnn import KdTreeFRNN, NaiveFRNN
from clemont.backends.kdtree import KdTree

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
EPSILONS = [round(0.1 * i, 1) for i in range(1, 6)]
TOL = 1e-8


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


def _distance(df: pd.DataFrame, i: int, j: int) -> float:
    point_i = df.loc[i].drop(labels=["prediction"]).to_numpy(dtype=float)
    point_j = df.loc[j].drop(labels=["prediction"]).to_numpy(dtype=float)
    return float(np.max(np.abs(point_i - point_j)))


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("limit", [256, 1000])
def test_monitor_kdtree_matches_naive_canned(metric_frnn, epsilon, limit):
    df = _load_dataframe(limit=limit)

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    monitor_naive = Monitor(lambda: NaiveFRNN(epsilon=epsilon, metric=metric_frnn))

    for idx, row in df.iterrows():
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        naive_result = monitor_naive.observe(point, decision, point_id=int(idx))

        kd_set = set(map(int, kd_result.counterexamples.ids))
        naive_set = set(map(int, naive_result.counterexamples.ids))

        if kd_set != naive_set:
            diff_ok = True
            for mid in kd_set.symmetric_difference(naive_set):
                dist = _distance(df, idx, mid)
                if not (abs(dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision):
                    diff_ok = False
                    break
            assert diff_ok, (
                f"epsilon={epsilon} row={idx}: kd={kd_set} naive={naive_set}"
            )


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_kdtree_matches_naive_random(metric_frnn, epsilon):
    df = _generate_random_dataframe(n_rows=512, n_features=8, seed=123)

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    monitor_naive = Monitor(lambda: NaiveFRNN(epsilon=epsilon, metric=metric_frnn))

    for idx, row in df.iterrows():
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        naive_result = monitor_naive.observe(point, decision, point_id=int(idx))

        kd_set = set(map(int, kd_result.counterexamples.ids))
        naive_set = set(map(int, naive_result.counterexamples.ids))

        if kd_set != naive_set:
            diff_ok = True
            for mid in kd_set.symmetric_difference(naive_set):
                dist = _distance(df, idx, mid)
                if not (abs(dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision):
                    diff_ok = False
                    break
            assert diff_ok, (
                f"epsilon={epsilon} row={idx}: kd={kd_set} naive={naive_set}"
            )


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("limit", [256, 1000])
def test_monitor_kdtree_matches_legacy_canned(metric_frnn, epsilon, limit):
    df = _load_dataframe(limit=limit)

    legacy_metric_map = {
        "linf": "infinity",
        "l2": "l2",
        "l1": "l1",
    }

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    legacy = KdTree(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric=legacy_metric_map[metric_frnn],
        batchsize=128,
        bf_threads=1,
    )

    for idx, row in df.iterrows():
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        legacy_res = legacy.observe(row.copy(), row_id=int(idx))

        kd_set = set(map(int, kd_result.counterexamples.ids))
        legacy_set = set(map(int, legacy_res))

        if kd_set != legacy_set:
            diff_ok = True
            for mid in kd_set.symmetric_difference(legacy_set):
                dist = _distance(df, idx, mid)
                if not (abs(dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision):
                    diff_ok = False
                    break
            assert diff_ok, (
                f"epsilon={epsilon} row={idx}: kd={kd_set} legacy={legacy_set}"
            )


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_kdtree_matches_legacy_random(metric_frnn, epsilon):
    df = _generate_random_dataframe(n_rows=512, n_features=8, seed=321)

    legacy_metric_map = {
        "linf": "infinity",
        "l2": "l2",
        "l1": "l1",
    }

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    legacy = KdTree(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric=legacy_metric_map[metric_frnn],
        batchsize=128,
        bf_threads=1,
    )

    for idx, row in df.iterrows():
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        legacy_res = legacy.observe(row.copy(), row_id=int(idx))
        print(kd_result, legacy_res)

        kd_set = set(map(int, kd_result.counterexamples.ids))
        legacy_set = set(map(int, legacy_res))

        if kd_set != legacy_set:
            diff_ok = True
            for mid in kd_set.symmetric_difference(legacy_set):
                dist = _distance(df, idx, mid)
                if not (abs(dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision):
                    diff_ok = False
                    break
            assert diff_ok, (
                f"epsilon={epsilon} row={idx}: kd={kd_set} legacy={legacy_set}"
            )

