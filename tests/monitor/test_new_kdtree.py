import pathlib

import numpy as np
import pandas as pd
import pytest
import typing

pytest.importorskip("sklearn")

from clemont.monitor import Monitor
from clemont.frnn import KdTreeFRNN, NaiveFRNN
from clemont.backends.kdtree import KdTree

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
EPSILONS = [round(0.1 * i, 1) for i in range(1, 6, 2)]
TOL = 1e-6


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


def _distance(df: pd.DataFrame, i: int, j: int, metric: str) -> float:
    """Calculate distance between two rows in a DataFrame using specified metric."""
    # Extract rows and exclude the decision column
    row_i = df.iloc[i].drop("prediction")
    row_j = df.iloc[j].drop("prediction")
    
    if metric == "linf": return np.max(np.abs(row_i - row_j))
    elif metric == "l2": return np.sqrt(np.sum((row_i - row_j) ** 2))
    elif metric == "l1": return np.sum(np.abs(row_i - row_j))
    else: raise ValueError(f"Unsupported metric: {metric}")


def _errmsg_mismatch(df, real_dist, epsilon, row, point, test_set, validation_set):
    # Create formatted table for row and point comparison
    table_lines = ["col        row        point      diff"]
    for colname in df.columns:
        row_val = df.iloc[row][colname]
        point_val = df.iloc[point][colname]
        diff = abs(row_val - point_val)
        colname = colname[:10].ljust(10)
        table_lines.append(f"{str(colname):<10} {row_val:>10.5f} {point_val:>10.5f} {diff:>10.5f}")
    
    table_str = "\n".join(table_lines)
    
    return f"FRNN backend returned neighbors row={row}, point={point} with distance {real_dist} for epsilon={epsilon}: test={test_set} validation={validation_set}.\n{table_str}"


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("limit", [256, 1000])
def test_monitor_kdtree_matches_naive_canned(metric_frnn, epsilon, limit):
    df = _load_dataframe(limit=limit)

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    monitor_naive = Monitor(lambda: NaiveFRNN(epsilon=epsilon, metric=metric_frnn))

    for idx, row in df.iterrows():
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        naive_result = monitor_naive.observe(point, decision, point_id=int(idx))

        test_set = set(map(int, kd_result.counterexamples.ids))
        validation_set = set(map(int, naive_result.counterexamples.ids))

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)

@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_kdtree_matches_naive_random(metric_frnn, epsilon):
    df = _generate_random_dataframe(n_rows=512, n_features=8, seed=123)

    monitor_kd = Monitor(lambda: KdTreeFRNN(epsilon=epsilon, metric=metric_frnn))
    monitor_naive = Monitor(lambda: NaiveFRNN(epsilon=epsilon, metric=metric_frnn))

    for idx, row in df.iterrows():
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        naive_result = monitor_naive.observe(point, decision, point_id=int(idx))

        test_set = set(map(int, kd_result.counterexamples.ids))
        validation_set = set(map(int, naive_result.counterexamples.ids))

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)


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
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        legacy_res = legacy.observe(row.copy(), row_id=int(idx))

        test_set = set(map(int, kd_result.counterexamples.ids))
        validation_set = set(map(int, legacy_res))

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)


@pytest.mark.parametrize("metric_frnn", ["linf", "l2", "l1"])
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_monitor_kdtree_matches_legacy_random(metric_frnn, epsilon):
    df = _generate_random_dataframe(n_rows=512, n_features=4, seed=321)

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
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        kd_result = monitor_kd.observe(point, decision, point_id=int(idx))
        legacy_res = legacy.observe(row.copy(), row_id=int(idx))

        test_set = set(map(int, kd_result.counterexamples.ids))
        validation_set = set(map(int, legacy_res))

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)
