import pathlib

import numpy as np
import pandas as pd
import pytest
import typing

pytest.importorskip("faiss")

from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN
from clemont.backends.faiss import BruteForce

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
LEGACY_METRICS = {"linf": "infinity", "l2": "l2", "l1": "l1"}
EPSILONS = [round(0.1 * i, 1) for i in range(1, 6)]
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
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        legacy_cexs = legacy_backend.observe(row.copy(), row_id=idx)
        new_result = monitor.observe(point, decision, point_id=idx)

        test_set = set(new_result.counterexamples.ids)
        validation_set = {int(x) for x in legacy_cexs}

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)



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
        idx = typing.cast(int, idx) # for type checker
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        legacy_cexs = legacy_backend.observe(row.copy(), row_id=idx)
        new_result = monitor.observe(point, decision, point_id=idx)

        test_set = set(new_result.counterexamples.ids)
        validation_set = {int(x) for x in legacy_cexs}

        if test_set != validation_set:
            for mid in test_set.symmetric_difference(validation_set):
                real_dist = _distance(df, idx, mid, metric_frnn)
                diff_ok = (abs(real_dist - epsilon) <= TOL and df.loc[mid, "prediction"] != decision)
                assert diff_ok, _errmsg_mismatch(df, real_dist, epsilon, idx, mid, test_set, validation_set)
