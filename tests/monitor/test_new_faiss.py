import pathlib

import pandas as pd
import pytest

pytest.importorskip("faiss")

from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN
from clemont.backends.faiss import BruteForce

DATA_PATH = pathlib.Path(__file__).parent.parent / "testdata0.csv"
LEGACY_METRICS = {"linf": "infinity", "l2": "l2", "l1": "l1"}


def _load_dataframe(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype=float)
    if limit is not None:
        df = df.head(limit)
    df["prediction"] = df["prediction"].astype(int)
    return df


@pytest.mark.parametrize(
    "metric_frnn,epsilon,limit",
    [
        ("linf", 0.1, 256),
        ("linf", 0.5, 1000),
        ("l2", 0.1, 256),
        ("l2", 0.3, 256),
        ("l1", 0.2, 256),
    ],
)
def test_monitor_matches_legacy_faiss_bruteforce(metric_frnn, epsilon, limit):
    metric_legacy = LEGACY_METRICS[metric_frnn]

    df = _load_dataframe(limit=limit)

    legacy_backend = BruteForce(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric=metric_legacy,
    )

    monitor = Monitor(FaissFRNN, epsilon=epsilon, metric=metric_frnn)

    for idx, row in df.iterrows():
        row_id = int(idx)
        point = row.drop(labels=["prediction"]).to_numpy(dtype=float)
        decision = int(row["prediction"])

        legacy_cexs = legacy_backend.observe(row.copy(), row_id=row_id)
        new_result = monitor.observe(point, decision, point_id=row_id)

        legacy_set = {int(x) for x in legacy_cexs}
        assert set(new_result.counterexamples.ids) == legacy_set

