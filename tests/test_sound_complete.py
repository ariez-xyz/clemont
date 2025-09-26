import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from clemont.backends.bdd import BAD_CHARS
except ImportError:  # pragma: no cover - BDD optional dependency missing
    BAD_CHARS = {}


DATA_DIR = Path(__file__).parent
MAX_ROWS_PER_DATASET = 2000
_PAIRWISE_CACHE = {}
BATCHSIZE_OPTIONS = {
    "kdtree": [1, 16, 64, 500],
    "snn": [1, 32, 128, 500],
}


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not BAD_CHARS:
        return df

    rename_map = {}
    for col in df.columns:
        new_col = col
        for bad, replacement in BAD_CHARS.items():
            new_col = new_col.replace(bad, replacement)
        if new_col != col:
            rename_map[col] = new_col
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _limit_rows(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_ROWS_PER_DATASET is None:
        return df
    limit = min(MAX_ROWS_PER_DATASET, len(df))
    return df.head(limit)


def _toy_dataset() -> pd.DataFrame:
    data = [
        {"prediction": 0, "f0": 0.00, "f1": 0.00},
        {"prediction": 1, "f0": 0.05, "f1": 0.05},
        {"prediction": 0, "f0": 0.90, "f1": 0.90},
        {"prediction": 1, "f0": 0.95, "f1": 0.95},
    ]
    df = pd.DataFrame(data, dtype=float)
    df["prediction"] = df["prediction"].astype(int)
    df = _sanitize_columns(df)
    return _limit_rows(df)


def _csv_dataset() -> pd.DataFrame:
    path = DATA_DIR / "testdata0.csv"
    df = pd.read_csv(path, dtype=float)
    df["prediction"] = df["prediction"].astype(int)
    df = _sanitize_columns(df)
    return _limit_rows(df)


def _as_index_set(values):
    if values is None:
        return set()
    if isinstance(values, set):
        return values
    if isinstance(values, (list, tuple, np.ndarray)):
        return {int(v) for v in values}
    return {int(values)}


def _pairwise_cache_key(df: pd.DataFrame, metric: str):
    metric = metric.lower()
    data = df.to_numpy(copy=False)
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    return (
        metric,
        tuple(df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        tuple(df.index.to_list()),
        df.shape,
        digest,
    )


def _get_pairwise(df: pd.DataFrame, metric: str):
    metric = metric.lower()
    key = _pairwise_cache_key(df, metric)
    if key not in _PAIRWISE_CACHE:
        features = df.drop(columns=["prediction"]).to_numpy(copy=False)
        diffs = features[:, None, :] - features[None, :, :]
        if metric == "l2":
            distances = np.linalg.norm(diffs, axis=-1)
        elif metric in {"infinity", "linf"}:
            distances = np.max(np.abs(diffs), axis=-1)
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
        predictions = df["prediction"].to_numpy(copy=False)
        index_values = df.index.to_numpy()
        _PAIRWISE_CACHE[key] = (index_values, predictions, distances)
    return _PAIRWISE_CACHE[key]


def _naive_counterexamples(df: pd.DataFrame, position: int, epsilon: float, metric: str):
    if position == 0:
        return set()

    index_values, predictions, distances = _get_pairwise(df, metric)
    epsilon = float(epsilon)
    tolerance = epsilon + 1e-9

    candidate_mask = predictions[:position] != predictions[position]
    close_mask = distances[position, :position] <= tolerance
    matches = np.where(candidate_mask & close_mask)[0]
    return {int(index_values[i]) for i in matches}


def _build_bruteforce(df: pd.DataFrame, *, batchsize=None):  # batchsize unused
    pytest.importorskip("faiss")
    from clemont.backends.faiss import BruteForce

    epsilon = 0.1
    backend = BruteForce(df.copy(deep=True), decision_col="prediction", epsilon=epsilon, metric="infinity")
    return backend, epsilon, "infinity"


def _build_kdtree(df: pd.DataFrame, *, batchsize=None):
    pytest.importorskip("sklearn.neighbors")
    pytest.importorskip("faiss")
    from clemont.backends.kdtree import KdTree

    epsilon = 0.1
    batchsize = batchsize if batchsize is not None else 1000
    backend = KdTree(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric="infinity",
        batchsize=batchsize,
        bf_threads=1,
    )
    return backend, epsilon, "infinity"


def _build_snn(df: pd.DataFrame, *, batchsize=None):
    pytest.importorskip("faiss")
    pytest.importorskip("snnpy")
    from clemont.backends.snn import Snn

    epsilon = 0.1
    batchsize = batchsize if batchsize is not None else 500
    backend = Snn(
        df.copy(deep=True),
        decision_col="prediction",
        epsilon=epsilon,
        metric="l2",
        batchsize=batchsize,
        bf_threads=1,
    )
    return backend, epsilon, "l2"


def _build_bdd(df: pd.DataFrame, *, batchsize=None):  # batchsize unused
    pytest.importorskip("dd.cudd")
    from clemont.backends.bdd import BDD

    n_bins = 10
    backend = BDD(df.copy(deep=True), n_bins=n_bins, decision_col="prediction")
    epsilon = 1.0 / n_bins
    return backend, epsilon, "infinity"


BACKEND_BUILDERS = [
    ("bruteforce", _build_bruteforce),
    ("kdtree", _build_kdtree),
    ("snn", _build_snn),
    ("bdd", _build_bdd),
]

DATASET_BUILDERS = [
    ("toy", _toy_dataset),
    ("csv", _csv_dataset),
]


@pytest.mark.parametrize("dataset_name,dataset_builder", DATASET_BUILDERS)
@pytest.mark.parametrize("backend_name,builder", BACKEND_BUILDERS)
def test_backends_are_sound_and_complete(dataset_name, dataset_builder, backend_name, builder):
    df = dataset_builder()
    batch_sizes = BATCHSIZE_OPTIONS.get(backend_name, [None])

    for batchsize in batch_sizes:
        backend, configured_epsilon, metric = builder(df, batchsize=batchsize)
        epsilon = float(backend.meta.get("epsilon", configured_epsilon))
        metric = str(backend.meta.get("metric", metric)).lower()
        label = backend_name if batchsize is None else f"{backend_name}(batchsize={batchsize})"

        for position, (idx, row) in enumerate(df.iterrows()):
            expected = _naive_counterexamples(df, position, epsilon, metric)

            observed = backend.observe(row.copy(deep=True), row_id=int(idx))
            observed_set = _as_index_set(observed)

            is_sound = bool(backend.meta.get("is_sound"))
            is_complete = bool(backend.meta.get("is_complete"))

            if is_sound:
                assert observed_set <= expected, (
                    f"{label} on {dataset_name}: produced false positives for row {idx}"
                )

            if is_complete:
                assert expected <= observed_set, (
                    f"{label} on {dataset_name}: missed violations for row {idx}"
                )

            if is_sound and is_complete:
                assert observed_set == expected, (
                    f"{label} on {dataset_name}: unexpected counterexamples for row {idx}"
                )
