from __future__ import annotations

import pandas as pd

from clemont.frnn import KdTreeFRNN
from clemont.monitor import Monitor


def main() -> None:
    df = pd.read_csv("example.csv")
    feature_cols = df.drop(columns=["label", "pred", "p0(<=50K)", "p1(>50K)"])
    preds = df["pred"]

    backend_factory = lambda: KdTreeFRNN(epsilon=0.2, metric="l2")

    monitor = Monitor(backend_factory)

    for index, (point, decision) in enumerate(zip(feature_cols.values, preds.values)):
        result = monitor.observe(point, decision, point_id=index)
        if result.counterexamples.ids:
            print(f"row {index}: violations {list(result.counterexamples.ids)}")

if __name__ == "__main__":
    main()
