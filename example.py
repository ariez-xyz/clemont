"""Minimal usage example for Clemont's FRNN-based monitor."""

import numpy as np
import pandas as pd

import functools

from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN, NaiveFRNN

num_rows = 30000
num_columns = 10
epsilon = 0.2

# Create random data with a prediction column followed by features
column_names = ["prediction"] + [f"f{i}" for i in range(1, num_columns)]
np.random.seed(42)
data = np.random.uniform(0, 1, size=(num_rows, num_columns))
df = pd.DataFrame(data, columns=column_names)
df["prediction"] = (df["prediction"] > 0.75).astype(int)

# Pick the desired FRNN backend. FaissFRNN provides high performance when the
# optional faiss dependency is installed. NaiveFRNN works without extras. You
# can customise backend arguments via ``functools.partial`` or a small factory.
backend_factory = lambda: NaiveFRNN(epsilon=epsilon, metric="linf")
# backend_factory = functools.partial(FaissFRNN, epsilon=epsilon, metric="linf", nthreads=4)
# backend_factory = lambda: FaissFRNN(epsilon=epsilon, metric="linf", nthreads=4)

backend_factory = lambda: FaissFRNN(epsilon=epsilon, metric="linf", nthreads=4)
monitor = Monitor(backend_factory)

# monitor = Monitor(backend_factory=lambda: FaissFRNN(epsilon=epsilon, metric="linf", nthreads=1))

for index, row in df.iterrows():
    decision = int(row["prediction"])
    point = row.drop(labels=["prediction"]).to_numpy(dtype=float)

    result = monitor.observe(point, decision, point_id=index)
    if result.counterexamples.ids:
        print(f"row {index}: violations {list(result.counterexamples.ids)}")
