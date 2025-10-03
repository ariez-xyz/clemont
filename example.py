"""Minimal usage example for Clemont's FRNN-based monitor."""

import numpy as np

from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN, NaiveFRNN, KdTreeFRNN
import numpy as np

num_rows = 1000
num_columns = 10
epsilon = 0.2

# Generate 1000 10-column datapoints and binary decisions
datapoints = np.random.rand(num_rows, num_columns)
decisions = np.random.choice([0, 1], size=num_rows)

# Pick the desired FRNN backend. FaissFRNN provides high performance when the
# optional faiss dependency is installed. NaiveFRNN works without extras.
# backend_factory = lambda: NaiveFRNN(epsilon=epsilon, metric="linf")
# backend_factory = lambda: KdTreeFRNN(epsilon=epsilon, metric="linf")
backend_factory = lambda: FaissFRNN(epsilon=epsilon, metric="linf", nthreads=4)

monitor = Monitor(backend_factory)

for index, (point, decision) in enumerate(zip(datapoints, decisions)):
    result = monitor.observe(point, decision, point_id=index)
    if result.counterexamples.ids:
        print(f"row {index}: violations {list(result.counterexamples.ids)}")
