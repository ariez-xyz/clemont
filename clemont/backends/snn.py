import time

import snnpy
import numpy as np

from clemont.backends.base import BaseBackend
from clemont.backends.faiss import BruteForce

try:
    import snnpy
    _HAS_SNNPY = True
except ImportError:
    _HAS_SNNPY = False
    snnpy = None

class Snn(BaseBackend):
    def __init__(self, df, decision_col, epsilon, metric='l2', batchsize=500, bf_threads=1):
        if not _HAS_SNNPY:
            raise ImportError(
                "\n" + "="*60 + "\n"
                "SNN backend requires 'snn' package binding\n"
                "This cannot be installed automatically via pip.\n\n"
                "Please see the installation instructions in the README.\n"
                "="*60
            )

        if metric != 'l2':
            raise NotImplementedError(f"invalid metric {metric}. snnpy only supports l2")
        if epsilon > 0.1:
            print("WARNING: Large epsilon values in L2 metric may produce issues with the prediction (a differing prediction may be epsilon-close)")

        self.classes = df[decision_col].unique()
        self.df = df
        self.batchsize = batchsize

        self.current_batch = 0
        self.bf = BruteForce(df, decision_col, epsilon, 'l2', bf_threads)
        self.history = []
        self.histories = {c: [] for c in self.classes}

        self._meta = {
            "snn_time": 0,
            "bf_time": 0,
            "index_time": 0,
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": "l2",
            "batchsize": batchsize,
            "bf_threads": bf_threads,
            "is_exact": True,
            "is_sound": True,
            "is_complete": True,
        }

    def index(self, df):
        self.snn = snnpy.build_snn_model(df.to_numpy())

    def observe(self, row, row_id=None):
        if len(self.history) < self.batchsize:
            self.history.append(row)
            self.current_batch += 1
            return self.bf.observe(row, row_id)

        decision = row[self.meta["decision_col"]]

        if self.current_batch >= self.batchsize: # Rebuild
            print(f"rebuilding at {len(self.history)}...")
            X = np.array([series.values for series in self.history])
            st = time.time()
            self.snn = snnpy.build_snn_model(X)
            self.bf = BruteForce(self.df, self.meta["decision_col"], self.meta["epsilon"], "l2", self.meta["bf_threads"])
            self.meta["index_time"] += time.time() - st
            self.current_batch = 0

        # First identify close points within the current batch using brute force
        st = time.time()
        cexs = self.bf.observe(row, row_id)
        self.meta["bf_time"] += time.time() - st

        # Now query the previous batches which are stored in the SNN index
        for c in self.classes:
            # For each possible decision class, flip the current row's decision to that class
            # in order to find epsilon-close points with that (different) decision.
            # TODO: Fix for large epsilon values: after overriding the decision, the search may return points close to the *original* point (that is, they have a matching prediction, but are still epsilon-close)
            if c == decision:
                continue # skip search for points with same decision
            row[self.meta["decision_col"]] = c
            st = time.time()
            snn_res = self.snn.query_radius(row, self.meta["epsilon"], return_distance=False)
            self.meta["snn_time"] += time.time() - st
            cexs.extend(snn_res)

        row[self.meta["decision_col"]] = decision # Restore point to correct decision.

        self.history.append(row)
        self.current_batch += 1

        return cexs

