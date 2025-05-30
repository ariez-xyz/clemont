import time

from sklearn.neighbors import KDTree

from clemont.backends.base import BaseBackend
from clemont.backends.faiss import BruteForce

class KdTree(BaseBackend):
    """
    KdTree monitoring backend.

    Attributes:
        df (pandas.DataFrame): Dataframe holding a sample of the data. Used to infer number of classes.
        decision_col (str): Name of the column holding the model decision.
        epsilon (float): maximum distance for samples to be considered close
        metric (string): Optional. Metric to use, defaults to 'infinity', possible values are given in sklearn.neighbors.KDTree.valid_metrics
        batchsize (int): Optional, defaults to 1000. Size of short term memory (how many samples to process until reindex)
        bf_threads (int): Optional, defaults to 1. Number of threads to use for brute force short term memory.
    """
    def __init__(self, df, decision_col, epsilon, metric='infinity', batchsize=1000, bf_threads=1):
        if metric not in KDTree([[0]]).valid_metrics:
            raise NotImplementedError(f"invalid metric {metric}. valid metrics: {KDTree([[0]]).valid_metrics}")
        if epsilon > 0.1 and metric == 'l2':
            print("WARNING: Large epsilon values in L2 metric may produce issues with the prediction (a differing prediction may be epsilon-close)")

        self.classes = df[decision_col].unique()
        self.df = df
        self.batchsize = batchsize

        self.current_batch = 0
        self.bf = BruteForce(df, decision_col, epsilon, metric, bf_threads)
        self.history = []
        self.histories = {c: [] for c in self.classes}

        self._meta = {
            "kdt_time": 0,
            "bf_time": 0,
            "index_time": 0,
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": metric,
            "batchsize": batchsize,
            "bf_threads": bf_threads,
            "is_exact": True,
            "is_sound": True,
            "is_complete": True,
        }

    def index(self, df):
        self.kdt = KDTree(df, metric=self.meta["metric"])

    def observe(self, row, row_id=None):
        if len(self.history) < self.batchsize:
            st = time.time()
            cexs = self.bf.observe(row, row_id)
            self.history.append(row)
            self.meta["bf_time"] += time.time() - st
            self.current_batch += 1
            return cexs

        decision = row[self.meta["decision_col"]]

        if self.current_batch >= self.batchsize: # Rebuild
            print(f"rebuilding at {len(self.history)}...", end='\r')
            st = time.time()
            self.kdt = KDTree(self.history, metric=self.meta["metric"])
            self.bf = BruteForce(self.df, self.meta["decision_col"], self.meta["epsilon"], self.meta["metric"], self.meta["bf_threads"])
            self.meta["index_time"] += time.time() - st
            self.current_batch = 0

        # First identify close points within the current batch using brute force
        st = time.time()
        cexs = self.bf.observe(row, row_id)
        self.meta["bf_time"] += time.time() - st

        # Now query the previous batches which are stored in the kd-tree
        for c in self.classes:
            # For each possible decision class, flip the current row's decision to that class
            # in order to find epsilon-close points with that (different) decision.
            # TODO: Fix for large epsilon values: after overriding the decision, the search may return points close to the *original* point (that is, they have a matching prediction, but are still epsilon-close)
            if c == decision:
                continue # skip search for points with same decision
            row[self.meta["decision_col"]] = c
            st = time.time()
            kdt_res = self.kdt.query_radius([row], self.meta["epsilon"])
            self.meta["kdt_time"] += time.time() - st
            cexs.extend(list(kdt_res[0]))

        row[self.meta["decision_col"]] = decision # Restore point to correct decision.

        self.history.append(row)
        self.current_batch += 1

        return cexs

