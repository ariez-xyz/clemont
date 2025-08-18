import faiss
import numpy as np

from clemont.backends.base import BaseBackend
from typing import List

class BruteForce(BaseBackend):
    """
    Brute force monitoring backend.

    Attributes:
        df (pandas.DataFrame): Dataframe holding a sample of the data. Used to infer dimensionality and classes.
        decision_col (str): Name of the column holding the model decision.
        epsilon (float): maximum distance for samples to be considered close
        metric (string): Optional. Metric to use, defaults to 'infinity', possible values listed in BruteForce.METRICS
        nthreads (int): Optional. number of threads to use, defaults to max available
    """

    METRICS = {
        'infinity': faiss.METRIC_Linf,
        'l2': faiss.METRIC_L2,
        'l1': faiss.METRIC_L1,
        'cosine': faiss.METRIC_INNER_PRODUCT,
    }

    def __init__(self, df, decision_col, epsilon, metric='infinity', nthreads=0):
        if metric not in self.METRICS.keys():
            raise NotImplementedError(f"invalid metric {metric}. valid metrics: {list(self.METRICS.keys())}")
        if metric == 'cosine':
            feature_cols = df.drop(columns=[decision_col])
            l2_norms = np.sqrt((feature_cols ** 2).sum(axis=1))
            assert np.allclose(l2_norms, 1.0), "All feature vectors must have unit L2 norm when using cosine metric"

        self.dim = df.shape[1] - 1 # kNN algo is blind to the decision column

        if nthreads > 0:
            faiss.omp_set_num_threads(nthreads)

        # Create separate indices for each unique class
        self.indices = {}
        for class_val in df[decision_col].unique():
            flat_index = faiss.IndexFlat(self.dim, self.METRICS[metric])
            with_custom_ids = faiss.IndexIDMap(flat_index) # This decorator adds support for add_with_ids()
            self.indices[class_val] = with_custom_ids

        self.epsilon = epsilon
        if metric == 'l2': 
            # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_l2
            # "Faiss reports squared Euclidean (L2) distance"
            # Don't ask how long this took to debug
            self.epsilon = epsilon ** 2

        self.decision_col = decision_col
        self.radius_query_ks = []
        self._meta = {
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": metric,
            "is_exact": True,
            "is_sound": True,
            "is_complete": True,
        }

    def index(self, df):
        for decision, group in df.groupby(self.meta["decision_col"]):
            group = group.drop(columns=self.meta["decision_col"])
            self.indices[decision].add_with_ids(group.values, group.index.values)

    def observe(self, row, row_id=None) -> List[int]:
        cexs = []
        row_data = np.array(row.drop(self.decision_col)).reshape(1, -1)

        for decision, idx in self.indices.items():
            # for the index matching the point's decision: skip search, instead add point
            if decision == row[self.decision_col]:
                # pass explicit id - the automatically assigned sequential ids are only unique within each index
                idx.add_with_ids(row_data, np.array([row_id]))
                continue 

            query_fn = lambda k: idx.search(row_data, k)

            if self._meta["metric"] == 'cosine': 
                # need to do (1 - cosine similarity) or epsilon-small values don't imply similarity
                def query_fn(k):
                    dists, ids = idx.search(row_data, k)
                    return 1-dists, ids

            _, indices = self.emulate_range_query(query_fn, self.epsilon)
            cexs.extend(indices)

        return cexs

