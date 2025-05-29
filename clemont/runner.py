import numpy as np
import time
import psutil

class DataframeRunner:
    """Class to run the Clemont monitoring tool on a pandas dataframe as input data and take performance measurements.

    This class manages execution metrics such as timing and accuracy by tracking positives,
    true positives, and timing information during monitoring runs.
    """
    def __init__(self, backend):
        self.backend = backend
        self.n_flagged = 0
        self.n_positives = 0
        self.n_true_positives = 0
        self.timings = []
        self.mem = []
        self.total_time = 0
        self.data_shape = None

    def get_backend_name(self):
        return self.backend.__class__.__name__

    def run(self, df, max_n=None, max_time=None):
        self.data_shape = df.shape
        start_time = time.time()
        process = psutil.Process() # Memory tracking

        for index, row in df.iterrows():
            start_iter_time = time.time()

            if max_n is not None and index >= max_n:
                break
            if max_time is not None and start_iter_time >= start_time + max_time:
                break

            iter_cexs = self.backend.observe(row, row_id=index)

            self.n_positives += len(iter_cexs)

            # For unsound backends (BDD) filter false positives.
            if not self.backend.meta['is_sound']:
                if self.backend.meta['metric']!= "Linf":
                    raise ValueError("false positive filtering is currently only implemented for Linf distance")
                pred = self.backend.meta['decision_col']
                eps = self.backend.meta['epsilon']
                iter_cexs = self.filter_false_positives_Linf(iter_cexs, row, df, pred, eps)

            # Count remaining true positives after exact post-verification
            self.n_true_positives += len(iter_cexs)
            if len(iter_cexs) > 0:
                self.n_flagged += 1

            # Performance measurements
            iter_time = time.time() - start_iter_time
            self.timings.append(iter_time)
            self.mem.append(process.memory_info().rss)
            self.total_time += iter_time

            yield [(cex, index) for cex in iter_cexs]

    def filter_false_positives_Linf(self, cexs, row, df, pred, eps):
        nofp = [cex for cex in cexs if np.all(np.abs(df.loc[cex].drop(pred) - row.drop(pred)) < eps)]
        return nofp
