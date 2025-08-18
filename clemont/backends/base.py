from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, cast
import pandas as pd
import numpy as np

class BaseBackend(ABC):
    """Abstract base class for monitoring backends.

    This class serves as a template for different prediction backend implementations.
    All concrete backends must implement the observe() method.

    Parameters
    ----------
    pred : str
        Prediction column.
    epsilon : float
        Closeness parameter

    Attributes
    ----------
    meta : dict, read-only
        Metadata dictionary containing backend information
    """

    def __init__(self, decision_col, epsilon):
        self.radius_query_ks = []
        self._meta = {
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": "Linf",
            "is_exact": False,
            "is_sound": False,
            "is_complete": False,
        }

    @property
    def meta(self):
        """dict: Read-only access to backend metadata."""
        return self._meta

    @abstractmethod
    def index(self, df: pd.DataFrame):
        """Preload the backend with the points in the dataframe (no monitoring)
        """
        pass

    @abstractmethod
    def observe(self, row, row_id=None) -> List[int]:
        """Process an observation through the backend.
        
        Returns
        -------
        List of row_ids that are unfair counterexamples to the parameter row.

        Raises
        ------
        NotImplementedError
            Because this is an abstract method that needs to be implemented
            by concrete backend classes.
        """
        pass

    def preload(self, df, pred, repeat=5):
        np.random.seed(42)
        base_data = df.drop(columns=pred).to_numpy()
        chunks = []
        for _ in range(repeat):
            noise = np.random.normal(0, 0.1, base_data.shape)
            noisy_data = np.clip(base_data + noise, 0, 1)
            chunks.append(noisy_data)
        augmented_data = np.concatenate(chunks)
        # Create augmented dataframe with repeated predictions
        augmented_df = pd.DataFrame(augmented_data, columns=df.drop(columns=pred).columns)
        augmented_df[pred] = pd.concat([df[pred]] * repeat).reset_index(drop=True)
        self.index(augmented_df)
        return augmented_df.shape

    def emulate_range_query(
        self,
        query_fn: Callable[[int], Tuple[List[List[float]], List[List[int]]]],
        epsilon: float,
    ) -> Tuple[List[float], List[int]]:
        distances = [[]]
        indices = [[]]
        k = 4
        max_k = 128  # Limit to avoid excessive iterations

        # Iteratively run kNN queries, doubling k until results outside epsilon range are returned
        while k <= max_k:
            distances, indices = query_fn(k)

            if len(distances[0]) == 0:
                self.radius_query_ks += [0]
                return [], []
            elif all(d < 0 for d in distances[0]):
                # faiss returns -inf if fewer than k points in index -- early break avoids looping to max_k
                break
            elif all(d < epsilon for d in distances[0]):
                k *= 2
            else:
                break

        if k > max_k:
            self.radius_query_ks += [-1] # Indicate that query exceeds max_k
        else:
            self.radius_query_ks += [k]

        # Filter results outside epsilon range
        valid_points = [i for i, d in enumerate(distances[0]) if 0 <= d < epsilon]
        filtered_distances = [float(distances[0][i]) for i in valid_points]
        filtered_indices = [int(indices[0][i]) for i in valid_points]

        return filtered_distances, filtered_indices

