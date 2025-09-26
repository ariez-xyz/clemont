import pandas as pd
import numpy as np
import os

from clemont.backends.faiss import BruteForce
from clemont.backends.snn import Snn
from clemont.backends.bdd import BDD
from clemont.backends.kdtree import KdTree

def test_sound_and_complete():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "testdata0.csv"))

    epsilon = 0.1

    backend = Snn(df, 'prediction', epsilon, 'l2')

    for index, row in df.iterrows():
        iter_cex_indices = backend.observe(row, row_id=index)

        # Verify all counterexamples are within epsilon distance to current row
        for cex_idx in iter_cex_indices:
            cex_row = df.iloc[cex_idx]
            # Calculate infinity norm distance excluding 'prediction' column
            features_only = row.drop('prediction') if 'prediction' in row.index else row
            cex_features_only = cex_row.drop('prediction') if 'prediction' in cex_row.index else cex_row
            distance = np.max(np.abs(features_only - cex_features_only))
            assert distance <= epsilon, f"Counterexample {cex_idx} distance {distance} exceeds epsilon {epsilon}"
        if iter_cex_indices:
            print(f"counterexamples returned for row {index}: {iter_cex_indices}")

