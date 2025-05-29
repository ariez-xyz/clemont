import pandas as pd
import numpy as np
from aimon.backends.faiss import BruteForce

num_rows = 1000
num_columns = 10
epsilon = 0.2

# Create random data
column_names = ['pred'] + [f'c{i}' for i in range(1, num_columns)]
np.random.seed(42)
data = np.random.uniform(0, 1, size=(num_rows, num_columns))
df = pd.DataFrame(data, columns=column_names)
df['pred'] = (df['pred'] > 0.75).astype(int) # Binary decision

# Set up backend
backend = BruteForce(df, 'pred', epsilon)

# Or, to use e.g. BDD:
# from aimon.backends.bdd import BDD
# backend = BDD(df, 1//epsilon, 'pred', collect_cex=True)

# Monitoring procedure
for index, row in df.iterrows():
    iter_cexs = backend.observe(row, row_id=index)
    if iter_cexs:
        print(f"counterexamples returned for row {index}: {iter_cexs}")

