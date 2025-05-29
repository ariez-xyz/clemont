# Clemont

Clemont is a Python package for monitoring AI models for fairness and robustness. It provides multiple monitoring backends (BDD, FAISS, KDTree, and SNN) to detect violations of fairness and robustness constraints in real-time. 

Clemont can maintain a throughput in the hundreds of samples per second even after processing tens of millions of samples, or at an input dimensionality in the tens of thousands, depending on the backend. See our paper for detailed methodology and backend comparisons.

## Installation

Install Clemont from PyPI:

```bash
pip install clemont
```

### BDD Backend Requirements

**Important**: The BDD backend requires the `dd.cudd` package, which cannot be automatically installed via pip due to its dependency on CUDD which is a C software package. If you plan to use the BDD backend, you must install `dd` with CUDD bindings manually using the official installation script:

```bash
curl -O https://raw.githubusercontent.com/tulip-control/dd/refs/heads/main/examples/install_dd_cudd.sh
chmod +x install_dd_cudd.sh
./install_dd_cudd.sh
```

### Docker

For easy experimentation with all backends including BDD, use the provided Docker container which has all dependencies pre-installed:

```bash
# Build the Docker image
docker build -t clemont .

# Run an interactive container with your current directory mounted
docker run -it -v $(pwd):/workspace clemont

# Inside the container, you can now use all backends including BDD
python -c "from aimon.backends.bdd import BDD; print('BDD backend ready!')"
```

The Docker container provides a bash shell where you can run Python scripts or start an interactive Python session with all Clemont backends available.

## Usage

Clemont's monitoring procedure is built around two core methods:

**Backend Constructor**: Initialize a monitoring backend with your training data sample. The constructor signature varies by backend, but typically takes a pandas DataFrame containing your data sample, which is used to infer information about dimensionality, column names, value ranges, and decision classes. Additional parameters are documented in each backend, and include $\epsilon$ or discretization bins (for BDD), distance metrics (for FAISS/KDTree), batch sizes, and the prediction column name.

**Observe Method**: Monitor new samples in real-time using the `observe()` method. This method takes a pandas Series representing a new data point and returns a list of row IDs from your training data that violate fairness or robustness constraints (i.e., samples that are epsilon-close to the new point but have different predictions). 

* For indexed backends, the `observe()` method transparently handles short-term and long-term memory management.
* The BDD backend may return false positives, so post-verification may be desired.
* There is a `DataframeRunner` class included that handles post-verification and takes performance metrics. For its usage, as well as a powerful script for running experiments on Clemont, see our separate [experiments repository](https://github.com/ariez-xyz/aimon). 

### Example

See also `example.py`.

```python
import pandas as pd
import numpy as np
from aimon.backends.faiss import BruteForce

# Create random example data
column_names = ['pred'] + [f'c{i}' for i in range(1, 10)]
df = pd.DataFrame(np.random.uniform(0, 1, size=(1000, 10)), columns=column_names)
df['pred'] = (df['pred'] > 0.75).astype(int)  # Binary decision

# Initialize monitoring backend
backend = BruteForce(df, 'pred', epsilon=0.2)

# Monitor new samples for fairness/robustness violations
for index, row in df.iterrows():
    violations = backend.observe(row, row_id=index)
    print(f"{index}: {violations}")
```

## Citation

If you use Clemont in your research, please cite our paper:

```bibtex
[TODO]
```
