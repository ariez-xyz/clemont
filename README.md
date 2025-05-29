# Clemont

Clemont is a Python package for monitoring AI models for fairness and robustness. It provides multiple monitoring backends (BDD, FAISS, KDTree, and SNN) to detect violations of fairness and robustness constraints in real-time. 

Clemont can maintain a throughput in the hundreds of samples per second even after processing tens of millions of samples, or at an input dimensionality in the tens of thousands, depending on the backend. See our paper for detailed methodology and backend comparisons.

## Installation

### Docker

For easy experimentation with all backends including BDD and SNN, use the provided Docker container:

```bash
# Build the Docker image
docker build -t clemont .
# Run the example script
docker run --rm -v $(pwd):/workspace clemont python example.py
# Run an interactive terminal for development
docker run -it --rm -v $(pwd):/workspace clemont
```


### pip

```bash
pip install clemont
```

The BDD and SNN dependencies require extra steps to install. As a result, the corresponding backends will not be available until the following steps have been completed (both can be done using `postinstall.sh`):

#### BDD requirements

The BDD backend requires the `dd.cudd` package, which cannot be automatically installed via pip due to its dependency on CUDD. If you plan to use the BDD backend, you must install `dd` with CUDD bindings manually using the official installation script.

```bash
curl -O https://raw.githubusercontent.com/tulip-control/dd/refs/heads/main/examples/install_dd_cudd.sh
chmod +x install_dd_cudd.sh
./install_dd_cudd.sh
```

#### SNN requirements

This installation may be problematic. A simple `pip install snnpy` may error out with a message about `pybind11`. If so, run:

```bash
pip install wheel setuptools pip --upgrade
pip install snnpy
```

On Apple Silicon, `brew install llvm libomp` may be necessary (see [here](https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis)) if the installation now fails for reasons related to OpenMP. If so, try the above command and set the CC and CXX environment variables accordingly, e.g.

```
export CC=/opt/homebrew/opt/llvm/bin/clang
export CCX=/opt/homebrew/opt/llvm/bin/clang++
```

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
from clemont.backends.faiss import BruteForce

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
