# Clemont

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15552183.svg)](https://doi.org/10.5281/zenodo.15552183)
[![pypi](https://img.shields.io/pypi/v/clemont?color=white)](https://pypi.org/project/clemont/)

Clemont is a Python package for monitoring AI models for fairness and robustness. It provides multiple monitoring backends (BDD, FAISS, KDTree, and SNN) to detect violations of fairness and robustness constraints in real-time. 

Clemont observes a sequence of input-decision pairs $(x_1, y_1,), \dots, (x_n, y_n)$ in an online fashion (or from a .csv file). The current pair is said to be a *violation* if there exists a past input-decision pair $x_j,y_j$ such that $d(x_j, x_n) < \epsilon$ and $y_j \neq y_n$, where $d$ is some distance metric on the input space, for example $L_\infty$. Clemont accepts input-decision pairs in its `.observe()` method and will return the index of any past pairs that form a violation with respect to the passed pair.

Clemont can maintain a throughput in the hundreds of samples per second even after processing tens of millions of samples, or at an input dimensionality in the tens of thousands, depending on the backend. See our [paper](https://doi.org/10.1145/3711896.3737054) for detailed methodology and backend comparisons.

### News

* August 3, 2025: Our paper has been [awarded](https://kdd2025.kdd.org/awards/) Runner Up for the Best Paper Award in the research track of KDD '25!


## Quickstart

See `example.py` for more detail.

```python
import numpy as np
from clemont.monitor import Monitor
from clemont.frnn import FaissFRNN, NaiveFRNN

# Generate random example data
datapoints = np.random.rand(1000, 10)
decisions = np.random.choice([0, 1], size=1000)

backend_factory = lambda: FaissFRNN(epsilon=0.2, metric="linf")
monitor = Monitor(backend_factory)

for index, (point, decision) in enumerate(zip(datapoints, decisions)):
    result = monitor.observe(point, decision, point_id=index)
    if result.counterexamples.ids: 
        raise RuntimeException("fairness violation!")
```


## Installation

### Docker

For easy experimentation with all backends including BDD, use the provided Docker container:

```bash
# Build the Docker image (may take a few minutes)
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

#### ⚠️ IMPORTANT: Manual installation of BDD dependency (pip only)

The BDD backend requires the `dd.cudd` package, which cannot be automatically installed via pip due to its dependency on CUDD. As a result, if you plan to use the BDD backend, you must install `dd.cudd` manually by running the [official installation script](https://github.com/tulip-control/dd/blob/main/examples/install_dd_cudd.sh) in your Python environment (note that this is done automatically in the Docker image):

```bash
curl -O https://raw.githubusercontent.com/tulip-control/dd/refs/heads/main/examples/install_dd_cudd.sh
chmod +x install_dd_cudd.sh
./install_dd_cudd.sh
```


## Notes

### SNN installation issues

SNN v0.0.6 should be installed automatically, and should work without issues, but note that more recent versions may be difficult to install due to packaging issues. A simple `pip install snnpy` may error out with a message about `pybind11`. If so, run:

```bash
pip install wheel setuptools pip --upgrade
pip install snnpy==0.0.6
```

On Apple Silicon, `brew install llvm libomp` may further be necessary (see [here](https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis)) if the installation now fails for reasons related to OpenMP. If so, execute the above command and set the CC and CXX environment variables accordingly:

```
brew install llvm libomp
export CC=/opt/homebrew/opt/llvm/bin/clang
export CCX=/opt/homebrew/opt/llvm/bin/clang++
```

The Dockerfile uses the most recent version of SNN and may be consulted for the installation procedure.

### Deprecated API

Files `runner.py` and `backends/*` implement the previous Clemont API and are deprecated.


## Citation

If you use Clemont in your research, please cite our paper.

```bibtex
@inproceedings{10.1145/3711896.3737054,
author = {Gupta, Ashutosh and Henzinger, Thomas A. and Kueffner, Konstantin and Mallik, Kaushik and Pape, David},
title = {Monitoring Robustness and Individual Fairness},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737054},
doi = {10.1145/3711896.3737054},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {790–801},
numpages = {12},
keywords = {adversarial robustness, fixed-radius nearest neighbor search, individual fairness, monitoring, semantic robustness, trustworthy ai},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
