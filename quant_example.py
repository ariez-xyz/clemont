from __future__ import annotations

import numpy as np

from clemont.frnn import KdTreeFRNN
from clemont.quantitative_monitor import QuantitativeMonitor


def main() -> None:
    rng = np.random.default_rng(seed=42)

    num_points = 1000
    feature_dim = 8
    num_classes = 4

    backend_factory = lambda: KdTreeFRNN(epsilon=0.2, metric="l2")

    monitor = QuantitativeMonitor(
        backend_factory,
        out_metric="linf",
        initial_k=4,
        max_k=256,
    )

    results = []
    for index in range(num_points):
        point = rng.normal(size=feature_dim)
        probabilities = rng.dirichlet(np.ones(num_classes))

        result = monitor.observe(point, probabilities, point_id=index)
        results.append(result)

        if index < 5:
            _print_observation(result)

    interesting = [res for res in results if np.isfinite(res.max_ratio)]
    interesting.sort(key=lambda res: res.max_ratio, reverse=True)

    print("\nTop 3 ratios from the run:")
    for entry in interesting[:3]:
        _print_observation(entry)


def _print_observation(result) -> None:
    print(
        f"point {result.point_id:3d} | "
        f"max_ratio={result.max_ratio:7.3f} | "
        f"compared={result.compared_count:4d} | "
        f"witness={result.witness_id}"
    )


if __name__ == "__main__":
    main()
