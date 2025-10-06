"""Quantitative monitor demonstration using the Adult toy dataset.

Streams paired feature/probability rows through Clemont's quantitative monitor,
prints every Nth observation (including the final one) with formatted point and
witness data, and concludes with percentile summaries plus representative
samples covering low/medium/high neighbour counts.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from clemont.frnn import *
from clemont.quantitative_monitor import QuantitativeMonitor, QuantitativeResult

MAXN = 200000

@dataclass
class Config:
    adult_csv: Path = Path("..") / ".." / "data" / "toydata" / "inputs_numeric.csv"
    predictions_csv: Path = Path("..") / ".." / "data" / "toydata" / "predictions_with_probs.csv"
    numeric_columns: Tuple[str, ...] | None = None
    prob_columns: Tuple[str, ...] | None = None
    frnn_metric: str = "l2"
    out_metric: str = "l2"
    display_stride: int = 1000
    frnn_threads: int = 4


def main() -> None:
    cfg = Config()
    inputs, probs, input_names, prob_names = load_data(cfg)
    num_points = inputs.shape[0]

    backend_factory = lambda: KdTreeFRNN(
        metric=cfg.frnn_metric,
        batchsize=5000
    )

    monitor = QuantitativeMonitor(
        backend_factory,
        out_metric=cfg.out_metric,
        initial_k=16,
    )

    full_records: List[Tuple[QuantitativeResult, np.ndarray, np.ndarray]] = []

    print("=== Streaming quantitative monitoring demo ===")
    print(
        f"Points={num_points}, input-dim={inputs.shape[1]}, probs-dim={probs.shape[1]}, "
        f"FRNN metric={cfg.frnn_metric}, output metric={cfg.out_metric}"
    )
    print(f"Every {cfg.display_stride}th observation (• denotes early stop via bound):")

    display_indices = set(range(0, num_points, cfg.display_stride))
    display_indices.add(num_points - 1)

    for idx, (x_vec, p_vec) in enumerate(zip(inputs, probs)):
        res = monitor.observe(x_vec, p_vec)
        if idx > 16:
            full_records.append((res, x_vec, p_vec))

        if idx in display_indices:
            _print_observation(
                idx,
                res,
                x_vec,
                p_vec,
                inputs,
                probs,
                prob_names,
                input_names,
            )

    ratios = np.array([rec[0].max_ratio for rec in full_records], dtype=float)
    compared = np.array([rec[0].compared_count for rec in full_records], dtype=int)
    early_stop = sum(rec[0].stopped_by_bound for rec in full_records)
    max_depth = max((rec[0].k_progression[-1] for rec in full_records if rec[0].k_progression), default=0)

    def create_scatter_plot(x_values, y_values, filename, title, ylabel, yscale_log=True, alpha=0.1):
        """Create a scatter plot with standardized formatting."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, alpha=alpha, s=4)
        plt.xlabel('Record Index')
        if yscale_log:
            plt.yscale('log')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_histogram(values, filename, title, xlabel, bins=50):
        """Create a histogram with standardized formatting."""
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 6))
        # Create logarithmic bins
        log_bins = np.logspace(np.log10(min(values)), np.log10(max(values)), bins)
        plt.hist(values, bins=log_bins, alpha=0.7, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
        ax = plt.gca()
        ax.set_ylim([0, 10e4])
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    k_final_values = [record[0].k_progression[-1] for record in full_records if record[0].k_progression]
    max_ratios = [record[0].max_ratio for record in full_records if record[0].k_progression]
    x_values = range(len(full_records))

    create_scatter_plot(x_values, k_final_values, 'k_final_values_scatter.png', 'Final k Values Across All Records', 'Final k Value', alpha=0.015)
    create_histogram(max_ratios, 'max_ratios_histogram.png', 'Distribution of Max Ratios', 'Max Ratio')

    print("\n=== Summary ===")
    finite_mask = np.isfinite(ratios)
    if finite_mask.any():
        _print_percentiles("Ratio", ratios[finite_mask])
    else:
        print("Ratio     : all observations produced infinite ratios")

    _print_percentiles("Compared", compared.astype(float))
    print(f"Infinities : {(~finite_mask).sum()} occurrences")
    print(f"Early stops: {early_stop} / {num_points} observations")
    print(f"Largest k  : {max_depth}")

    # Representative examples across neighbour counts
    print("\nSample observations by neighbour count:")
    sorted_indices = np.argsort(compared)
    l = len(sorted_indices)
    sample_positions = [0,1,2,3,4, l // 2, l-4,l-3,l-2,l-1]
    labels = [str(s) for s in sample_positions]
    seen: set[int] = set()
    for label, pos in zip(labels, sample_positions):
        idx = int(sorted_indices[pos])
        if idx in seen:
            continue
        seen.add(idx)
        res, x_vec, p_vec = full_records[idx]
        print(f"-- {label} (index {idx}, compared={res.compared_count})")
        _print_observation(
            idx,
            res,
            x_vec,
            p_vec,
            inputs,
            probs,
            prob_names,
            input_names,
        )


def load_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load numeric Adult features and probability predictions from CSV files."""

    def _ensure_exists(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Expected file at {path.resolve()} (adjust Config paths)")
        return path

    adult_path = _ensure_exists(cfg.adult_csv)
    preds_path = _ensure_exists(cfg.predictions_csv)

    with adult_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        feature_columns = list(cfg.numeric_columns) if cfg.numeric_columns else [
            col for col in (reader.fieldnames or []) if col != "row_id"
        ]
        missing = [col for col in feature_columns if col not in reader.fieldnames]
        if missing:
            raise KeyError(f"Columns {missing} missing in {adult_path}")
        inputs = [[float(row[col]) for col in feature_columns] for row in reader]

    with preds_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if cfg.prob_columns:
            prob_cols = list(cfg.prob_columns)
        else:
            prob_cols = [col for col in (reader.fieldnames or []) if col.lower().startswith("prob")]
        missing_probs = [col for col in prob_cols if col not in reader.fieldnames]
        if missing_probs:
            raise KeyError(
                f"Columns {missing_probs} missing in {preds_path}. Available: {reader.fieldnames}"
            )
        probs = [[float(row[col]) for col in prob_cols] for row in reader]

    n = min(len(inputs), len(probs), MAXN)
    if n == 0:
        raise ValueError("No overlapping rows between feature and probability files")
    if len(inputs) != len(probs):
        print(
            f"Warning: trimming to {n} rows (features={len(inputs)}, probabilities={len(probs)})"
        )

    prob_names: List[str] = []
    for name in prob_cols:
        cleaned = name
        if name.startswith("prob_"):
            cleaned = name.replace("prob_", "p(") + ")"
        prob_names.append(cleaned.lower())

    return (
        np.asarray(inputs[:n], dtype=np.float32),
        np.asarray(probs[:n], dtype=np.float64),
        feature_columns,
        prob_names,
    )


def _print_observation(
    idx: int,
    res: QuantitativeResult,
    x_vec: np.ndarray,
    p_vec: np.ndarray,
    inputs: np.ndarray,
    probs: np.ndarray,
    prob_names: Sequence[str],
    feature_names: Sequence[str],
) -> None:
    witness = res.witness_id if res.witness_id is not None else None
    ratio_disp = "inf" if math.isinf(res.max_ratio) else f"{res.max_ratio:8.4f}"
    flag = "•" if res.stopped_by_bound else " "
    print(f"  [{idx:05d}] ratio={ratio_disp} compared={res.compared_count:5d} witness={witness if witness is not None else '--':>6} {flag}")
    
    if res.witness_in_distance and res.witness_out_distance:
        print(f"          ratio={round(res.witness_out_distance/res.witness_in_distance,4)} d_out={round(res.witness_out_distance, 4)} d_in={round(res.witness_in_distance, 4)}")

    columns = list(prob_names) + list(feature_names)
    header = "                " + " ".join(name.rjust(10)[:10] for name in columns)
    point_row = _format_row(np.concatenate([p_vec, x_vec]))
    print(header)
    print(f"      point   {point_row}")

    if witness is not None and 0 <= witness < inputs.shape[0]:
        witness_row = _format_row(np.concatenate([probs[witness], inputs[witness]]))
        print(f"      witness {witness_row}")
    else:
        print("      witness --")


def _format_row(values: Iterable[float]) -> str:
    return " ".join(f"{float(val):10.4f}".rjust(10)[:10] for val in values)


def _print_percentiles(label: str, values: np.ndarray) -> None:
    perc_points = [0, 50, 90, 95, 99, 100]
    percs = np.percentile(values, perc_points)
    stats = (
        f"min={percs[0]:.4f} median={percs[1]:.4f} p90={percs[2]:.4f} "
        f"p95={percs[3]:.4f} p99={percs[4]:.4f} max={percs[5]:.4f}"
    )
    print(f"{label:<10}: {stats} mean={values.mean():.4f}")


if __name__ == "__main__":
    main()
