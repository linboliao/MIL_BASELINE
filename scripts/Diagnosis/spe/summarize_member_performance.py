"""Summarize every SPE member model on the independent test cohort.

The output contains one row per architecture. Confidence intervals use
patient-level bootstrap resampling so patients with multiple slides remain
clustered. Only NumPy and the Python standard library are required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "result/Diagnosis/SPE/Internal/bacc/member_predictions"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "result/Diagnosis/SPE/Internal/bacc/performance/member_metrics.csv"
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def zero_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(np.sum(labels == 1))
    if not positive_count:
        return float("nan")
    order = np.argsort(-probabilities, kind="stable")
    ranked_labels = labels[order]
    precision = np.cumsum(ranked_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision[ranked_labels == 1]) / positive_count)


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    positive_precision = zero_divide(tp, tp + fp)
    negative_precision = zero_divide(tn, tn + fn)
    positive_recall = zero_divide(tp, tp + fn)
    negative_recall = zero_divide(tn, tn + fp)
    positive_f1 = zero_divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = zero_divide(2 * tn, 2 * tn + fp + fn)
    both_classes = bool(np.any(labels == 0) and np.any(labels == 1))
    return {
        "accuracy": zero_divide(tp + tn, len(labels)),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2.0 if both_classes else float("nan")
        ),
        "roc_auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "macro_f1": (positive_f1 + negative_f1) / 2.0,
        "macro_precision": (positive_precision + negative_precision) / 2.0,
        "macro_recall": (positive_recall + negative_recall) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def discover_prediction_files(input_root: Path) -> list[tuple[str, str, Path]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    discovered = []
    for model_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        candidates = sorted(model_dir.glob("*/test_predictions.csv"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one test_predictions.csv for {model_dir.name}, "
                f"found {len(candidates)}."
            )
        source = candidates[0]
        discovered.append((model_dir.name, source.parent.name, source))
    if not discovered:
        raise FileNotFoundError(f"No member prediction files found under {input_root}")
    return discovered


def load_predictions(path: Path, expected_model: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "patient_id", "label", "prob_1", "prob_0", "architecture"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"{path} is empty.")

    slide_ids = [row["slide_id"] for row in rows]
    patient_ids = np.asarray([row["patient_id"] for row in rows], dtype=object)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["prob_1"]) for row in rows], dtype=float)
    prob_0 = np.asarray([float(row["prob_0"]) for row in rows], dtype=float)
    architectures = {row["architecture"] for row in rows}
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError(f"{path} contains duplicate slide_id values.")
    if not set(labels).issubset({0, 1}):
        raise ValueError(f"{path} contains labels outside 0/1.")
    if not np.isfinite(probabilities).all() or not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError(f"{path} contains invalid prob_1 values.")
    if not np.allclose(probabilities + prob_0, 1.0, atol=1e-6):
        raise ValueError(f"{path} contains probabilities that do not sum to one.")
    if architectures != {expected_model}:
        raise ValueError(f"{path} architecture values do not match {expected_model}: {architectures}")
    return {
        "slide_ids": slide_ids,
        "patient_ids": patient_ids,
        "labels": labels,
        "probabilities": probabilities,
    }


def bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    groups = [np.flatnonzero(patient_ids == patient) for patient in np.unique(patient_ids)]
    rng = np.random.default_rng(seed)
    values = {name: np.empty(iterations, dtype=float) for name in METRIC_NAMES}
    for iteration in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        metrics = calculate_metrics(labels[indices], probabilities[indices], threshold)
        for name in METRIC_NAMES:
            values[name][iteration] = float(metrics[name])
    intervals = {}
    for name, samples in values.items():
        finite = samples[np.isfinite(samples)]
        if len(finite):
            low, high = np.percentile(finite, [2.5, 97.5])
            intervals[name] = (
                float(np.clip(low, 0, 1)),
                float(np.clip(high, 0, 1)),
            )
        else:
            intervals[name] = (float("nan"), float("nan"))
    return intervals


def build_summary(
    input_root: Path, threshold: float, iterations: int, seed: int
) -> list[dict[str, Any]]:
    rows = []
    cohort_signature: tuple[list[str], list[int]] | None = None
    for index, (model, run, source) in enumerate(discover_prediction_files(input_root)):
        data = load_predictions(source, model)
        signature = (data["slide_ids"], data["labels"].tolist())
        if cohort_signature is None:
            cohort_signature = signature
        elif signature != cohort_signature:
            raise ValueError(f"Test cohort or labels differ for {model}.")
        metrics = calculate_metrics(data["labels"], data["probabilities"], threshold)
        intervals = bootstrap_intervals(
            data["labels"],
            data["probabilities"],
            data["patient_ids"],
            threshold,
            iterations,
            seed + index,
        )
        row: dict[str, Any] = {
            "model": model,
            "run": run,
            "n_slides": len(data["labels"]),
            "n_patients": len(np.unique(data["patient_ids"])),
            "n_negative": int(np.sum(data["labels"] == 0)),
            "n_positive": int(np.sum(data["labels"] == 1)),
            "threshold": threshold,
            **metrics,
        }
        for name in METRIC_NAMES:
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = intervals[name]
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1].")
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")
    rows = build_summary(
        args.input_root.resolve(),
        args.threshold,
        args.bootstrap_iterations,
        args.seed,
    )
    output = args.output.resolve()
    write_csv(rows, output)
    print(f"Saved {len(rows)} model rows to {output}")


if __name__ == "__main__":
    main()
