"""Summarize WSI representation OOF predictions into analysis-ready CSV files."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
)
PAIRWISE_METRICS = ("balanced_accuracy", "roc_auc", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create pooled, fold-level, specimen-level, and paired-bootstrap "
            "CSV summaries from OOF prediction files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("result/Diagnosis/Mag/OOF"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/Diagnosis/Mag/Statistics"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--exclude-unstained-baseline", action="store_true", help=(
        "Do not reuse Mag/10x as Stains/Unnormalized in the "
        "stain comparison."
    ))
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Skip paired patient-bootstrap differences between variants.",
                        )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_metrics(
        labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": auc,
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def patient_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    patient_values = frame["patient_id"].astype(str).to_numpy()
    return [
        np.flatnonzero(patient_values == patient)
        for patient in pd.unique(patient_values)
    ]


def bootstrap_metric_values(
        frame: pd.DataFrame,
        iterations: int,
        seed: int,
        threshold: float,
) -> dict[str, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["prob_1"].to_numpy(dtype=float)
    groups = patient_groups(frame)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRIC_NAMES}

    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        metrics = calculate_metrics(
            labels[indices], probabilities[indices], threshold
        )
        for metric in METRIC_NAMES:
            values[metric].append(metrics[metric])
    return {
        metric: np.asarray(metric_values, dtype=float)
        for metric, metric_values in values.items()
    }


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def validate_prediction_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {
        "comparison",
        "variant",
        "training_run",
        "fold",
        "slide_id",
        "patient_id",
        "type",
        "label",
        "prob_0",
        "prob_1",
        "prediction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if frame["slide_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate slide_id values.")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{source} contains labels outside 0/1.")
    if not ((frame["prob_0"] + frame["prob_1"] - 1).abs() < 1e-5).all():
        raise ValueError(f"{source} contains probabilities that do not sum to one.")
    frame = frame.copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def discover_latest_predictions(
        input_root: Path,
) -> tuple[dict[tuple[str, str], pd.DataFrame], list[dict]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"OOF input root not found: {input_root}")

    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    sources: list[dict] = []
    for comparison_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for variant_dir in sorted(path for path in comparison_dir.iterdir() if path.is_dir()):
            candidates = sorted(
                variant_dir.glob("*/oof_predictions.csv"),
                key=lambda path: (path.stat().st_mtime, path.as_posix()),
                reverse=True,
            )
            if not candidates:
                continue
            source = candidates[0]
            frame = validate_prediction_frame(
                pd.read_csv(
                    source,
                    dtype={"slide_id": "string", "patient_id": "string"},
                ),
                source,
            )
            key = (comparison_dir.name, variant_dir.name)
            predictions[key] = frame
            sources.append(
                {
                    "comparison": key[0],
                    "variant": key[1],
                    "training_run": str(frame["training_run"].iloc[0]),
                    "prediction_csv": source.as_posix(),
                    "is_reused_baseline": False,
                }
            )
    if not predictions:
        raise FileNotFoundError(f"No oof_predictions.csv files found under {input_root}")
    return predictions, sources


def add_unstained_baseline(
        predictions: dict[tuple[str, str], pd.DataFrame], sources: list[dict]
) -> None:
    source_key = ("Mag", "10x")
    target_key = ("Stains", "Unnormalized")
    if source_key not in predictions or target_key in predictions:
        return
    baseline = predictions[source_key].copy()
    baseline["comparison"] = "Stains"
    baseline["variant"] = "Unnormalized"
    predictions[target_key] = baseline
    source_record = next(
        record
        for record in sources
        if (record["comparison"], record["variant"]) == source_key
    )
    sources.append(
        {
            **source_record,
            "comparison": "Stains",
            "variant": "Unnormalized",
            "is_reused_baseline": True,
        }
    )


def pooled_summary(
        predictions: dict[tuple[str, str], pd.DataFrame],
        iterations: int,
        seed: int,
        threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment_index, ((comparison, variant), frame) in enumerate(
            sorted(predictions.items())
    ):
        metrics = calculate_metrics(
            frame["label"].to_numpy(),
            frame["prob_1"].to_numpy(),
            threshold,
        )
        bootstrap = bootstrap_metric_values(
            frame,
            iterations=iterations,
            seed=seed + experiment_index,
            threshold=threshold,
        )
        row = {
            "comparison": comparison,
            "variant": variant,
            "training_run": str(frame["training_run"].iloc[0]),
            "n_slides": len(frame),
            "n_patients": frame["patient_id"].nunique(),
            "n_negative": int((frame["label"] == 0).sum()),
            "n_positive": int((frame["label"] == 1).sum()),
            "threshold": threshold,
            **metrics,
        }
        for metric in METRIC_NAMES:
            low, high = confidence_interval(bootstrap[metric])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)

    summary = pd.DataFrame(rows)
    for metric in ("balanced_accuracy", "roc_auc", "macro_f1"):
        summary[f"rank_{metric}"] = (
            summary.groupby("comparison")[metric]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
    return summary.sort_values(
        ["comparison", "rank_macro_f1", "variant"]
    ).reset_index(drop=True)


def grouped_metrics(
        predictions: dict[tuple[str, str], pd.DataFrame],
        threshold: float,
        group_column: str,
        output_group_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (comparison, variant), frame in sorted(predictions.items()):
        for group_value, group in sorted(frame.groupby(group_column)):
            rows.append(
                {
                    "comparison": comparison,
                    "variant": variant,
                    output_group_name: group_value,
                    "n_slides": len(group),
                    "n_patients": group["patient_id"].nunique(),
                    "n_negative": int((group["label"] == 0).sum()),
                    "n_positive": int((group["label"] == 1).sum()),
                    "threshold": threshold,
                    **calculate_metrics(
                        group["label"].to_numpy(),
                        group["prob_1"].to_numpy(),
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_rows(
        comparison: str,
        variant_a: str,
        frame_a: pd.DataFrame,
        variant_b: str,
        frame_b: pd.DataFrame,
        iterations: int,
        seed: int,
        threshold: float,
) -> list[dict]:
    left = frame_a.sort_values("slide_id").reset_index(drop=True)
    right = frame_b.sort_values("slide_id").reset_index(drop=True)
    if left["slide_id"].tolist() != right["slide_id"].tolist():
        raise ValueError(
            f"Cannot pair {comparison}/{variant_a} and {variant_b}: slide sets differ."
        )
    if not np.array_equal(left["label"].to_numpy(), right["label"].to_numpy()):
        raise ValueError(
            f"Cannot pair {comparison}/{variant_a} and {variant_b}: labels differ."
        )
    if left["patient_id"].astype(str).tolist() != right["patient_id"].astype(str).tolist():
        raise ValueError(
            f"Cannot pair {comparison}/{variant_a} and {variant_b}: patients differ."
        )

    labels = left["label"].to_numpy(dtype=int)
    probs_a = left["prob_1"].to_numpy(dtype=float)
    probs_b = right["prob_1"].to_numpy(dtype=float)
    groups = patient_groups(left)
    rng = np.random.default_rng(seed)
    differences = {metric: [] for metric in PAIRWISE_METRICS}

    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        metrics_a = calculate_metrics(labels[indices], probs_a[indices], threshold)
        metrics_b = calculate_metrics(labels[indices], probs_b[indices], threshold)
        for metric in PAIRWISE_METRICS:
            differences[metric].append(metrics_a[metric] - metrics_b[metric])

    point_a = calculate_metrics(labels, probs_a, threshold)
    point_b = calculate_metrics(labels, probs_b, threshold)
    rows: list[dict] = []
    for metric in PAIRWISE_METRICS:
        values = np.asarray(differences[metric], dtype=float)
        finite = values[np.isfinite(values)]
        low, high = confidence_interval(finite)
        p_value = (
            min(
                1.0,
                2
                * min(
                    float(np.mean(finite <= 0)),
                    float(np.mean(finite >= 0)),
                ),
            )
            if len(finite)
            else float("nan")
        )
        rows.append(
            {
                "comparison": comparison,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "metric": metric,
                "value_a": point_a[metric],
                "value_b": point_b[metric],
                "difference_a_minus_b": point_a[metric] - point_b[metric],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_p_two_sided": p_value,
                "bootstrap_iterations": iterations,
                "bootstrap_unit": "patient",
            }
        )
    return rows


def paired_comparisons(
        predictions: dict[tuple[str, str], pd.DataFrame],
        iterations: int,
        seed: int,
        threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    comparison_names = sorted({key[0] for key in predictions})
    pair_index = 0
    for comparison in comparison_names:
        variants = sorted(
            variant
            for candidate_comparison, variant in predictions
            if candidate_comparison == comparison
        )
        for variant_a, variant_b in itertools.combinations(variants, 2):
            rows.extend(
                paired_bootstrap_rows(
                    comparison=comparison,
                    variant_a=variant_a,
                    frame_a=predictions[(comparison, variant_a)],
                    variant_b=variant_b,
                    frame_b=predictions[(comparison, variant_b)],
                    iterations=iterations,
                    seed=seed + pair_index,
                    threshold=threshold,
                )
            )
            pair_index += 1
    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1.")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    predictions, sources = discover_latest_predictions(args.input_root)
    if not args.exclude_unstained_baseline:
        add_unstained_baseline(predictions, sources)

    summary = pooled_summary(
        predictions,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        threshold=args.threshold,
    )
    fold_metrics = grouped_metrics(
        predictions,
        threshold=args.threshold,
        group_column="fold",
        output_group_name="fold",
    )
    specimen_metrics = grouped_metrics(
        predictions,
        threshold=args.threshold,
        group_column="type",
        output_group_name="specimen_type",
    )

    outputs = {
        "oof_performance_summary.csv": summary,
        "oof_fold_metrics.csv": fold_metrics,
        "oof_specimen_metrics.csv": specimen_metrics,
        "oof_sources.csv": pd.DataFrame(sources).sort_values(
            ["comparison", "variant"]
        ),
    }
    if not args.skip_pairwise:
        outputs["oof_pairwise_patient_bootstrap.csv"] = paired_comparisons(
            predictions,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 100_000,
            threshold=args.threshold,
        )

    for filename, frame in outputs.items():
        output_path = args.output_dir / filename
        write_csv_atomic(frame, output_path)
        print(f"Saved {len(frame)} rows: {output_path}")

    manifest = {
        "input_root": args.input_root.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "threshold": args.threshold,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "bootstrap_unit": "patient",
        "unstained_10x_reused_as_stain_baseline": (
                not args.exclude_unstained_baseline
                and ("Stains", "Unnormalized") in predictions
        ),
        "experiments": [
            {"comparison": comparison, "variant": variant}
            for comparison, variant in sorted(predictions)
        ],
        "output_csvs": sorted(outputs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
