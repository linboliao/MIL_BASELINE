"""Summarize SPE performance overall, by center, and by specimen type.

Metrics are calculated at WSI level. Confidence intervals use patient-level
bootstrap resampling so patients with multiple slides remain clustered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "result/Diagnosis/SPE/spe_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/Diagnosis/SPE/performance"
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
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--center-metadata",
        type=Path,
        default=None,
        help=(
            "CSV containing center plus slide_id or test_slide_path. Required "
            "when the prediction CSV has no center column."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--skip-center",
        action="store_true",
        help="Write overall/type results even when center metadata is unavailable.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def slide_id_from_value(value: object) -> str:
    normalized = str(value).strip().replace("\\", "/")
    return PurePosixPath(normalized).stem


def load_predictions(path: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"slide_id": "string", "patient_id": "string", "center": "string"},
    )
    required = {"slide_id", "patient_id", "type", "label", "prob_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["slide_id"].isna().any() or frame["slide_id"].duplicated().any():
        raise ValueError("Prediction slide_id values must be present and unique.")
    frame = frame.copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["prob_1"] = pd.to_numeric(frame["prob_1"], errors="raise")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only binary 0/1 values.")
    if not np.isfinite(frame["prob_1"]).all() or not frame["prob_1"].between(0, 1).all():
        raise ValueError("prob_1 must contain finite probabilities in [0, 1].")
    if "prob_0" in frame.columns and not np.allclose(
        frame["prob_0"].to_numpy(dtype=float) + frame["prob_1"].to_numpy(dtype=float),
        1.0,
        atol=1e-6,
    ):
        raise ValueError("prob_0 and prob_1 do not sum to one.")
    frame["analysis_prediction"] = (frame["prob_1"] >= threshold).astype(int)
    return frame


def attach_center(frame: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    if "center" in frame.columns and frame["center"].notna().all():
        result = frame.copy()
        result["center"] = result["center"].astype(str)
        return result
    if metadata_path is None:
        raise ValueError(
            "spe_predictions.csv has no complete center column. Supply "
            "--center-metadata with a CSV containing center and slide_id or "
            "test_slide_path, or use --skip-center."
        )
    metadata = pd.read_csv(metadata_path, dtype={"slide_id": "string", "center": "string"})
    if "center" not in metadata.columns:
        raise ValueError(f"{metadata_path} has no center column.")
    id_column = next(
        (column for column in ("slide_id", "test_slide_path") if column in metadata.columns),
        None,
    )
    if id_column is None:
        raise ValueError(f"{metadata_path} needs slide_id or test_slide_path.")
    mapping = metadata[[id_column, "center"]].dropna().copy()
    mapping["slide_id_key"] = mapping[id_column].map(slide_id_from_value)
    conflicts = mapping.groupby("slide_id_key")["center"].nunique().gt(1)
    if conflicts.any():
        examples = conflicts[conflicts].index[:5].tolist()
        raise ValueError(f"Center metadata contains conflicting slide mappings: {examples}")
    mapping = mapping.drop_duplicates("slide_id_key")[["slide_id_key", "center"]]
    result = frame.drop(columns=["center"], errors="ignore").copy()
    result["slide_id_key"] = result["slide_id"].map(slide_id_from_value)
    result = result.merge(mapping, on="slide_id_key", how="left", validate="one_to_one")
    missing_center = result["center"].isna()
    if missing_center.any():
        examples = result.loc[missing_center, "slide_id"].head().tolist()
        raise ValueError(
            f"Center metadata did not match {int(missing_center.sum())} prediction "
            f"slides; examples: {examples}"
        )
    return result.drop(columns="slide_id_key")


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = safe_divide(int(tp), int(tp + fn))
    specificity = safe_divide(int(tn), int(tn + fp))
    both_classes = np.unique(labels).size == 2
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2.0) if both_classes else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if both_classes else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(labels, probabilities))
            if (labels == 1).any()
            else float("nan")
        ),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                labels, predictions, labels=[0, 1], average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                labels, predictions, labels=[0, 1], average="macro", zero_division=0
            )
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": safe_divide(int(tp), int(tp + fp)),
        "npv": safe_divide(int(tn), int(tn + fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def patient_index_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    return [
        np.flatnonzero(patient_ids == patient_id)
        for patient_id in pd.unique(patient_ids)
    ]


def bootstrap_metrics(
    frame: pd.DataFrame,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    groups = patient_index_groups(frame)
    if not groups:
        raise ValueError("Cannot bootstrap an empty group.")
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["prob_1"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = {name: np.empty(iterations, dtype=float) for name in METRIC_NAMES}
    for iteration in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        metrics = calculate_metrics(labels[indices], probabilities[indices], threshold)
        for name in METRIC_NAMES:
            values[name][iteration] = float(metrics[name])
    return values


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def summarize_group(
    frame: pd.DataFrame,
    level: str,
    group: str,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    metrics = calculate_metrics(
        frame["label"].to_numpy(dtype=int),
        frame["prob_1"].to_numpy(dtype=float),
        threshold,
    )
    bootstrap = bootstrap_metrics(frame, threshold, iterations, seed)
    row: dict[str, Any] = {
        "analysis_level": level,
        "group": str(group),
        "n_slides": len(frame),
        "n_patients": frame["patient_id"].nunique(),
        "n_negative": int(frame["label"].eq(0).sum()),
        "n_positive": int(frame["label"].eq(1).sum()),
        "threshold": threshold,
        **metrics,
    }
    for name in METRIC_NAMES:
        low, high = confidence_interval(bootstrap[name])
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    return row


def grouped_summary(
    frame: pd.DataFrame,
    level: str,
    group_column: str | None,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    if group_column is None:
        groups = [("ALL", frame)]
    else:
        if frame[group_column].isna().any():
            raise ValueError(f"{group_column} contains missing values.")
        groups = list(frame.groupby(group_column, sort=True, observed=True))
    rows = [
        summarize_group(
            group_frame,
            level,
            str(group_name),
            threshold,
            iterations,
            seed + index,
        )
        for index, (group_name, group_frame) in enumerate(groups)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1.")
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be at least 1.")
    prediction_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    frame = load_predictions(prediction_path, args.threshold)

    overall = grouped_summary(
        frame, "overall", None, args.threshold, args.bootstrap_iterations, args.seed
    )
    specimen = grouped_summary(
        frame,
        "specimen_type",
        "type",
        args.threshold,
        args.bootstrap_iterations,
        args.seed + 10_000,
    )
    outputs = {
        "overall": output_dir / "overall_metrics.csv",
        "specimen_type": output_dir / "specimen_type_metrics.csv",
    }
    atomic_csv(overall, outputs["overall"])
    atomic_csv(specimen, outputs["specimen_type"])

    summaries = [overall, specimen]
    center_status = "skipped"
    if not args.skip_center:
        metadata_path = (
            resolve_path(args.center_metadata) if args.center_metadata is not None else None
        )
        centered = attach_center(frame, metadata_path)
        center = grouped_summary(
            centered,
            "center",
            "center",
            args.threshold,
            args.bootstrap_iterations,
            args.seed + 20_000,
        )
        outputs["center"] = output_dir / "center_metrics.csv"
        atomic_csv(center, outputs["center"])
        summaries.append(center)
        center_status = "complete"

    combined = pd.concat(summaries, ignore_index=True)
    outputs["combined"] = output_dir / "performance_summary.csv"
    atomic_csv(combined, outputs["combined"])
    atomic_json(
        {
            "prediction_csv": str(prediction_path),
            "center_metadata_csv": (
                str(resolve_path(args.center_metadata))
                if args.center_metadata is not None
                else None
            ),
            "center_analysis": center_status,
            "metric_unit": "WSI",
            "bootstrap_unit": "patient",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.seed,
            "classification_threshold": args.threshold,
            "n_slides": len(frame),
            "n_patients": frame["patient_id"].nunique(),
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
        output_dir / "manifest.json",
    )
    print(f"SPE performance summary complete: {outputs['combined']}")


if __name__ == "__main__":
    main()
