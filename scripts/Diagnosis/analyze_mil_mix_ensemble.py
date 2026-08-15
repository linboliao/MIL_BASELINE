"""Analyze fold and architecture ensembles from numeric prediction CSV files.

The script is deliberately separate from inference: it consumes one
``fold_ensemble_predictions.csv`` per architecture, verifies that every file
describes the same cohort, and reports exhaustive equal-weight subsets together
with model diversity and oracle-headroom diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "roc_auc": roc_auc_score(labels, probabilities),
        "sensitivity": predictions[labels == 1].mean(),
        "specificity": (predictions[labels == 0] == 0).mean(),
    }


def discover(root: Path, run_regex: str | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in root.glob("*/seed_*/fold_ensemble_predictions.csv"):
        if run_regex is not None and re.search(run_regex, path.parent.name) is None:
            continue
        model = path.parents[1].name
        if model in paths:
            raise ValueError(f"Multiple prediction files found for {model}")
        paths[model] = path
    if not paths:
        raise FileNotFoundError(f"No fold ensemble predictions under {root}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-subset-size", type=int, default=6)
    parser.add_argument("--run-regex", default=None)
    args = parser.parse_args()

    paths = discover(args.prediction_root, args.run_regex)
    names = sorted(paths)
    reference: pd.DataFrame | None = None
    probabilities: list[np.ndarray] = []
    individual_rows: list[dict[str, object]] = []
    for name in names:
        frame = pd.read_csv(paths[name], dtype={"slide_id": "string"})
        required = {"slide_id", "label", "prob_1"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{paths[name]} missing {sorted(missing)}")
        metadata = frame[["slide_id", "label"]].copy()
        metadata["label"] = metadata["label"].astype(int)
        if reference is None:
            reference = metadata
        else:
            pd.testing.assert_frame_equal(reference, metadata, check_dtype=False)
        prob = frame["prob_1"].to_numpy(float)
        probabilities.append(prob)
        individual_rows.append({"model": name, **metrics(metadata.label.to_numpy(), prob)})

    assert reference is not None
    labels = reference.label.to_numpy()
    matrix = np.column_stack(probabilities)
    predictions = matrix >= 0.5
    correct = predictions == labels[:, None]

    subset_rows: list[dict[str, object]] = []
    maximum = min(args.max_subset_size, len(names))
    for size in range(2, maximum + 1):
        for indices in itertools.combinations(range(len(names)), size):
            prob = matrix[:, indices].mean(axis=1)
            subset_rows.append(
                {
                    "size": size,
                    "models": "+".join(names[index] for index in indices),
                    **metrics(labels, prob),
                }
            )

    all_prob = matrix.mean(axis=1)
    if maximum < len(names):
        subset_rows.append({"size": len(names), "models": "ALL", **metrics(labels, all_prob)})
    subsets = pd.DataFrame(subset_rows).sort_values(
        ["balanced_accuracy", "macro_f1", "roc_auc"], ascending=False
    )

    disagreement = np.zeros((len(names), len(names)), dtype=float)
    error_overlap = np.zeros_like(disagreement)
    for left in range(len(names)):
        for right in range(len(names)):
            disagreement[left, right] = np.mean(predictions[:, left] != predictions[:, right])
            union = np.logical_or(~correct[:, left], ~correct[:, right]).sum()
            error_overlap[left, right] = (
                np.logical_and(~correct[:, left], ~correct[:, right]).sum() / union
                if union
                else 0.0
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(individual_rows).sort_values(
        ["balanced_accuracy", "macro_f1"], ascending=False
    ).to_csv(args.output_dir / "individual_metrics.csv", index=False)
    subsets.to_csv(args.output_dir / "equal_weight_subset_metrics.csv", index=False)
    pd.DataFrame(disagreement, index=names, columns=names).to_csv(
        args.output_dir / "prediction_disagreement.csv"
    )
    pd.DataFrame(error_overlap, index=names, columns=names).to_csv(
        args.output_dir / "error_jaccard.csv"
    )

    oracle = correct.any(axis=1).mean()
    all_metrics = metrics(labels, all_prob)
    best = subsets.iloc[0].to_dict()
    summary = pd.DataFrame(
        [
            {"diagnostic": "n_samples", "value": len(labels)},
            {"diagnostic": "n_models", "value": len(names)},
            {"diagnostic": "all_model_balanced_accuracy", "value": all_metrics["balanced_accuracy"]},
            {"diagnostic": "best_subset", "value": best["models"]},
            {"diagnostic": "best_subset_balanced_accuracy", "value": best["balanced_accuracy"]},
            {"diagnostic": "oracle_any_model_accuracy", "value": oracle},
            {"diagnostic": "all_models_wrong_samples", "value": int((~correct.any(axis=1)).sum())},
        ]
    )
    summary.to_csv(args.output_dir / "headroom_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
