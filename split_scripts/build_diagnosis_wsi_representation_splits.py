"""Build patient-disjoint development folds for WSI representation selection.

This script intentionally uses only ``datasets/Diagnosis/train_val.csv``.
The independent internal and external test cohorts must remain untouched until
the PFM, magnification, and stain-normalization settings have been locked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PFM_VARIANTS = (
    "CONCH",
    "h-optimus-1",
    "mstar",
    "omiclip",
    "UNI",
    "UNI2",
    "virchow2",
)

MAG_VARIANTS = {
    "20x": "feat_0_224",
    "10x": "feat_0_448",
    "5x": "feat_0_896",
}

STAIN_VARIANTS = ("Macenko", "Reinhard", "Vahadane")


def patient_id_from_slide_id(slide_id: str) -> str:
    """Return a stable patient ID without changing the feature filename."""
    canonical_id = re.sub(
        r"(?:有癌|无癌)[ _.]*$",
        "",
        slide_id.strip(),
    )
    return canonical_id.split(".", maxsplit=1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build identical patient-level five-fold train/validation splits "
            "for PFM, magnification, and stain-normalization comparisons."
        )
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("datasets/Diagnosis/train_val.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/Diagnosis"),
    )
    parser.add_argument(
        "--nas-root",
        default="/NAS145/liaolinbo/Data/MXB/CLS",
        help="POSIX root containing the development-cohort feature directories.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--conflict-policy",
        choices=("error", "exclude"),
        default="error",
        help=(
            "How to handle a slide_id associated with more than one label. "
            "Use 'exclude' only after recording the source-data issue."
        ),
    )
    return parser.parse_args()


def load_and_clean_source(
    source_csv: Path, conflict_policy: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(source_csv, dtype={"slide_id": "string", "type": "string"})
    required = {"slide_id", "label", "type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Source CSV contains missing slide_id, label, or type values.")

    df = df.loc[:, ["slide_id", "label", "type"]].copy()
    if df["slide_id"].str.strip().eq("").any():
        raise ValueError("Source CSV contains an empty slide_id.")
    df["type"] = df["type"].str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype("int64")

    invalid_labels = sorted(set(df["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Expected binary labels 0/1, found: {invalid_labels}")

    exact_duplicate_mask = df.duplicated(
        subset=["slide_id", "label", "type"], keep="first"
    )
    exact_duplicate_rows = df.loc[exact_duplicate_mask].copy()
    df = df.loc[~exact_duplicate_mask].copy()

    conflicting_slide_ids = sorted(
        df.groupby("slide_id", sort=False)["label"]
        .nunique()
        .loc[lambda values: values > 1]
        .index.tolist()
    )
    conflict_rows = df.loc[df["slide_id"].isin(conflicting_slide_ids)].copy()
    if conflicting_slide_ids and conflict_policy == "error":
        raise ValueError(
            "Conflicting labels found for slide_id(s): "
            f"{conflicting_slide_ids}. Rerun with --conflict-policy exclude "
            "only if exclusion is acceptable."
        )
    if conflicting_slide_ids:
        df = df.loc[~df["slide_id"].isin(conflicting_slide_ids)].copy()

    # Preserve slide_id exactly because spaces, underscores, punctuation, and
    # 有癌/无癌 can be part of the real .pt filename. Remove only the cancer
    # annotation when deriving the patient grouping key.
    df["patient_id"] = df["slide_id"].map(patient_id_from_slide_id)
    df = df.reset_index(drop=True)

    exclusions = pd.concat(
        [
            exact_duplicate_rows.assign(exclusion_reason="exact_duplicate"),
            conflict_rows.assign(exclusion_reason="conflicting_label"),
        ],
        ignore_index=True,
    )
    audit = {
        "source_rows": int(len(df) + len(exact_duplicate_rows) + len(conflict_rows)),
        "included_rows": int(len(df)),
        "included_patients": int(df["patient_id"].nunique()),
        "exact_duplicate_rows_excluded": int(len(exact_duplicate_rows)),
        "conflicting_rows_excluded": int(len(conflict_rows)),
        "conflicting_slide_ids": conflicting_slide_ids,
        "label_counts": {
            str(key): int(value)
            for key, value in df["label"].value_counts().sort_index().items()
        },
        "type_counts": {
            str(key): int(value)
            for key, value in df["type"].value_counts().sort_index().items()
        },
    }
    return df, exclusions, audit


def make_fold_indices(
    df: pd.DataFrame, folds: int, seed: int
) -> tuple[list[tuple[list[int], list[int]]], pd.Series, list[dict]]:
    group_label_counts = (
        df.groupby("patient_id")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    group_vectors = np.column_stack(
        [
            group_label_counts.to_numpy(dtype="int64"),
            np.ones(len(group_label_counts), dtype="int64"),
        ]
    )
    totals = group_vectors.sum(axis=0)
    if (totals == 0).any():
        raise ValueError("Cannot construct folds with an empty label or patient group.")

    # Assign large patient groups first. Seeded jitter makes ties deterministic
    # without letting the source-row order determine the partitions.
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(group_vectors))
    order = np.lexsort(
        (
            jitter,
            -np.abs(group_vectors[:, 0] - group_vectors[:, 1]),
            -group_vectors[:, :2].max(axis=1),
            -group_vectors[:, :2].sum(axis=1),
        )
    )

    fold_vectors = np.zeros((folds, 3), dtype="float64")
    patient_fold = np.full(len(group_vectors), -1, dtype="int64")
    target_fraction = 1.0 / folds

    for patient_position in order:
        candidate_scores: list[tuple[float, float, float, int]] = []
        for fold_index in range(folds):
            candidate = fold_vectors.copy()
            candidate[fold_index] += group_vectors[patient_position]
            relative_fraction = candidate / totals
            label_error = np.square(
                relative_fraction[:, :2] - target_fraction
            ).sum()
            patient_error = np.square(
                relative_fraction[:, 2] - target_fraction
            ).sum()
            score = float(label_error + 0.2 * patient_error)
            candidate_scores.append(
                (
                    score,
                    float(fold_vectors[fold_index, :2].sum()),
                    float(fold_vectors[fold_index, 2]),
                    fold_index,
                )
            )

        selected_fold = min(candidate_scores)[3]
        patient_fold[patient_position] = selected_fold
        fold_vectors[selected_fold] += group_vectors[patient_position]

    patient_to_fold = dict(
        zip(group_label_counts.index.tolist(), patient_fold.tolist())
    )
    assigned_fold = df["patient_id"].map(patient_to_fold)

    split_indices: list[tuple[list[int], list[int]]] = []
    validation_fold = pd.Series(pd.NA, index=df.index, dtype="Int64")
    fold_summaries: list[dict] = []

    for fold_number in range(1, folds + 1):
        val_index = df.index[assigned_fold == fold_number - 1].tolist()
        train_index = df.index[assigned_fold != fold_number - 1].tolist()
        train_patients = set(df.loc[train_index, "patient_id"])
        val_patients = set(df.loc[val_index, "patient_id"])
        overlap = train_patients.intersection(val_patients)
        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in fold {fold_number}: {sorted(overlap)[:5]}"
            )

        validation_fold.loc[val_index] = fold_number
        split_indices.append((train_index, val_index))
        fold_summaries.append(
            {
                "fold": fold_number,
                "train_slides": len(train_index),
                "val_slides": len(val_index),
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "train_label_counts": {
                    str(key): int(value)
                    for key, value in df.loc[train_index, "label"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "val_label_counts": {
                    str(key): int(value)
                    for key, value in df.loc[val_index, "label"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            }
        )

    if validation_fold.isna().any():
        raise RuntimeError("At least one development slide was not assigned to a fold.")
    return split_indices, validation_fold, fold_summaries


def feature_path(nas_root: str, relative_dir: str, slide_id: str) -> str:
    return str(
        PurePosixPath(nas_root) / PurePosixPath(relative_dir) / f"{slide_id}.pt"
    )


def build_split_frame(
    df: pd.DataFrame,
    train_index: list[int],
    val_index: list[int],
    nas_root: str,
    relative_dir: str,
) -> pd.DataFrame:
    train_df = df.loc[train_index]
    val_df = df.loc[val_index]
    row_count = max(len(train_df), len(val_df))

    train_paths = [
        feature_path(nas_root, relative_dir, slide_id)
        for slide_id in train_df["slide_id"]
    ]
    val_paths = [
        feature_path(nas_root, relative_dir, slide_id)
        for slide_id in val_df["slide_id"]
    ]

    result = pd.DataFrame(index=range(row_count))
    result["train_slide_path"] = pd.Series(train_paths, dtype="string")
    result["train_label"] = pd.Series(
        pd.array(train_df["label"].tolist(), dtype="Int64")
    )
    result["val_slide_path"] = pd.Series(val_paths, dtype="string")
    result["val_label"] = pd.Series(
        pd.array(val_df["label"].tolist(), dtype="Int64")
    )
    result["test_slide_path"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["test_label"] = pd.Series(
        pd.array([pd.NA] * row_count, dtype="Int64")
    )
    return result


def all_variants() -> dict[str, dict[str, str]]:
    return {
        "PFM": {
            model: f"feat_0_224/pt_files/{model}" for model in PFM_VARIANTS
        },
        "Mag": {
            magnification: f"{feature_dir}/pt_files/h-optimus-1"
            for magnification, feature_dir in MAG_VARIANTS.items()
        },
        "Stains": {
            stain: (
                f"feat_0_448/stains/{stain}/pt_files/h-optimus-1"
            )
            for stain in STAIN_VARIANTS
        },
    }


def write_variant_folds(
    output_root: Path,
    df: pd.DataFrame,
    split_indices: list[tuple[list[int], list[int]]],
    nas_root: str,
    folds: int,
) -> list[str]:
    written_files: list[str] = []
    for comparison, variants in all_variants().items():
        for variant, relative_dir in variants.items():
            variant_dir = output_root / comparison / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for fold_number, (train_index, val_index) in enumerate(
                split_indices, start=1
            ):
                split_df = build_split_frame(
                    df,
                    train_index,
                    val_index,
                    nas_root,
                    relative_dir,
                )
                output_path = (
                    variant_dir
                    / f"Total_{folds}-fold_{variant}_{fold_number}fold.csv"
                )
                split_df.to_csv(output_path, index=False)
                written_files.append(output_path.as_posix())
    return written_files


def main() -> None:
    args = parse_args()
    df, exclusions, source_audit = load_and_clean_source(
        args.source_csv, args.conflict_policy
    )
    split_indices, validation_fold, fold_summaries = make_fold_indices(
        df, args.folds, args.seed
    )

    written_files = write_variant_folds(
        args.output_root,
        df,
        split_indices,
        args.nas_root,
        args.folds,
    )

    assignments = df.loc[:, ["slide_id", "label", "type", "patient_id"]].copy()
    assignments["validation_fold"] = validation_fold
    assignments_path = (
        args.output_root / "wsi_representation_fold_assignments.csv"
    )
    assignments.to_csv(assignments_path, index=False)

    exclusions_path = args.output_root / "wsi_representation_exclusions.csv"
    exclusions.to_csv(exclusions_path, index=False)

    summary = {
        "purpose": "development-only WSI representation selection",
        "source_csv": args.source_csv.as_posix(),
        "independent_test_data_included": False,
        "patient_id_rule": (
            "remove terminal 有癌/无癌 filename annotation, then take the "
            "substring before the first '.'"
        ),
        "seed": args.seed,
        "folds": args.folds,
        "splitter": (
            "deterministic greedy patient-group assignment balancing "
            "label-0 slides, label-1 slides, and patient counts"
        ),
        "nas_root": args.nas_root,
        "source_audit": source_audit,
        "fold_summaries": fold_summaries,
        "comparisons": all_variants(),
        "generated_fold_csv_count": len(written_files),
        "generated_fold_csvs": written_files,
    }
    summary_path = args.output_root / "wsi_representation_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
