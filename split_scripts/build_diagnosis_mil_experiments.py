"""Build the shared Diagnosis MIL folds and SPE training configurations.

The representation is intentionally fixed to the configuration selected during
model development: H-optimus-1 features at 10x with Reinhard normalization.
All MIL architectures receive byte-identical folds so that model comparisons
and ensemble diversity are not confounded by different data partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml


MODEL_SETTINGS = {
    "AB_MIL": {
        "in_dim": 1536,
        "L": 512,
        "D": 128,
        "dropout": 0.1,
        "act": "relu",
    },
    "CLAM_SB_MIL": {
        "in_dim": 1536,
        "dropout": 0.1,
        "act": "relu",
        "k_sample": 8,
        "size_arg": "small",
        "instance_loss_fn": "ce",
        "subtyping": True,
        "instance_eval": True,
        "gate": True,
        "bag_weight": 0.7,
    },
    "CLAM_MB_MIL": {
        "in_dim": 1536,
        "dropout": 0.1,
        "act": "relu",
        "k_sample": 8,
        "size_arg": "small",
        "instance_loss_fn": "ce",
        "subtyping": True,
        "instance_eval": True,
        "gate": True,
        "bag_weight": 0.7,
    },
    "TRANS_MIL": {
        "in_dim": 1536,
        "dropout": 0.1,
        "act": "relu",
    },
    "WIKG_MIL": {
        "in_dim": 1536,
        "dim_hidden": 512,
        "topk": 6,
        "agg_type": "bi-interaction",
        "pool": "attn",
        "dropout": 0.1,
        "act": "LeakyReLU",
    },
    "MAMBA2D_MIL": {
        "in_dim": 1536,
        "d_model": 512,
        "d_state": 16,
        "n_layers": 2,
        "grid_size": None,
        "dropout": 0.1,
        "act": "gelu",
    },
    "AEM_MIL": {
        "in_dim": 1536,
        "L": 512,
        "D": 128,
        "dropout": 0.1,
        "act": "relu",
        "temperature": 1.0,
        "lambda_entropy": 0.1,
    },
    "MICO_MIL": {
        "in_dim": 1536,
        "embedding_dim": 512,
        "num_clusters": 64,
        "num_enhancers": 3,
        "dropout": 0.25,
        "hard": False,
        "similarity_method": "l2",
        "cluster_init_path": None,
    },
    "MSM_MIL": {
        "in_dim": 1536,
        "dropout": 0.1,
        "act": "relu",
        "layer": 2,
        "rate": 10,
        "mamba_type": "SRMamba",
    },
    "TDA_MIL": {
        "in_dim": 1536,
        "embed_dim": 512,
        "num_layers": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "td_mlp_ratio": 2.0,
        "clamp_min": 0.0,
        "clamp_max": 1.0,
        "force_cls_score": 1.0,
        "share_weights_step12": True,
        "max_seq_len": 2048,
    },
    "GDF_MIL": {
        "in_dim": 1536,
        "hid_dim": 256,
        "out_dim": 128,
        "k_components": 10,
        "k_neighbors": 10,
        "dropout": 0.1,
        "act": "leaky_relu",
        "lambda_smooth": 0.0,
        "lambda_nce": 0.0,
    },
}

REQUIRED_COLUMNS = {
    "train_slide_path",
    "train_label",
    "val_slide_path",
    "val_label",
    "test_slide_path",
    "test_label",
}
EXPECTED_PATH_PART = "/feat_0_448/stains/Reinhard/pt_files/h-optimus-1/"
DEFAULT_TEST_PREFIX = (
    "/NAS145/liaolinbo/Data/MXB/CLS测试/feat_0_448/stains/Reinhard/"
    "pt_files/h-optimus-1/"
)


def common_config(model_name: str, dataset_root: str) -> dict:
    model = dict(MODEL_SETTINGS[model_name])
    model.update(
        {
            "optimizer": {
                "which": "adam",
                "adam_config": {"lr": 0.0002, "weight_decay": 0.00001},
                "adamw_config": {"lr": 0.0002, "weight_decay": 0.00001},
            },
            "criterion": {"loss": "ce"},
            "scheduler": {
                "warmup": 2,
                "which": "step",
                "step_config": {"step_size": 3, "gamma": 0.9},
                "multi_step_config": {
                    "milestones": [20, 30, 40],
                    "gamma": 0.9,
                },
                "exponential_config": {"gamma": 0.9},
                "cosine_config": {"T_max": 10, "eta_min": 0.0001},
            },
        }
    )
    return {
        "General": {
            "MODEL_NAME": model_name,
            "seed": 42,
            "num_classes": 2,
            "num_epochs": 50,
            "device": 0,
            "num_workers": 4,
            "best_model_metric": "macro_f1",
            "earlystop": {
                "use": True,
                "patience": 30,
                "metric": "macro_f1",
            },
            "checkpoint": {
                "save_mode": "every_epoch",
                "spe": {
                    "metric": "macro_f1",
                    "stability_threshold": 0.003,
                    "min_consecutive": 5,
                    "max_checkpoints": 5,
                },
            },
        },
        "Dataset": {
            "DATASET_NAME": "HOptimus1_10x_Reinhard",
            "comparison_group": "MIL",
            "representation": "h-optimus-1",
            "magnification": "10x",
            "stain_normalization": "Reinhard",
            "dataset_csv_path": None,
            "dataset_root_dir": dataset_root,
            "balanced_sampler": {"use": False, "replacement": True},
        },
        "Logs": {"log_root_dir": "result/Diagnosis/MIL"},
        "Model": model,
    }


def validate_fold(path: Path) -> dict:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    split_counts = {}
    for split in ("train", "val"):
        paths = frame[f"{split}_slide_path"].dropna().astype(str)
        if paths.empty:
            raise ValueError(f"{path} has an empty {split} split")
        invalid = paths[~paths.str.contains(EXPECTED_PATH_PART, regex=False)]
        if not invalid.empty:
            raise ValueError(
                f"{path} contains paths outside the selected representation: "
                f"{invalid.iloc[0]}"
            )
        split_counts[split] = int(len(paths))

    test_count = int(frame["test_slide_path"].notna().sum())
    if test_count:
        raise ValueError(f"{path} unexpectedly contains {test_count} test slides")
    split_counts["test"] = test_count
    return split_counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_independent_test(test_csv: Path, test_prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(test_csv)
    required = {"test_slide_path", "test_label", "test_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{test_csv} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{test_csv} is empty")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{test_csv} contains empty test fields")
    filenames = frame["test_slide_path"].astype(str)
    if filenames.duplicated().any():
        raise ValueError(f"{test_csv} contains duplicate test filenames")
    if filenames.str.contains(r"[/\\]", regex=True).any():
        raise ValueError("test_slide_path must contain filenames, not full paths")
    prefix = test_prefix.rstrip("/") + "/"
    result = frame[["test_slide_path", "test_label", "test_type"]].copy()
    result["test_slide_path"] = prefix + filenames
    return result


def normalize_slide_id(path: str) -> str:
    stem = Path(str(path)).stem.rstrip()
    return re.sub(r"[\u4e00-\u9fff\s]+$", "", stem)


def build_overlap_audit(
    development_fold: Path, independent_test: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    development = pd.read_csv(development_fold)
    development_records = []
    for split in ("train", "val"):
        for path, label in zip(
            development[f"{split}_slide_path"],
            development[f"{split}_label"],
        ):
            if pd.isna(path):
                continue
            slide_id = normalize_slide_id(path)
            development_records.append(
                {
                    "slide_id": slide_id,
                    "patient_id": slide_id.split(".", 1)[0],
                    "path": str(path),
                    "split": split,
                    "label": int(label),
                }
            )

    by_slide = {}
    by_patient = {}
    for record in development_records:
        by_slide.setdefault(record["slide_id"], []).append(record)
        by_patient.setdefault(record["patient_id"], []).append(record)

    rows = []
    patient_ids = set()
    exact_rows = 0
    for test_row in independent_test.itertuples(index=False):
        slide_id = normalize_slide_id(test_row.test_slide_path)
        patient_id = slide_id.split(".", 1)[0]
        exact_matches = by_slide.get(slide_id, [])
        patient_matches = by_patient.get(patient_id, [])
        if exact_matches:
            level = "exact_wsi"
            matches = exact_matches
            exact_rows += 1
        elif patient_matches:
            level = "patient_only"
            matches = patient_matches
        else:
            continue
        patient_ids.add(patient_id)
        rows.append(
            {
                "overlap_level": level,
                "patient_id": patient_id,
                "normalized_test_slide_id": slide_id,
                "test_slide_path": test_row.test_slide_path,
                "test_label": int(test_row.test_label),
                "test_type": test_row.test_type,
                "development_slide_paths": " | ".join(
                    sorted({record["path"] for record in matches})
                ),
                "development_splits": " | ".join(
                    sorted({record["split"] for record in matches})
                ),
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "independence_pass": audit.empty,
        "overlapping_test_rows": int(len(audit)),
        "exact_wsi_rows": int(exact_rows),
        "patient_only_rows": int(len(audit) - exact_rows),
        "overlapping_patient_ids": int(len(patient_ids)),
        "clean_test_rows": int(len(independent_test) - len(audit)),
        "audit_csv": "datasets/Diagnosis/MIL/independent_test_overlap_audit.csv",
    }
    return audit, summary


def build(
    source_root: Path,
    dataset_root: Path,
    config_root: Path,
    test_csv: Path,
    test_prefix: str,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    independent_test = load_independent_test(test_csv, test_prefix)
    first_source_fold = source_root / "Total_5-fold_Reinhard_1fold.csv"
    overlap_audit, overlap_summary = build_overlap_audit(
        first_source_fold, independent_test
    )
    overlap_audit.to_csv(
        dataset_root / "independent_test_overlap_audit.csv", index=False
    )
    manifest = {
        "representation": "h-optimus-1",
        "magnification": "10x",
        "stain_normalization": "Reinhard",
        "source_root": source_root.as_posix(),
        "independent_test": {
            "source_csv": test_csv.as_posix(),
            "path_prefix": test_prefix.rstrip("/") + "/",
            "slides": int(len(independent_test)),
            "label_counts": {
                str(key): int(value)
                for key, value in independent_test["test_label"]
                .value_counts()
                .sort_index()
                .items()
            },
            "type_counts": {
                str(key): int(value)
                for key, value in independent_test["test_type"]
                .value_counts()
                .sort_index()
                .items()
            },
            "overlap_audit": overlap_summary,
        },
        "folds": [],
        "models": list(MODEL_SETTINGS),
    }

    for fold in range(1, 6):
        source = source_root / f"Total_5-fold_Reinhard_{fold}fold.csv"
        target = dataset_root / f"Total_5-fold_MIL_{fold}fold.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        counts = validate_fold(source)
        frame = pd.read_csv(source)
        if len(frame) < len(independent_test):
            raise ValueError(f"{source} has too few rows for the independent test set")
        frame["test_slide_path"] = pd.NA
        frame["test_label"] = pd.NA
        frame["test_type"] = pd.NA
        test_rows = len(independent_test)
        frame.loc[: test_rows - 1, "test_slide_path"] = independent_test[
            "test_slide_path"
        ].to_numpy()
        frame.loc[: test_rows - 1, "test_label"] = independent_test[
            "test_label"
        ].to_numpy()
        frame.loc[: test_rows - 1, "test_type"] = independent_test[
            "test_type"
        ].to_numpy()
        frame.to_csv(target, index=False)
        counts["test"] = test_rows
        manifest["folds"].append(
            {
                "fold": fold,
                "file": target.name,
                "sha256": sha256(target),
                **counts,
            }
        )

    # Training must remain development-only. ``dataset_root`` contains the
    # independent test columns for locked evaluation, whereas ``source_root``
    # contains the same five train/validation folds with empty test columns.
    dataset_root_value = source_root.as_posix()
    for model_name in MODEL_SETTINGS:
        target = config_root / f"{model_name}.yaml"
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                common_config(model_name, dataset_root_value),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )

    with (dataset_root / "dataset_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/Diagnosis/Stains/Reinhard"),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/Diagnosis/MIL")
    )
    parser.add_argument(
        "--config-root", type=Path, default=Path("configs/Diagnosis/MIL")
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("datasets/Diagnosis/test_wo_type.csv"),
    )
    parser.add_argument("--test-prefix", default=DEFAULT_TEST_PREFIX)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    build(
        cli_args.source_root,
        cli_args.dataset_root,
        cli_args.config_root,
        cli_args.test_csv,
        cli_args.test_prefix,
    )
