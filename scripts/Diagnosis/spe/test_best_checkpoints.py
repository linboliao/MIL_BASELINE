"""Evaluate every MIL architecture's five best checkpoints via test_mil.py.

Each fold-specific best checkpoint is tested independently. The numeric
positive-class probabilities are then averaged across five folds to produce
one model-level prediction file and one compact metrics row per architecture.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.Diagnosis.spe.run import patient_id_from_slide
from scripts.Diagnosis.spe.summarize_performance import calculate_metrics
from utils.yaml_utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-dataset-root", type=Path)
    source.add_argument("--test-dataset-csv", type=Path)
    parser.add_argument("--target-name", required=True)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/Diagnosis/MIL"),
    )
    parser.add_argument("--configs", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("result/Diagnosis/ModelTest"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def atomic_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output_path)


def atomic_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def fold_number(path: Path) -> int:
    match = re.search(r"_(\d+)fold\.csv$", path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse fold number from: {path}")
    return int(match.group(1))


def test_fold_csvs(root: Path) -> dict[int, Path]:
    paths = {
        fold_number(path): path
        for path in root.glob("*.csv")
        if re.search(r"_(\d+)fold\.csv$", path.name, flags=re.IGNORECASE)
    }
    if sorted(paths) != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected test folds 1..5 in {root}, found {sorted(paths)}")
    return paths


def best_checkpoints(run_dir: Path) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for fold in range(1, 6):
        matches = sorted((run_dir / f"fold_{fold}").glob("Best_EPOCH_*.pth"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one best checkpoint in {run_dir / f'fold_{fold}'}, "
                f"found {len(matches)}."
            )
        checkpoints[fold] = matches[0]
    return checkpoints


def latest_complete_run(config) -> tuple[Path, dict[int, Path]]:
    model_root = (
        resolve_path(config.Logs.log_root_dir)
        / str(config.Dataset.DATASET_NAME)
        / str(config.General.MODEL_NAME)
    )
    candidates = sorted(
        [path for path in model_root.glob(f"seed_{config.General.seed}_*") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for candidate in candidates:
        try:
            return candidate, best_checkpoints(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No complete five-fold run found for {config.General.MODEL_NAME}: {model_root}")


def test_metadata(dataset_csv: Path) -> pd.DataFrame:
    source = pd.read_csv(
        dataset_csv,
        dtype={"test_slide_path": "string", "test_type": "string", "center": "string"},
    )
    required = {"test_slide_path", "test_label"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{dataset_csv} is missing columns: {sorted(missing)}")
    frame = source.loc[source["test_slide_path"].notna()].copy()
    frame["slide_id"] = frame["test_slide_path"].map(
        lambda value: PurePosixPath(str(value).replace("\\", "/")).stem
    )
    frame["patient_id"] = frame["slide_id"].map(patient_id_from_slide)
    frame["label"] = frame["test_label"].astype(int)
    frame["type"] = (
        frame["test_type"].astype("string")
        if "test_type" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    frame["center"] = (
        frame["center"].astype("string")
        if "center" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    result = frame[["slide_id", "patient_id", "type", "center", "label"]].reset_index(drop=True)
    if result["slide_id"].duplicated().any():
        raise ValueError(f"Duplicate slide IDs in test cohort: {dataset_csv}")
    return result


def load_fold_prediction(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"slide_id": "string"})
    required = {"slide_id", "label", "prob_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing numeric prediction columns: {sorted(missing)}")
    observed = frame[["slide_id", "label"]].copy()
    observed["label"] = observed["label"].astype(int)
    expected = metadata[["slide_id", "label"]].copy()
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    probabilities = frame["prob_1"].astype(float)
    if not probabilities.between(0.0, 1.0).all():
        raise ValueError(f"Invalid probabilities in {path}")
    return frame


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1.")
    if args.configs is None:
        config_paths = sorted(resolve_path(args.config_dir).glob("*.yaml"))
    else:
        config_paths = [resolve_path(path) for path in args.configs]
    if not config_paths:
        raise FileNotFoundError("No MIL config files selected.")

    if args.test_dataset_csv is not None:
        standalone = resolve_path(args.test_dataset_csv)
        if not standalone.is_file():
            raise FileNotFoundError(standalone)
        test_folds = {fold: standalone for fold in range(1, 6)}
    else:
        test_folds = test_fold_csvs(resolve_path(args.test_dataset_root))
    metadata = test_metadata(test_folds[1])
    output_root = resolve_path(args.output_root) / args.target_name

    summary_rows = []
    manifests = {}
    for config_path in config_paths:
        config = read_yaml(str(config_path))
        model_name = str(config.General.MODEL_NAME)
        run_dir, checkpoints = latest_complete_run(config)
        model_root = output_root / model_name / run_dir.name
        fold_probabilities = []
        for fold in range(1, 6):
            fold_dir = model_root / f"fold_{fold}"
            prediction_path = fold_dir / "Infer_Result.csv"
            usable_cache = False
            if prediction_path.is_file() and not args.overwrite:
                try:
                    prediction = load_fold_prediction(prediction_path, metadata)
                    usable_cache = True
                except (ValueError, AssertionError):
                    usable_cache = False
            if not usable_cache:
                command = [
                    sys.executable,
                    str(REPO_ROOT / "test_mil.py"),
                    "--yaml_path",
                    str(config_path),
                    "--test_dataset_csv",
                    str(test_folds[fold]),
                    "--model_weight_path",
                    str(checkpoints[fold]),
                    "--test_log_dir",
                    str(fold_dir),
                    "--device",
                    args.device,
                ]
                print(f"Testing {model_name} fold {fold}: {checkpoints[fold].name}")
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                prediction = load_fold_prediction(prediction_path, metadata)
            else:
                print(f"Reusing {prediction_path}")
            fold_probabilities.append(prediction["prob_1"].to_numpy(dtype=float))

        matrix = np.column_stack(fold_probabilities)
        ensemble = metadata.copy()
        for fold in range(1, 6):
            ensemble[f"prob_1_fold_{fold}"] = matrix[:, fold - 1]
        ensemble["prob_1"] = matrix.mean(axis=1)
        ensemble["prob_0"] = 1.0 - ensemble["prob_1"]
        ensemble["fold_std"] = matrix.std(axis=1, ddof=0)
        ensemble["prediction"] = (ensemble["prob_1"] >= args.threshold).astype(int)
        prediction_output = model_root / "fold_ensemble_predictions.csv"
        atomic_csv(ensemble, prediction_output)

        metrics = calculate_metrics(
            ensemble["label"].to_numpy(),
            ensemble["prob_1"].to_numpy(),
            args.threshold,
        )
        summary_rows.append(
            {
                "target": args.target_name,
                "model": model_name,
                "training_run": run_dir.name,
                "n_slides": len(ensemble),
                "n_patients": ensemble["patient_id"].nunique(),
                "threshold": args.threshold,
                **metrics,
                "prediction_csv": str(prediction_output),
            }
        )
        manifests[model_name] = {
            "config": str(config_path),
            "training_run": str(run_dir),
            "checkpoints": {str(fold): str(path) for fold, path in checkpoints.items()},
            "prediction_csv": str(prediction_output),
        }

    summary_path = output_root / "model_metrics.csv"
    summary = pd.DataFrame(summary_rows)
    if summary_path.is_file():
        previous = pd.read_csv(summary_path)
        replaced_models = set(summary["model"].astype(str))
        previous = previous.loc[~previous["model"].astype(str).isin(replaced_models)]
        summary = pd.concat([previous, summary], ignore_index=True)
    summary = summary.sort_values(
        ["balanced_accuracy", "roc_auc"], ascending=False
    )
    atomic_csv(summary, summary_path)
    manifest_path = output_root / "manifest.json"
    existing_models = {}
    if manifest_path.is_file():
        existing_models = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("models", {})
    existing_models.update(manifests)
    atomic_json(
        {
            "target": args.target_name,
            "test_dataset_csvs": {str(fold): str(path) for fold, path in test_folds.items()},
            "threshold": args.threshold,
            "models": existing_models,
            "summary_csv": str(summary_path),
        },
        manifest_path,
    )
    print(f"Best-checkpoint model test complete: {summary_path}")


if __name__ == "__main__":
    main()
