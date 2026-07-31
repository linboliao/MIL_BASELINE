"""Generate pooled out-of-fold predictions for one WSI representation config."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.model_utils import get_model_from_yaml
from utils.wsi_utils import WSI_Dataset
from utils.yaml_utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run each fold's best checkpoint on that fold's validation WSIs "
            "and combine the predictions into one OOF CSV."
        )
    )
    parser.add_argument("--yaml-path", type=Path, required=True)
    parser.add_argument(
        "--training-run-dir",
        type=Path,
        help="Specific seed_* training directory. Defaults to the latest complete run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("result/Diagnosis/Mag/OOF"),
    )
    parser.add_argument(
        "--assignment-csv",
        type=Path,
        default=Path("datasets/Diagnosis/wsi_representation_fold_assignments.csv"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload validation features into RAM before inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute fold predictions that already exist.",
    )
    return parser.parse_args()


def fold_number(path: Path) -> int:
    match = re.search(r"_(\d+)fold\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot determine fold number from: {path}")
    return int(match.group(1))


def find_fold_csvs(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    fold_csvs = sorted(dataset_root.glob("*.csv"), key=fold_number)
    if not fold_csvs:
        raise FileNotFoundError(f"No fold CSV files found in: {dataset_root}")
    numbers = [fold_number(path) for path in fold_csvs]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"Fold numbering is not consecutive: {numbers}")
    return fold_csvs


def checkpoints_for_run(run_dir: Path, fold_count: int) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for fold in range(1, fold_count + 1):
        fold_dir = run_dir / f"fold_{fold}"
        matches = sorted(fold_dir.glob("Best_EPOCH_*.pth"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one best checkpoint in {fold_dir}, "
                f"found {len(matches)}."
            )
        checkpoints[fold] = matches[0]
    return checkpoints


def find_latest_complete_run(
    config, fold_count: int, explicit_run_dir: Path | None
) -> tuple[Path, dict[int, Path]]:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir
        return run_dir, checkpoints_for_run(run_dir, fold_count)

    model_root = (
        Path(config.Logs.log_root_dir)
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
            return candidate, checkpoints_for_run(candidate, fold_count)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No complete {fold_count}-fold training run found in: {model_root}"
    )


def load_state_dict(checkpoint: Path):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        path,
        dtype={"slide_id": "string", "patient_id": "string", "type": "string"},
    )
    required = {"slide_id", "patient_id", "type", "label", "validation_fold"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Assignment CSV is missing columns: {sorted(missing)}")
    if metadata["slide_id"].duplicated().any():
        raise ValueError("Assignment CSV contains duplicate slide_id values.")
    metadata["label"] = metadata["label"].astype(int)
    metadata["validation_fold"] = metadata["validation_fold"].astype(int)
    return metadata


def infer_fold(
    config,
    fold_csv: Path,
    checkpoint: Path,
    fold: int,
    device: torch.device,
    metadata_by_slide: pd.DataFrame,
    comparison: str,
    variant: str,
    training_run: str,
    num_workers: int,
    preload: bool,
) -> pd.DataFrame:
    dataset = WSI_Dataset(str(fold_csv), "val", preload=preload)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = get_model_from_yaml(config)
    model.load_state_dict(load_state_dict(checkpoint))
    model = model.to(device).eval()

    rows: list[dict] = []
    with torch.inference_mode():
        for features, label, slide_ids in loader:
            slide_id = str(slide_ids[0])
            output = model(features.to(device, non_blocking=True).float())
            if not isinstance(output, dict) or "logits" not in output:
                raise TypeError(
                    f"{config.General.MODEL_NAME} did not return a logits dictionary."
                )
            logits = output["logits"].reshape(-1, int(config.General.num_classes))[0]
            probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            if len(probabilities) != 2:
                raise ValueError("This OOF workflow expects binary probabilities.")

            if slide_id not in metadata_by_slide.index:
                raise ValueError(f"Slide missing from assignment CSV: {slide_id}")
            metadata = metadata_by_slide.loc[slide_id]
            observed_label = int(label.item())
            if observed_label != int(metadata["label"]):
                raise ValueError(
                    f"Label mismatch for {slide_id}: fold={observed_label}, "
                    f"assignment={metadata['label']}"
                )
            if fold != int(metadata["validation_fold"]):
                raise ValueError(
                    f"Fold mismatch for {slide_id}: predicted in fold {fold}, "
                    f"assignment says fold {metadata['validation_fold']}"
                )

            rows.append(
                {
                    "comparison": comparison,
                    "variant": variant,
                    "training_run": training_run,
                    "fold": fold,
                    "slide_id": slide_id,
                    "patient_id": str(metadata["patient_id"]),
                    "type": str(metadata["type"]),
                    "label": observed_label,
                    "prob_0": float(probabilities[0]),
                    "prob_1": float(probabilities[1]),
                    "prediction": int(probabilities[1] >= 0.5),
                    "checkpoint": checkpoint.name,
                }
            )

    del loader, dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def validate_fold_predictions(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    fold: int,
    comparison: str,
    variant: str,
    training_run: str,
    source: Path,
) -> pd.DataFrame:
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
        "checkpoint",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["slide_id"] = frame["slide_id"].astype(str)
    expected = metadata.loc[metadata["validation_fold"] == fold].copy()
    expected["slide_id"] = expected["slide_id"].astype(str)
    if set(frame["slide_id"]) != set(expected["slide_id"]):
        raise ValueError(f"{source} does not contain exactly the slides for fold {fold}.")
    if frame["slide_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate slide IDs.")
    if set(frame["comparison"].astype(str)) != {comparison}:
        raise ValueError(f"{source} has the wrong comparison value.")
    if set(frame["variant"].astype(str)) != {variant}:
        raise ValueError(f"{source} has the wrong variant value.")
    if set(frame["training_run"].astype(str)) != {training_run}:
        raise ValueError(f"{source} was generated from a different training run.")
    if set(frame["fold"].astype(int)) != {fold}:
        raise ValueError(f"{source} has the wrong fold value.")

    observed_labels = frame.set_index("slide_id")["label"].astype(int).sort_index()
    expected_labels = expected.set_index("slide_id")["label"].astype(int).sort_index()
    if not observed_labels.equals(expected_labels):
        raise ValueError(f"{source} contains labels that differ from the assignment CSV.")
    return frame


def main() -> None:
    args = parse_args()
    config = read_yaml(str(args.yaml_path))
    if str(config.General.MODEL_NAME) != "AB_MIL":
        raise ValueError(
            "The WSI representation configs are expected to use MODEL_NAME=AB_MIL."
        )

    comparison = str(config.Dataset.comparison_group)
    variant = str(config.Dataset.variant)
    fold_csvs = find_fold_csvs(Path(config.Dataset.dataset_root_dir))
    run_dir, checkpoints = find_latest_complete_run(
        config, len(fold_csvs), args.training_run_dir
    )
    metadata = read_metadata(args.assignment_csv)
    metadata_by_slide = metadata.set_index("slide_id", drop=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = args.output_root / comparison / variant / run_dir.name
    all_folds: list[pd.DataFrame] = []
    for fold_csv in fold_csvs:
        fold = fold_number(fold_csv)
        fold_output = output_dir / f"fold_{fold}" / "predictions.csv"
        if fold_output.exists() and not args.overwrite:
            print(f"Reusing existing predictions: {fold_output}")
            fold_frame = pd.read_csv(
                fold_output,
                dtype={"slide_id": "string", "patient_id": "string"},
            )
        else:
            print(
                f"OOF fold {fold}: dataset={fold_csv} checkpoint={checkpoints[fold]}"
            )
            fold_frame = infer_fold(
                config=config,
                fold_csv=fold_csv,
                checkpoint=checkpoints[fold],
                fold=fold,
                device=device,
                metadata_by_slide=metadata_by_slide,
                comparison=comparison,
                variant=variant,
                training_run=run_dir.name,
                num_workers=args.num_workers,
                preload=args.preload,
            )
            write_csv_atomic(fold_frame, fold_output)
        fold_frame = validate_fold_predictions(
            frame=fold_frame,
            metadata=metadata,
            fold=fold,
            comparison=comparison,
            variant=variant,
            training_run=run_dir.name,
            source=fold_output,
        )
        all_folds.append(fold_frame)

    oof = pd.concat(all_folds, ignore_index=True)
    if oof["slide_id"].duplicated().any():
        duplicates = oof.loc[oof["slide_id"].duplicated(False), "slide_id"].tolist()
        raise ValueError(f"OOF predictions contain duplicate slides: {duplicates[:5]}")
    expected_slides = set(metadata["slide_id"])
    observed_slides = set(oof["slide_id"].astype(str))
    if observed_slides != expected_slides:
        missing = sorted(expected_slides - observed_slides)
        extra = sorted(observed_slides - expected_slides)
        raise ValueError(
            f"OOF coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if not ((oof["prob_0"] + oof["prob_1"] - 1.0).abs() < 1e-5).all():
        raise ValueError("At least one probability row does not sum to one.")

    oof = oof.sort_values(["fold", "slide_id"]).reset_index(drop=True)
    combined_path = output_dir / "oof_predictions.csv"
    write_csv_atomic(oof, combined_path)

    manifest = {
        "yaml_path": args.yaml_path.as_posix(),
        "comparison": comparison,
        "variant": variant,
        "training_run_dir": run_dir.as_posix(),
        "output_csv": combined_path.as_posix(),
        "fold_count": len(fold_csvs),
        "slide_count": len(oof),
        "patient_count": int(oof["patient_id"].nunique()),
        "device": str(device),
        "checkpoints": {
            str(fold): path.as_posix() for fold, path in checkpoints.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(oof)} OOF predictions to: {combined_path}")


if __name__ == "__main__":
    main()
