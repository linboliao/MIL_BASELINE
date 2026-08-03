"""Rebuild validation/test CSV logs from per-epoch MIL checkpoints.

This utility is intended for an interrupted training run whose
``epoch_checkpoints`` directory survived but whose Log/Best_Log CSV files did
not.  It currently supports the standard validation path used by TDA_MIL.
Training loss cannot be reconstructed from model weights and is left empty.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import add_epoch_info_log, init_epoch_info_log, set_global_seed
from utils.loop_utils import val_loop
from utils.model_utils import get_criterion, get_model_from_yaml
from utils.wsi_utils import WSI_Dataset
from utils.yaml_utils import read_yaml


CHECKPOINT_RE = re.compile(r"^Epoch_(\d+)\.pth$")
FOLD_RE = re.compile(r"^fold_(\d+)$", re.IGNORECASE)


def discover_checkpoints(path: Path) -> list[tuple[int, Path]]:
    """Return unique ``(epoch, path)`` pairs sorted by epoch."""
    checkpoint_dir = path / "epoch_checkpoints" if (path / "epoch_checkpoints").is_dir() else path
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for item in checkpoint_dir.iterdir():
        match = CHECKPOINT_RE.match(item.name)
        if not match or not item.is_file():
            continue
        epoch = int(match.group(1))
        if epoch in seen:
            raise ValueError(f"Duplicate checkpoint for epoch {epoch}: {checkpoint_dir}")
        seen.add(epoch)
        checkpoints.append((epoch, item))
    checkpoints.sort(key=lambda pair: pair[0])
    if not checkpoints:
        raise FileNotFoundError(f"No Epoch_XXXX.pth files found in: {checkpoint_dir}")
    return checkpoints


def _fold_sort_key(path: Path) -> tuple[float, str]:
    match = re.search(r"_(\d+)fold\.csv$", path.name, re.IGNORECASE)
    return (int(match.group(1)), path.name) if match else (float("inf"), path.name)


def _infer_fold(checkpoint_path: Path) -> int | None:
    for part in reversed(checkpoint_path.resolve().parts):
        match = FOLD_RE.match(part)
        if match:
            return int(match.group(1))
    return None


def resolve_dataset_csv(config, checkpoint_path: Path, explicit: str | None) -> Path:
    """Resolve the fold CSV, preferring an explicit command-line path."""
    if explicit:
        result = Path(explicit)
    elif config.Dataset.dataset_csv_path not in (None, {}):
        result = Path(str(config.Dataset.dataset_csv_path))
    else:
        fold = _infer_fold(checkpoint_path)
        if fold is None:
            raise ValueError(
                "Cannot infer the fold from the checkpoint path. Pass --dataset-csv explicitly."
            )
        root = Path(str(config.Dataset.dataset_root_dir))
        candidates = sorted(
            (item for item in root.iterdir() if item.is_file() and item.suffix.lower() == ".csv"),
            key=_fold_sort_key,
        )
        if fold > len(candidates):
            raise ValueError(f"fold_{fold} has no matching CSV under {root}")
        result = candidates[fold - 1]
    if not result.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {result}")
    return result


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def best_row(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    column = metric if metric.startswith("val_") else f"val_{metric}"
    if metric == "val_loss":
        column = "val_loss"
    if column not in frame.columns:
        raise KeyError(f"Best-model metric column is absent from the recovered log: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Best-model metric has no finite values: {column}")
    index = values.idxmin() if column == "val_loss" else values.idxmax()
    return frame.loc[[index]]


def _save_outputs(frame: pd.DataFrame, log_path: Path, best_path: Path, metric: str) -> None:
    frame = frame.sort_values("epoch").reset_index(drop=True)
    _atomic_write_csv(frame, log_path)
    _atomic_write_csv(best_row(frame, metric), best_path)


def recover(args: argparse.Namespace) -> tuple[Path, Path]:
    config = read_yaml(args.yaml_path)
    if str(config.General.MODEL_NAME).lower() != "tda_mil":
        raise ValueError("This recovery script currently supports TDA_MIL checkpoints only.")

    checkpoint_arg = Path(args.checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_arg)
    if args.expected_epochs and len(checkpoints) != args.expected_epochs:
        raise ValueError(
            f"Expected {args.expected_epochs} checkpoints, found {len(checkpoints)} in {checkpoint_arg}"
        )
    if args.expected_epochs:
        actual_epochs = [epoch for epoch, _ in checkpoints]
        expected_epochs = list(range(1, args.expected_epochs + 1))
        if actual_epochs != expected_epochs:
            raise ValueError(
                "Checkpoint epochs must be continuous from 1 through "
                f"{args.expected_epochs}; found: {actual_epochs}"
            )

    dataset_csv = resolve_dataset_csv(config, checkpoint_arg, args.dataset_csv)
    output_dir = Path(args.output_dir) if args.output_dir else (
        checkpoint_arg.parent if checkpoint_arg.name == "epoch_checkpoints" else checkpoint_arg
    )
    dataset_name = str(config.Dataset.DATASET_NAME)
    model_name = str(config.General.MODEL_NAME)
    seed = int(config.General.seed)
    log_path = output_dir / f"Log_seed{seed}_{dataset_name}_{model_name}.csv"
    best_path = output_dir / f"Best_Log_seed{seed}_{dataset_name}_{model_name}.csv"

    recovered = pd.DataFrame()
    if log_path.exists() and not args.restart:
        recovered = pd.read_csv(log_path)
        if "epoch" not in recovered.columns:
            raise ValueError(f"Existing recovery log has no epoch column: {log_path}")
        recovered = recovered[recovered["epoch"].isin([epoch for epoch, _ in checkpoints])]
        recovered = recovered.drop_duplicates(subset="epoch", keep="last")
    elif (log_path.exists() or best_path.exists()) and not args.restart:
        raise FileExistsError(f"Refusing to replace existing output. Use --restart: {output_dir}")

    completed = set(pd.to_numeric(recovered.get("epoch", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
    pending = [(epoch, path) for epoch, path in checkpoints if epoch not in completed]

    set_global_seed(seed)
    device = torch.device(args.device or (f"cuda:{config.General.device}" if torch.cuda.is_available() else "cpu"))
    model = get_model_from_yaml(config).to(device)
    criterion = get_criterion(config.Model.criterion)
    val_loader = DataLoader(
        WSI_Dataset(str(dataset_csv), "val"), batch_size=1, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        WSI_Dataset(str(dataset_csv), "test"), batch_size=1, shuffle=False,
        num_workers=args.num_workers,
    )
    if len(val_loader.dataset) == 0 or len(test_loader.dataset) == 0:
        raise ValueError("Both validation and test splits must be non-empty for this recovery.")

    print(f"Found {len(checkpoints)} checkpoints; {len(pending)} epochs need evaluation")
    print(f"Dataset CSV: {dataset_csv}")
    print(f"Device: {device}")
    for position, (epoch, checkpoint_path) in enumerate(pending, start=1):
        print(f"[{position}/{len(pending)}] Evaluating epoch {epoch}: {checkpoint_path.name}")
        # Do not pass ``weights_only`` here: the training server may use a
        # PyTorch release older than the version that introduced that option.
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(_extract_state_dict(payload), strict=True)
        val_loss, val_metrics, _ = val_loop(
            device, int(config.General.num_classes), model, val_loader, criterion
        )
        test_loss, test_metrics, _ = val_loop(
            device, int(config.General.num_classes), model, test_loader, criterion
        )
        epoch_log = init_epoch_info_log()
        add_epoch_info_log(
            epoch_log, epoch - 1, None, val_loss, test_loss, val_metrics, test_metrics
        )
        recovered = pd.concat([recovered, pd.DataFrame(epoch_log)], ignore_index=True)
        recovered = recovered.drop_duplicates(subset="epoch", keep="last")
        _save_outputs(recovered, log_path, best_path, str(config.General.best_model_metric))

    _save_outputs(recovered, log_path, best_path, str(config.General.best_model_metric))
    print(f"Recovered log: {log_path}")
    print(f"Recovered best log: {best_path}")
    return log_path, best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-path", required=True, help="The exact YAML used for training")
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Run/fold directory or its epoch_checkpoints directory",
    )
    parser.add_argument(
        "--dataset-csv",
        help="Exact fold CSV. Strongly recommended; otherwise inferred from fold_N in the path.",
    )
    parser.add_argument("--output-dir", help="Defaults to the fold/run directory")
    parser.add_argument("--device", help="For example cuda:0 or cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--expected-epochs", type=int, default=48,
        help="Fail unless this many checkpoints exist; use 0 to disable (default: 48)",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Discard an existing partial recovered Log CSV and evaluate all epochs again",
    )
    return parser


if __name__ == "__main__":
    recover(build_parser().parse_args())
