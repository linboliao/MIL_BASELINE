"""Build class-protected CP-AWA checkpoint soups from saved MIL trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ensemble.cp_awa import (  # noqa: E402
    cp_metric_from_probabilities,
    greedy_cp_awa,
    pareto_candidates,
)
from scripts.Diagnosis.spe._engine import (  # noqa: E402
    fold_csvs,
    infer_checkpoint,
    load_state_dict,
    resolve_path,
)
from utils.yaml_utils import read_yaml  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one validation-guarded CP_AWA.pth per model/fold."
    )
    parser.add_argument(
        "--spe-config",
        type=Path,
        default=Path("configs/Diagnosis/SPE/cp_awa_single_anchor.yaml"),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated architecture names.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated fold numbers.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help=(
            "Remap feature-path prefixes in a temporary dataset CSV. May be "
            "specified more than once; the source CSV is never modified."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate manifests and print candidate pools without inference.",
    )
    return parser


def _selected_names(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names:
        raise ValueError("--models did not contain an architecture name.")
    return names


def _selected_folds(value: str) -> list[int]:
    folds = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("--folds must contain values from 1 through 5.")
    return folds


def _path_mappings(values: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --path-map {value!r}; expected SOURCE=DESTINATION."
            )
        source, destination = value.split("=", 1)
        source, destination = source.strip(), destination.strip()
        if not source or not destination:
            raise ValueError(
                f"Invalid --path-map {value!r}; both prefixes are required."
            )
        mappings.append((source, destination))
    return mappings


def _remapped_dataset_csv(
    source: Path,
    mappings: list[tuple[str, str]],
    destination: Path,
) -> Path:
    if not mappings:
        return source
    frame = pd.read_csv(source)
    path_columns = [column for column in frame.columns if column.endswith("_slide_path")]
    for column in path_columns:
        values = frame[column].astype("string")
        for old_prefix, new_prefix in mappings:
            values = values.str.replace(old_prefix, new_prefix, regex=False)
        frame[column] = values
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def _atomic_torch_save(state: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _policy(settings: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "metric_tolerance": 0.005,
        "macro_f1_tolerance": 0.005,
        "candidate_sensitivity_drop_tolerance": 0.010,
        "candidate_specificity_drop_tolerance": 0.010,
        "temperature": 0.0025,
        "class_temperature": 0.020,
        "max_candidates_to_evaluate": 10,
        "max_checkpoints": 5,
        "greedy_bacc_drop_tolerance": 0.001,
        "greedy_sensitivity_drop_tolerance": 0.005,
        "greedy_specificity_drop_tolerance": 0.005,
        "greedy_log_loss_increase_tolerance": 0.005,
    }
    defaults.update(settings.get("cp_awa", {}))
    return defaults


def build_fold(
    *,
    name: str,
    architecture: dict[str, Any],
    fold: int,
    dataset_csv: Path,
    device: torch.device,
    num_workers: int,
    preload: bool,
    policy: dict[str, Any],
    overwrite: bool,
    preflight: bool,
) -> None:
    if not architecture.get("run_dir"):
        raise ValueError(f"CP-AWA requires an immutable run_dir for {name}.")
    run_dir = resolve_path(architecture["run_dir"])
    fold_dir = run_dir / f"fold_{fold}"
    source_manifest_path = fold_dir / "checkpoint_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing trajectory manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    candidates, reference = pareto_candidates(
        source_manifest,
        metric_tolerance=float(policy["metric_tolerance"]),
        macro_f1_tolerance=float(policy["macro_f1_tolerance"]),
        sensitivity_drop_tolerance=float(
            policy["candidate_sensitivity_drop_tolerance"]
        ),
        specificity_drop_tolerance=float(
            policy["candidate_specificity_drop_tolerance"]
        ),
        temperature=float(policy["temperature"]),
        class_temperature=float(policy["class_temperature"]),
        max_candidates_to_evaluate=int(policy["max_candidates_to_evaluate"]),
    )
    missing = [
        fold_dir / candidate.checkpoint
        for candidate in candidates
        if not (fold_dir / candidate.checkpoint).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"CP-AWA source checkpoints are missing: {missing}")
    epoch_text = ",".join(str(candidate.epoch) for candidate in candidates)
    print(
        f"[{name}] fold={fold} reference={reference.epoch} candidates={epoch_text}",
        flush=True,
    )
    if preflight:
        return

    output_dir = fold_dir / "cp_awa"
    checkpoint_path = output_dir / "CP_AWA.pth"
    manifest_path = output_dir / "cp_awa_manifest.json"
    if checkpoint_path.is_file() and manifest_path.is_file() and not overwrite:
        print(f"[{name}] fold={fold} already built: {checkpoint_path}", flush=True)
        return

    config = read_yaml(str(resolve_path(architecture["config"])))
    if str(config.General.MODEL_NAME) != name:
        raise ValueError(
            f"Architecture {name} does not match {architecture['config']}: "
            f"{config.General.MODEL_NAME}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = output_dir / ".CP_AWA_candidate.pth"

    def load_candidate(candidate):
        return load_state_dict(fold_dir / candidate.checkpoint)

    def evaluate_state(state, accepted):
        _atomic_torch_save(state, temporary_checkpoint)
        try:
            frame = infer_checkpoint(
                config,
                dataset_csv,
                "val",
                temporary_checkpoint,
                device,
                num_workers,
                preload,
            )
        finally:
            if temporary_checkpoint.is_file():
                temporary_checkpoint.unlink()
        metrics = cp_metric_from_probabilities(frame["label"], frame["prob_1"])
        accepted_text = ",".join(str(candidate.epoch) for candidate in accepted)
        print(
            f"[{name}] fold={fold} soup={accepted_text} "
            f"bacc={metrics.balanced_accuracy:.6f} "
            f"sens={metrics.sensitivity:.6f} spec={metrics.specificity:.6f} "
            f"logloss={metrics.log_loss:.6f}",
            flush=True,
        )
        return metrics

    result = greedy_cp_awa(
        candidates,
        reference,
        load_state=load_candidate,
        evaluate_state=evaluate_state,
        max_checkpoints=int(policy["max_checkpoints"]),
        bacc_drop_tolerance=float(policy["greedy_bacc_drop_tolerance"]),
        sensitivity_drop_tolerance=float(
            policy["greedy_sensitivity_drop_tolerance"]
        ),
        specificity_drop_tolerance=float(
            policy["greedy_specificity_drop_tolerance"]
        ),
        log_loss_increase_tolerance=float(
            policy["greedy_log_loss_increase_tolerance"]
        ),
    )
    _atomic_torch_save(result.state, checkpoint_path)
    payload = {
        "schema_version": 1,
        "strategy": "cp_awa",
        "model": name,
        "fold": fold,
        "source_manifest": str(source_manifest_path),
        "checkpoint": checkpoint_path.name,
        "policy": policy,
        "reference": reference.as_dict(),
        "candidate_pool": [candidate.as_dict() for candidate in candidates],
        "accepted_candidates": [
            candidate.as_dict() for candidate in result.accepted_candidates
        ],
        "normalized_weights": list(result.normalized_weights),
        "reference_metrics": result.reference_metrics.as_dict(),
        "final_metrics": result.final_metrics.as_dict(),
        "trace": list(result.trace),
    }
    _atomic_json(payload, manifest_path)
    print(
        f"[{name}] fold={fold} wrote {checkpoint_path} with "
        f"{len(result.accepted_candidates)} states",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    config_path = resolve_path(args.spe_config)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = settings["experiment"]
    architectures = settings["architectures"]
    requested_names = _selected_names(args.models)
    if requested_names is not None:
        available = {str(item["name"]) for item in architectures}
        missing = requested_names - available
        if missing:
            raise ValueError(f"Unknown --models values: {sorted(missing)}")
        architectures = [
            item for item in architectures if str(item["name"]) in requested_names
        ]
    folds = _selected_folds(args.folds)
    dataset_folds = fold_csvs(resolve_path(experiment["development_dataset_root"]))
    mappings = _path_mappings(args.path_map)
    device = torch.device(args.device or str(experiment.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    num_workers = int(
        experiment.get("num_workers", 0)
        if args.num_workers is None
        else args.num_workers
    )
    policy = _policy(settings)
    with tempfile.TemporaryDirectory(prefix="cp_awa_dataset_") as temporary:
        temporary_root = Path(temporary)
        effective_folds = {
            fold: _remapped_dataset_csv(
                dataset_folds[fold], mappings, temporary_root / dataset_folds[fold].name
            )
            for fold in folds
        }
        for architecture in architectures:
            name = str(architecture["name"])
            for fold in folds:
                build_fold(
                    name=name,
                    architecture=architecture,
                    fold=fold,
                    dataset_csv=effective_folds[fold],
                    device=device,
                    num_workers=num_workers,
                    preload=bool(args.preload or experiment.get("preload", False)),
                    policy=policy,
                    overwrite=bool(args.overwrite),
                    preflight=bool(args.preflight),
                )
    print("CP-AWA build complete." if not args.preflight else "CP-AWA preflight complete.")


if __name__ == "__main__":
    main()
