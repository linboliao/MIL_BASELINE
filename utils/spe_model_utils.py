"""Checkpoint persistence and stable-state selection for SPE experiments.

The default policy remains ``best_last`` for backward compatibility.  Set
``General.checkpoint.save_mode`` to ``every_epoch`` when a training trajectory
must later be reconstructed for the stability-prioritized ensemble (SPE).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


BEST_LAST = "best_last"
EVERY_EPOCH = "every_epoch"
SUPPORTED_SAVE_MODES = {BEST_LAST, EVERY_EPOCH}


@dataclass(frozen=True)
class CheckpointPolicy:
    save_mode: str = BEST_LAST
    spe_metric: str = "macro_f1"
    stability_threshold: float = 0.003
    min_consecutive: int = 5
    max_checkpoints: int = 5

    def validate(self) -> "CheckpointPolicy":
        if self.save_mode not in SUPPORTED_SAVE_MODES:
            raise ValueError(
                f"Unknown checkpoint save_mode={self.save_mode!r}. "
                f"Supported values: {sorted(SUPPORTED_SAVE_MODES)}"
            )
        if self.stability_threshold < 0:
            raise ValueError("stability_threshold must be non-negative.")
        if self.min_consecutive < 2:
            raise ValueError("min_consecutive must be at least 2.")
        if self.max_checkpoints < 1:
            raise ValueError("max_checkpoints must be at least 1.")
        return self


def _mapping_get(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        result = value.get(key, default)
    else:
        result = getattr(value, key, default)
    if result == {} or result is None:
        return default
    return result


def get_checkpoint_policy(args: Any) -> CheckpointPolicy:
    """Read checkpoint settings, falling back to the legacy best/last policy."""
    general = _mapping_get(args, "General", {})
    checkpoint = _mapping_get(general, "checkpoint", {})
    spe = _mapping_get(checkpoint, "spe", {})
    policy = CheckpointPolicy(
        save_mode=str(_mapping_get(checkpoint, "save_mode", BEST_LAST)),
        spe_metric=str(_mapping_get(spe, "metric", "macro_f1")),
        stability_threshold=float(
            _mapping_get(spe, "stability_threshold", 0.003)
        ),
        min_consecutive=int(_mapping_get(spe, "min_consecutive", 5)),
        max_checkpoints=int(_mapping_get(spe, "max_checkpoints", 5)),
    )
    return policy.validate()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_torch_save(state_dict: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, output_path)


def _atomic_json_save(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)


def _metadata(args: Any, policy: CheckpointPolicy) -> dict[str, Any]:
    general = _mapping_get(args, "General", {})
    dataset = _mapping_get(args, "Dataset", {})
    return {
        "schema_version": 1,
        "save_mode": policy.save_mode,
        "model_name": str(_mapping_get(general, "MODEL_NAME", "")),
        "dataset_name": str(_mapping_get(dataset, "DATASET_NAME", "")),
        "seed": _json_safe(_mapping_get(general, "seed", None)),
        "fold": _json_safe(_mapping_get(dataset, "now_fold", None)),
        "spe": {
            "metric": policy.spe_metric,
            "stability_threshold": policy.stability_threshold,
            "min_consecutive": policy.min_consecutive,
            "max_checkpoints": policy.max_checkpoints,
        },
        "epochs": [],
    }


def _manifest_path(args: Any) -> Path:
    return Path(args.Logs.now_log_dir) / "checkpoint_manifest.json"


def _load_manifest(
    args: Any, policy: CheckpointPolicy
) -> tuple[dict[str, Any], Path]:
    manifest_path = _manifest_path(args)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("save_mode") != policy.save_mode:
            raise ValueError(
                "Checkpoint save_mode changed inside one training directory: "
                f"{manifest.get('save_mode')} -> {policy.save_mode}"
            )
    else:
        manifest = _metadata(args, policy)
    return manifest, manifest_path


def record_epoch_checkpoint(
    args: Any,
    model_state_dict: Any,
    epoch: int,
    val_metrics: Mapping[str, Any] | None,
    selection_metric: str,
    is_best: bool,
) -> Path | None:
    """Record epoch metadata and optionally persist this epoch's weights."""
    policy = get_checkpoint_policy(args)
    now_log_dir = Path(args.Logs.now_log_dir)
    checkpoint_path: Path | None = None
    if policy.save_mode == EVERY_EPOCH:
        checkpoint_path = (
            now_log_dir / "epoch_checkpoints" / f"Epoch_{epoch:04d}.pth"
        )
        _atomic_torch_save(model_state_dict, checkpoint_path)

    safe_metrics = _json_safe(val_metrics or {})
    metric_value = safe_metrics.get(policy.spe_metric)
    if metric_value is None:
        metric_value = safe_metrics.get(selection_metric)

    record = {
        "epoch": int(epoch),
        "checkpoint": (
            checkpoint_path.relative_to(now_log_dir).as_posix()
            if checkpoint_path is not None
            else None
        ),
        "val_metrics": safe_metrics,
        "selection_metric": str(selection_metric),
        "selection_metric_value": _json_safe(
            safe_metrics.get(selection_metric)
        ),
        "spe_metric": policy.spe_metric,
        "spe_metric_value": _json_safe(metric_value),
        "is_best": bool(is_best),
    }

    manifest, manifest_path = _load_manifest(args, policy)
    records_by_epoch = {
        int(item["epoch"]): item for item in manifest.get("epochs", [])
    }
    records_by_epoch[int(epoch)] = record
    manifest["epochs"] = [
        records_by_epoch[key] for key in sorted(records_by_epoch)
    ]
    if is_best:
        manifest["best_checkpoint"] = {
            "epoch": int(epoch),
            "checkpoint": f"Best_EPOCH_{epoch}.pth",
            "selection_metric": str(selection_metric),
            "selection_metric_value": _json_safe(
                safe_metrics.get(selection_metric)
            ),
        }
    _atomic_json_save(manifest, manifest_path)
    return checkpoint_path


def record_last_checkpoint(args: Any, epoch: int) -> Path:
    """Record the terminal checkpoint and mark the trajectory complete."""
    policy = get_checkpoint_policy(args)
    manifest, manifest_path = _load_manifest(args, policy)
    manifest["last_checkpoint"] = {
        "epoch": int(epoch),
        "checkpoint": f"Last_EPOCH_{epoch}.pth",
    }
    manifest["trajectory_complete"] = True
    _atomic_json_save(manifest, manifest_path)
    return manifest_path


def _stable_intervals(
    records: list[dict[str, Any]],
    threshold: float,
    min_consecutive: int,
) -> list[list[dict[str, Any]]]:
    valid = [
        record
        for record in records
        if record.get("checkpoint")
        and isinstance(record.get("spe_metric_value"), (int, float))
    ]
    valid.sort(key=lambda record: int(record["epoch"]))
    if not valid:
        return []

    runs: list[list[dict[str, Any]]] = []
    current = [valid[0]]
    for record in valid[1:]:
        previous = current[-1]
        consecutive_epoch = int(record["epoch"]) == int(previous["epoch"]) + 1
        stable_change = (
            abs(
                float(record["spe_metric_value"])
                - float(previous["spe_metric_value"])
            )
            <= threshold
        )
        if consecutive_epoch and stable_change:
            current.append(record)
        else:
            if len(current) >= min_consecutive:
                runs.append(current)
            current = [record]
    if len(current) >= min_consecutive:
        runs.append(current)
    return runs


def _evenly_spaced_records(
    records: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if len(records) <= count:
        return records
    if count == 1:
        return [records[-1]]
    indices = [
        round(index * (len(records) - 1) / (count - 1))
        for index in range(count)
    ]
    return [records[index] for index in indices]


def finalize_spe_checkpoint_selection(args: Any) -> Path | None:
    """Select and record the stable checkpoint set after training finishes."""
    policy = get_checkpoint_policy(args)
    if policy.save_mode != EVERY_EPOCH:
        return None

    manifest, _ = _load_manifest(args, policy)
    records = sorted(
        [
            record
            for record in manifest.get("epochs", [])
            if record.get("checkpoint")
        ],
        key=lambda record: int(record["epoch"]),
    )
    if not records:
        raise FileNotFoundError(
            f"No epoch checkpoints found in {args.Logs.now_log_dir}"
        )

    intervals = _stable_intervals(
        records,
        threshold=policy.stability_threshold,
        min_consecutive=policy.min_consecutive,
    )
    if intervals:
        retained = intervals[-1]
        fallback_used = False
    else:
        retained = records[-policy.min_consecutive :]
        fallback_used = True

    selected = _evenly_spaced_records(
        retained,
        count=policy.max_checkpoints,
    )
    now_log_dir = Path(args.Logs.now_log_dir)
    selection = {
        "schema_version": 1,
        "save_mode": policy.save_mode,
        "spe_metric": policy.spe_metric,
        "stability_threshold": policy.stability_threshold,
        "min_consecutive": policy.min_consecutive,
        "max_checkpoints": policy.max_checkpoints,
        "stable_interval_found": bool(intervals),
        "fallback_used": fallback_used,
        "retained_interval": {
            "start_epoch": int(retained[0]["epoch"]),
            "end_epoch": int(retained[-1]["epoch"]),
            "checkpoint_count": len(retained),
        },
        "selected_checkpoints": [
            {
                "epoch": int(record["epoch"]),
                "checkpoint": record["checkpoint"],
                "spe_metric_value": record.get("spe_metric_value"),
            }
            for record in selected
        ],
        "best_checkpoint": manifest.get("best_checkpoint"),
        "last_checkpoint": manifest.get("last_checkpoint"),
    }
    output_path = now_log_dir / "spe_checkpoint_selection.json"
    _atomic_json_save(selection, output_path)
    return output_path


def checkpoint_policy_dict(args: Any) -> dict[str, Any]:
    """Return the resolved policy for logging, tests, and CLI diagnostics."""
    return asdict(get_checkpoint_policy(args))
