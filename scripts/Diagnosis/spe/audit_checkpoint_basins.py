"""Audit validation-only high-performance stable-basin checkpoint selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.Diagnosis.spe._engine import (  # noqa: E402
    _checkpoint_metric,
    selected_checkpoints_with_details,
)


DEFAULT_POLICY: dict[str, Any] = {
    "strategy": "high_performance_stable_basin",
    "metric": "bacc",
    "metric_tolerance": 0.005,
    "secondary_metric": "macro_f1",
    "secondary_metric_tolerance": 0.005,
    "min_consecutive": 5,
    "metric_range_tolerance": 0.003,
    "sensitivity_range_tolerance": 0.010,
    "specificity_range_tolerance": 0.010,
    "max_checkpoints": 5,
    "allow_fallback": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit high-performance stable-basin selection without inference."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("result/Diagnosis/MIL_Mix/HOptimus1_10x_Reinhard"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/Diagnosis/SPE/trajectory_robust_anchor.yaml"),
        help="Read checkpoint_selection from this SPE YAML; omit if absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


def latest_trajectory_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidates = []
        for run_dir in model_dir.glob("seed_42_2026-08-1*-*"):
            if all(
                (run_dir / f"fold_{fold}" / "checkpoint_manifest.json").is_file()
                for fold in range(1, 6)
            ):
                candidates.append(run_dir)
        if candidates:
            runs.append(max(candidates, key=lambda path: path.name))
    return runs


def audit(root: Path, policy: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric = str(policy.get("metric", "bacc"))
    secondary = policy.get("secondary_metric")
    for run_dir in latest_trajectory_runs(root):
        for fold in range(1, 6):
            fold_dir = run_dir / f"fold_{fold}"
            manifest = json.loads(
                (fold_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            records = [
                record
                for record in manifest.get("epochs", [])
                if record.get("checkpoint")
                and _checkpoint_metric(record, metric) is not None
            ]
            if not records:
                continue
            _, details = selected_checkpoints_with_details(fold_dir, policy)
            selected_epochs = set(details.get("selected_epochs", []))
            selected_records = [
                record for record in records if int(record["epoch"]) in selected_epochs
            ]
            best_primary = max(_checkpoint_metric(record, metric) for record in records)
            selected_primary = [
                _checkpoint_metric(record, metric) for record in selected_records
            ]
            selected_secondary = (
                []
                if not secondary
                else [
                    _checkpoint_metric(record, str(secondary))
                    for record in selected_records
                    if _checkpoint_metric(record, str(secondary)) is not None
                ]
            )
            rows.append(
                {
                    "model": run_dir.parent.name,
                    "run": run_dir.name,
                    "fold": fold,
                    "selection_tier": details["selection_tier"],
                    "fallback_used": details["fallback_used"],
                    "selected_count": len(selected_records),
                    "selected_epochs": ",".join(
                        str(epoch) for epoch in details.get("selected_epochs", [])
                    ),
                    "retained_start_epoch": details.get("retained_interval", {}).get(
                        "start_epoch"
                    ),
                    "retained_end_epoch": details.get("retained_interval", {}).get(
                        "end_epoch"
                    ),
                    "best_primary": best_primary,
                    "selected_primary_min": min(selected_primary),
                    "selected_primary_mean": sum(selected_primary) / len(selected_primary),
                    "gap_best_to_selected_min": best_primary - min(selected_primary),
                    "selected_primary_range": max(selected_primary) - min(selected_primary),
                    "selected_secondary_min": (
                        min(selected_secondary) if selected_secondary else None
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No every-epoch checkpoint trajectories found below {root}.")
    return pd.DataFrame(rows).sort_values(["model", "fold"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    policy = DEFAULT_POLICY.copy()
    if args.config.is_file():
        settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        policy.update(settings.get("checkpoint_selection", {}))
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "ensemble_analysis" / "stable_basin_selection_audit.csv"
    )
    frame = audit(root, policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} fold selections to {output}")
    print(frame.groupby("selection_tier").size().sort_values(ascending=False).to_string())
    print(
        "max(best-selected_min gap)="
        f"{frame['gap_best_to_selected_min'].max():.6f}; "
        "mean="
        f"{frame['gap_best_to_selected_min'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
