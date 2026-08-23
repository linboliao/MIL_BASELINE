"""Run a configured Stability-Prioritized Ensemble (SPE) variant.

Examples:
    python scripts/Diagnosis/spe/run.py --variant hierarchical_bacc --preflight
    python scripts/Diagnosis/spe/run.py --variant macro_f1 --devices cuda:0,cuda:1

The legacy ``--spe-config`` option remains available for custom configurations.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.Diagnosis.spe.infra import _engine as engine


@dataclass(frozen=True)
class Variant:
    """Configuration contract for one named SPE experiment."""

    config: str
    aggregation: str
    selection_objective: str | None = None


DEFAULT_VARIANT = "hierarchical_bacc"
VARIANTS: dict[str, Variant] = {
    "hierarchical_bacc": Variant(
        config="configs/Diagnosis/SPE/hierarchical_spe.yaml",
        aggregation="weighted_mean",
        selection_objective="balanced_accuracy",
    ),
    "macro_f1": Variant(
        config="configs/Diagnosis/SPE/hierarchical_spe_macro_f1.yaml",
        aggregation="weighted_mean",
        selection_objective="macro_f1",
    ),
    "hierarchical_v1": Variant(
        config="configs/Diagnosis/SPE/hierarchical_spe_v1.yaml",
        aggregation="weighted_mean",
    ),
    "best_state_anchor": Variant(
        config="configs/Diagnosis/SPE/best_state_anchor.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "best_state_anchor_v1": Variant(
        config="configs/Diagnosis/SPE/best_state_anchor_v1.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "clam_sb_robust_anchor": Variant(
        config="configs/Diagnosis/SPE/clam_sb_robust_anchor.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "trajectory_robust_anchor": Variant(
        config="configs/Diagnosis/SPE/trajectory_robust_anchor.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "gdf_stability_locked": Variant(
        config="configs/Diagnosis/SPE/gdf_stability_locked.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "cp_awa_single_anchor": Variant(
        config="configs/Diagnosis/SPE/cp_awa_single_anchor.yaml",
        aggregation="top1_anchor_fallback",
    ),
    "constrained_linear_stacking": Variant(
        config="configs/Diagnosis/SPE/constrained_linear_stacking.yaml",
        aggregation="constrained_linear_stacking",
    ),
    "ra_spe": Variant(
        config="configs/Diagnosis/SPE/ra_spe.yaml",
        aggregation="ra_spe",
    ),
    "diversity_veto": Variant(
        config="configs/Diagnosis/SPE/diversity_veto.yaml",
        aggregation="diversity_veto",
    ),
    "sensitivity_constrained": Variant(
        config="configs/Diagnosis/SPE/sensitivity_constrained.yaml",
        aggregation="sensitivity_constrained_subset",
    ),
}


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = engine.build_parser()
    parser.set_defaults(spe_config=None)
    variants = parser.add_argument_group("variant selection")
    variants.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        help=f"Named experiment variant (default: {DEFAULT_VARIANT}).",
    )
    variants.add_argument(
        "--list-variants",
        action="store_true",
        help="List variants and their configuration files, then exit.",
    )
    return parser.parse_args(arguments)


def _load_settings(config_path: Path) -> Mapping[str, Any]:
    resolved = config_path if config_path.is_absolute() else REPO_ROOT / config_path
    if not resolved.is_file():
        raise FileNotFoundError(f"SPE configuration not found: {resolved}")

    settings = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(settings, Mapping):
        raise ValueError(f"SPE configuration must be a mapping: {resolved}")
    return settings


def _section(settings: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = settings.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"SPE configuration field {name!r} must be a mapping.")
    return section


def _validate_variant(config_path: Path, variant: Variant) -> None:
    settings = _load_settings(config_path)
    aggregation = str(
        _section(settings, "aggregation").get("strategy", "weighted_mean")
    )
    if aggregation != variant.aggregation:
        raise ValueError(
            f"Variant requires aggregation.strategy={variant.aggregation!r}, "
            f"but {config_path} contains {aggregation!r}."
        )

    selection = _section(settings, "selection")
    objective = (
        str(selection.get("objective", "patient_equal_log_loss"))
        if bool(selection.get("enabled", False))
        else None
    )
    if objective != variant.selection_objective:
        raise ValueError(
            f"Variant requires selection objective "
            f"{variant.selection_objective!r}, but {config_path} contains "
            f"{objective!r}."
        )


def main(arguments: Sequence[str] | None = None) -> None:
    """Validate the selected variant, then delegate execution to the engine."""
    cli = parse_args(arguments)
    if cli.list_variants:
        for name, variant in VARIANTS.items():
            print(f"{name}: {variant.config}")
        return

    variant_name = cli.variant
    if variant_name is None and cli.spe_config is None:
        variant_name = DEFAULT_VARIANT

    if variant_name is not None:
        variant = VARIANTS[variant_name]
        cli.spe_config = cli.spe_config or Path(variant.config)
        _validate_variant(cli.spe_config, variant)

    engine.main(cli)


# Keep imports from the former monolithic run.py working during migration.
def __getattr__(name: str) -> Any:
    return getattr(engine, name)


if __name__ == "__main__":
    main()
