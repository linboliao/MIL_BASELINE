"""Shared plotting utilities for Diagnosis configuration comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "result" / "Diagnosis"

COLORS = {
    "validation": "#ef3b2c",
    "oof": "#756bb1",
    "grid": "#e8e8e8",
    "text": "#2f3e46",
    "missing": "#8c8c8c",
}


def build_parser(comparison: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Plot the {comparison} Diagnosis configuration comparison using "
            "five-fold validation summaries and OOF bootstrap estimates."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Diagnosis result directory (default: result/Diagnosis).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <result-root>/config).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window after saving the figures.",
    )
    return parser


def _normalise_name(name: object) -> str:
    return str(name).strip().casefold().replace("_", "-")


def _read_validation(comparison_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not comparison_dir.is_dir():
        raise FileNotFoundError(f"Comparison directory does not exist: {comparison_dir}")

    # Direct child directories define the configurations that belong in the plot.
    for variant_dir in sorted(p for p in comparison_dir.iterdir() if p.is_dir()):
        metric_file = variant_dir / "val_merge_5_fold_metrics.json"
        row: dict[str, object] = {
            "variant": variant_dir.name,
            "key": _normalise_name(variant_dir.name),
            "validation_file": metric_file,
            "val_bacc": np.nan,
            "val_bacc_std": np.nan,
            "val_auc": np.nan,
            "val_auc_std": np.nan,
        }
        if metric_file.is_file():
            metrics = json.loads(metric_file.read_text(encoding="utf-8"))
            for source, mean_col, std_col in (
                ("bacc", "val_bacc", "val_bacc_std"),
                ("macro_auc", "val_auc", "val_auc_std"),
            ):
                if source not in metrics:
                    raise KeyError(f"{metric_file} is missing metric '{source}'")
                row[mean_col] = float(metrics[source]["mean"])
                row[std_col] = float(metrics[source]["std"])
        rows.append(row)

    if not rows:
        raise ValueError(f"No configuration directories found in {comparison_dir}")
    return pd.DataFrame(rows)


def _read_oof(statistics_dir: Path, comparison: str) -> pd.DataFrame:
    summary_file = statistics_dir / "oof_performance_summary.csv"
    if not summary_file.is_file():
        raise FileNotFoundError(f"OOF summary does not exist: {summary_file}")

    df = pd.read_csv(summary_file)
    required = {
        "comparison",
        "variant",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{summary_file} is missing columns: {sorted(missing)}")

    df = df.loc[df["comparison"].astype(str).str.casefold() == comparison.casefold()].copy()
    df["key"] = df["variant"].map(_normalise_name)
    if df["key"].duplicated().any():
        duplicates = sorted(df.loc[df["key"].duplicated(False), "variant"].astype(str).unique())
        raise ValueError(f"Duplicate OOF variants for {comparison}: {duplicates}")
    return df


def load_comparison_data(
    result_root: Path,
    comparison: str,
    display_order: Iterable[str],
) -> pd.DataFrame:
    result_root = result_root.expanduser().resolve()
    validation = _read_validation(result_root / comparison)
    oof = _read_oof(result_root / "Statistics", comparison)

    # Validation folders are the source of truth for membership. A configuration
    # folder without a completed validation JSON (currently Virchow2) is retained
    # so its available OOF estimate can still be shown.
    oof_columns = [
        "key",
        "balanced_accuracy",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
    ]
    merged = validation.merge(oof[oof_columns], on="key", how="left", validate="one_to_one")

    order_map = {_normalise_name(name): index for index, name in enumerate(display_order)}
    merged["display_order"] = merged["key"].map(order_map).fillna(len(order_map))
    return merged.sort_values(["display_order", "variant"]).reset_index(drop=True)


def _finite_bounds(values: Iterable[np.ndarray]) -> tuple[float, float]:
    finite_parts = [part[np.isfinite(part)] for part in values]
    finite = np.concatenate([part for part in finite_parts if part.size])
    if not finite.size:
        return 0.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, 0.01)
    return max(0.0, low - 0.18 * span), min(1.005, high + 0.12 * span)


def _draw_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    val_mean: str,
    val_std: str,
    oof_mean: str,
    oof_low: str,
    oof_high: str,
) -> None:
    y = np.arange(len(df), dtype=float)
    val_y = y - 0.12
    oof_y = y + 0.12

    val = df[val_mean].to_numpy(float)
    val_err = df[val_std].to_numpy(float)
    oof = df[oof_mean].to_numpy(float)
    ci_low = df[oof_low].to_numpy(float)
    ci_high = df[oof_high].to_numpy(float)

    ax.hlines(y, 0.0, 1.01, color=COLORS["grid"], lw=0.9, zorder=0)

    val_mask = np.isfinite(val) & np.isfinite(val_err)
    ax.errorbar(
        val[val_mask],
        val_y[val_mask],
        xerr=val_err[val_mask],
        fmt="*",
        color=COLORS["validation"],
        ecolor=COLORS["validation"],
        markersize=15,
        elinewidth=1.6,
        capsize=3.5,
        capthick=1.6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    oof_mask = np.isfinite(oof) & np.isfinite(ci_low) & np.isfinite(ci_high)
    oof_xerr = np.vstack((oof[oof_mask] - ci_low[oof_mask], ci_high[oof_mask] - oof[oof_mask]))
    ax.errorbar(
        oof[oof_mask],
        oof_y[oof_mask],
        xerr=oof_xerr,
        fmt="o",
        color=COLORS["oof"],
        ecolor=COLORS["oof"],
        markersize=7.5,
        elinewidth=1.6,
        capsize=3.5,
        capthick=1.6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    missing_val = ~val_mask
    missing_oof = ~oof_mask
    x_low, x_high = _finite_bounds(
        [val - np.nan_to_num(val_err), val + np.nan_to_num(val_err), ci_low, ci_high]
    )
    ax.set_xlim(x_low, x_high)
    missing_x = x_low + 0.012 * (x_high - x_low)
    for yy in val_y[missing_val]:
        ax.text(missing_x, yy, "Validation unavailable", va="center", ha="left", fontsize=8.5, color=COLORS["missing"])
    for yy in oof_y[missing_oof]:
        ax.text(missing_x, yy, "OOF unavailable", va="center", ha="left", fontsize=8.5, color=COLORS["missing"])

    ax.set_title(title, fontsize=14, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlabel("Score", fontsize=12, fontweight="bold", color=COLORS["text"])
    ax.set_ylim(len(df) - 0.55, -0.55)
    ax.grid(axis="x", linestyle="--", color=COLORS["grid"], alpha=0.75, zorder=0)
    ax.tick_params(axis="both", labelsize=10.5, colors=COLORS["text"])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")


def plot_comparison(
    *,
    comparison: str,
    display_order: Iterable[str],
    result_root: Path,
    output_dir: Path | None,
    output_stem: str,
    show: bool = False,
) -> tuple[Path, Path]:
    df = load_comparison_data(result_root, comparison, display_order)

    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans", "Arial", "Arial Unicode MS", "SimHei", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    height = max(4.7, 0.62 * len(df) + 1.9)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, height), sharey=True, facecolor="white")
    _draw_metric_panel(
        axes[0],
        df,
        title="Balanced Accuracy",
        val_mean="val_bacc",
        val_std="val_bacc_std",
        oof_mean="balanced_accuracy",
        oof_low="balanced_accuracy_ci_low",
        oof_high="balanced_accuracy_ci_high",
    )
    _draw_metric_panel(
        axes[1],
        df,
        title="ROC AUC",
        val_mean="val_auc",
        val_std="val_auc_std",
        oof_mean="roc_auc",
        oof_low="roc_auc_ci_low",
        oof_high="roc_auc_ci_high",
    )

    y = np.arange(len(df))
    axes[0].set_yticks(y, df["variant"], fontsize=12, fontweight="bold", color=COLORS["text"])
    axes[1].tick_params(axis="y", left=False)

    legend = [
        Line2D(
            [], [], marker="*", color=COLORS["validation"], linestyle="None", markersize=13,
            markeredgecolor="white", label="Five-fold validation (mean ± SD)",
        ),
        Line2D(
            [], [], marker="o", color=COLORS["oof"], linestyle="None", markersize=8,
            markeredgecolor="white", label="OOF (estimate + patient-bootstrap 95% CI)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#e0e0e0",
        fontsize=11,
    )
    fig.suptitle(f"Diagnosis — {comparison} configurations", y=1.09, fontsize=17, fontweight="bold", color=COLORS["text"])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=2.2)

    result_root = result_root.expanduser().resolve()
    target_dir = (output_dir or (result_root / "config")).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    svg_path = target_dir / f"{output_stem}.svg"
    png_path = target_dir / f"{output_stem}.png"
    fig.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return svg_path, png_path
