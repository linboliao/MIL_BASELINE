"""Stratify external-test SPE performance by malignant tumor content.

Tumor content is defined only for malignant slides.  For each tumor-content
stratum, sensitivity is calculated on the positive slides in that stratum,
whereas specificity is calculated on the same complete negative reference set.
This makes balanced accuracy well-defined and comparable across strata.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path(
    r"D:\Workspace\PythonProject\MIL_BASELINE\result\Diagnosis\SPE"
    r"\Ensemble_Weighted_Result.csv"
)
DEFAULT_AREAS = {
    "301": Path(r"E:\download\301\yolo\area.csv"),
    "SL": Path(r"E:\download\省立医院\yolo\area.csv"),
    "YNZL": Path(r"E:\download\云南省肿瘤\yolo\area.csv"),
}
DEFAULT_OUTPUT_DIR = Path(
    r"D:\Workspace\PythonProject\MIL_BASELINE\result\Diagnosis\SPE"
    r"\tumor_content"
)

# Intervals are left-closed and right-open, except the final open-ended bin.
BIN_EDGES = [-np.inf, 1.0, 5.0, 20.0, np.inf]
STRATA = [
    ("Very low (<1%)", "极低（<1%）"),
    ("Low (1–<5%)", "低（1–<5%）"),
    ("Medium (5–<20%)", "中（5–<20%）"),
    ("High (≥20%)", "高（≥20%）"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--area-301", type=Path, default=DEFAULT_AREAS["301"])
    parser.add_argument("--area-sl", type=Path, default=DEFAULT_AREAS["SL"])
    parser.add_argument("--area-ynzl", type=Path, default=DEFAULT_AREAS["YNZL"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalized_slide_id(value: object) -> str:
    """Remove punctuation differences such as the missing dot in some SL IDs."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"slide_id": str, "center": str})
    required = {"slide_id", "label", "prediction", "type", "center"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if frame["slide_id"].duplicated().any():
        raise ValueError("Prediction file contains duplicate slide_id values.")
    for column in ("label", "prediction"):
        values = set(frame[column].dropna().astype(int).unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must contain only binary 0/1 values.")
        frame[column] = frame[column].astype(int)
    frame["match_key"] = frame["slide_id"].map(normalized_slide_id)
    if frame.duplicated(["center", "match_key"]).any():
        raise ValueError("Normalized prediction IDs are not unique within center.")
    return frame


def load_areas(paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for center, path in paths.items():
        frame = pd.read_csv(path, dtype={"slide_id": str})
        if not {"slide_id", "area"}.issubset(frame.columns):
            raise ValueError(f"{path} must contain slide_id and area columns.")
        frame = frame[["slide_id", "area"]].copy()
        frame["center"] = center
        frame["match_key"] = frame["slide_id"].map(normalized_slide_id)
        frame["tumor_area_pct"] = pd.to_numeric(
            frame["area"].astype(str).str.strip().str.rstrip("%"), errors="coerce"
        )
        invalid = frame["tumor_area_pct"].isna() | ~frame["tumor_area_pct"].between(0, 100)
        if invalid.any():
            bad = frame.loc[invalid, ["slide_id", "area"]].to_dict("records")
            raise ValueError(f"Invalid area values in {path}: {bad[:5]}")
        frames.append(frame)
    areas = pd.concat(frames, ignore_index=True)
    if areas.duplicated(["center", "match_key"]).any():
        raise ValueError("Normalized area IDs are not unique within center.")
    return areas


def assign_tumor_content(
    predictions: pd.DataFrame, areas: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    area_columns = ["center", "match_key", "slide_id", "tumor_area_pct"]
    merged = predictions.merge(
        areas[area_columns],
        on=["center", "match_key"],
        how="left",
        suffixes=("", "_area"),
        validate="one_to_one",
        indicator=True,
    )

    # area.csv is specified to describe malignant slides only.
    negative_with_area = merged[merged["label"].eq(0) & merged["tumor_area_pct"].notna()]
    if not negative_with_area.empty:
        examples = negative_with_area["slide_id"].head().tolist()
        raise ValueError(f"Area records unexpectedly matched negative slides: {examples}")

    unused = areas.merge(
        predictions[["center", "match_key"]],
        on=["center", "match_key"],
        how="left",
        indicator=True,
    )
    unused = unused[unused["_merge"].eq("left_only")]
    if not unused.empty:
        examples = unused["slide_id"].head().tolist()
        raise ValueError(f"Area records did not match prediction slides: {examples}")

    positive = merged[merged["label"].eq(1)].copy()
    positive["area_missing_assumed_very_low"] = positive["tumor_area_pct"].isna()
    stratum_labels = [english for english, _ in STRATA]
    positive["stratum"] = pd.cut(
        positive["tumor_area_pct"].fillna(-np.inf),
        bins=BIN_EDGES,
        labels=stratum_labels,
        right=False,
        ordered=True,
    )
    chinese = dict(STRATA)
    positive["stratum_cn"] = positive["stratum"].astype(str).map(chinese)
    return merged, positive


def summarize(predictions: pd.DataFrame, positive: pd.DataFrame) -> pd.DataFrame:
    negative = predictions[predictions["label"].eq(0)]
    tn = int((negative["prediction"] == 0).sum())
    fp = int((negative["prediction"] == 1).sum())
    specificity = safe_divide(tn, tn + fp)

    rows = []
    for order, (stratum, stratum_cn) in enumerate(STRATA, start=1):
        subgroup = positive[positive["stratum"].astype(str).eq(stratum)]
        tp = int((subgroup["prediction"] == 1).sum())
        fn = int((subgroup["prediction"] == 0).sum())
        sensitivity = safe_divide(tp, tp + fn)
        balanced_accuracy = float(np.nanmean([sensitivity, specificity]))
        rows.append(
            {
                "order": order,
                "stratum": stratum,
                "stratum_cn": stratum_cn,
                "n_positive": len(subgroup),
                "n_negative_reference": len(negative),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "n_missing_area_assumed_very_low": int(
                    subgroup["area_missing_assumed_very_low"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    fs = 14
    plt.rcParams.update(
        {
            "font.size": fs,
            "xtick.labelsize": fs - 1,
            "ytick.labelsize": fs - 1,
            "legend.fontsize": fs - 1,
            "axes.unicode_minus": False,
        }
    )
    metrics = {
        "balanced_accuracy": {
            "name": "Bal. Acc.", "color": "#1a5276", "marker": "*", "offset": 0.00
        },
        "sensitivity": {
            "name": "Sensitivity", "color": "#a93226", "marker": "o", "offset": -0.16
        },
        "specificity": {
            "name": "Specificity", "color": "#1e8449", "marker": "D", "offset": 0.16
        },
    }

    frame = summary.sort_values("order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=300)
    for i, row in frame.iterrows():
        values = [float(row[key]) for key in metrics]
        ax.plot(
            [min(values), max(values)], [i, i], color="#b9c0c4", lw=2.2,
            alpha=0.8, solid_capstyle="round", zorder=1
        )
        for key, config in metrics.items():
            value = float(row[key])
            y = i + config["offset"]
            ax.scatter(
                value, y, s=95, marker=config["marker"], color=config["color"],
                edgecolors="white", linewidths=1.0, zorder=3
            )
            text_position = {
                "balanced_accuracy": ((9, 0), "left"),
                "sensitivity": ((0, 9), "center"),
                "specificity": ((0, -12), "center"),
            }[key]
            weight = "bold" if key == "balanced_accuracy" else "normal"
            ax.annotate(
                f"{value:.3f}", (value, y), xytext=text_position[0],
                textcoords="offset points", ha=text_position[1], va="center",
                fontsize=10, color=config["color"], fontweight=weight
            )

    all_values = frame[list(metrics)].to_numpy(float)
    x_min = max(0.0, np.floor((np.nanmin(all_values) - 0.015) / 0.02) * 0.02)
    ax.axvline(0.90, color="#999999", ls="--", lw=1, alpha=0.45)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["stratum"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_ylim(len(frame) - 0.55, -0.55)
    ax.set_xlim(x_min, 1.005)
    ax.set_xticks(np.arange(x_min, 1.001, 0.02))
    ax.set_xlabel("Performance")
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker=config["marker"], color="w",
                markerfacecolor=config["color"], markersize=9,
                label=config["name"]
            )
            for config in metrics.values()
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
        frameon=True, edgecolor="#e5e5e5"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)
    fig.tight_layout()
    fig.savefig(output_dir / "spe_tumor_content_performance.svg", bbox_inches="tight")
    fig.savefig(output_dir / "spe_tumor_content_performance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    area_paths = {
        "301": args.area_301,
        "SL": args.area_sl,
        "YNZL": args.area_ynzl,
    }
    predictions = load_predictions(args.predictions)
    areas = load_areas(area_paths)
    merged, positive = assign_tumor_content(predictions, areas)
    summary = summarize(merged, positive)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        args.output_dir / "spe_tumor_content_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positive[
        [
            "slide_id", "center", "type", "label", "prediction",
            "tumor_area_pct", "area_missing_assumed_very_low", "stratum", "stratum_cn"
        ]
    ].to_csv(
        args.output_dir / "spe_positive_tumor_content_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_summary(summary, args.output_dir)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
