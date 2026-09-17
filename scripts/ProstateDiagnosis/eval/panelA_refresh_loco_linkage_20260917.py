#!/usr/bin/env python3
"""Refresh Panel A stage-2 LOCO linkage using an external standardized LOCO collector JSON.

This is a derived, no-clobber analysis. It does not recompute Panel A features,
registration, or matched-region metrics. It removes legacy ``loco_*`` columns
from the parent stage-2 summary and links the current standardized LOCO matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXPECTED_MODELS = [
    "conch",
    "uni",
    "uni2",
    "virchow2",
    "h-optimus-1",
    "mstar",
    "gigapath",
    "gpfm",
]

PANEL_METRICS = {
    "matched_patch_same_over_different_ratio": "lower_is_better",
    "consistency_ratio_matched_region": "lower_is_better",
    "domain_cv_balanced_accuracy_matched_region": "lower_is_better",
    "label_case_cv_auc_matched_region": "higher_is_better",
}

LOCO_OUTCOMES = [
    "loco_overall_worst_bacc",
    "loco_internal_mean_bacc",
    "loco_type_mean_bacc",
    "loco_leave_rp_bacc",
    "loco_leave_rp_spec",
]


def safe_csv(path: Path, df: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)


def safe_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def safe_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_models(values, source: str) -> None:
    got = list(values)
    missing = sorted(set(EXPECTED_MODELS) - set(got))
    extra = sorted(set(got) - set(EXPECTED_MODELS))
    duplicates = sorted(pd.Series(got)[pd.Series(got).duplicated()].unique().tolist())
    if missing or extra or duplicates or len(got) != len(EXPECTED_MODELS):
        raise RuntimeError(
            f"{source}: model mismatch; missing={missing}, extra={extra}, "
            f"duplicates={duplicates}, n={len(got)}"
        )


def fold_metric(folds: dict, held_out: str, metric: str):
    value = folds.get(held_out, {}).get(metric)
    return np.nan if value is None else float(value)


def build_loco_matrix(rows: list[dict], std_tag: str, source_server: str) -> pd.DataFrame:
    require_models([r.get("pfm") for r in rows], "std collector JSON")
    out = []
    for r in rows:
        pfm = r["pfm"]
        folds = r.get("folds", {})
        agg = r.get("agg", {})
        n_folds = int(agg.get("n_folds_present", 0))
        if n_folds != 5:
            raise RuntimeError(f"{pfm}: expected 5 standardized folds, got {n_folds}")
        row = {
            "model": pfm,
            "loco_std_tag": std_tag,
            "loco_source_server": source_server,
            "loco_n_folds_present": n_folds,
            "loco_leave_shengli_bacc": fold_metric(folds, "省立", "bacc"),
            "loco_leave_xinchang_bacc": fold_metric(folds, "新昌", "bacc"),
            "loco_leave_cnb_bacc": fold_metric(folds, "CNB", "bacc"),
            "loco_leave_rp_bacc": fold_metric(folds, "RP", "bacc"),
            "loco_leave_turp_bacc": fold_metric(folds, "TURP", "bacc"),
            "loco_leave_rp_sens": fold_metric(folds, "RP", "sens"),
            "loco_leave_rp_spec": fold_metric(folds, "RP", "spec"),
            "loco_internal_mean_bacc": float(agg["internal_LOCO_mean_bACC"]),
            "loco_type_mean_bacc": float(agg["type_mean_bACC"]),
            "loco_worst_type_bacc": float(agg["worst_type_bACC"]),
            "loco_overall_worst_bacc": float(agg["overall_worst_case_bACC"]),
        }
        out.append(row)
    df = pd.DataFrame(out)
    return df.sort_values("loco_overall_worst_bacc", ascending=False).reset_index(drop=True)


def correlation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel_metric, direction in PANEL_METRICS.items():
        if panel_metric not in summary.columns:
            raise KeyError(f"missing panel metric: {panel_metric}")
        raw_x = summary[panel_metric].to_numpy(dtype=float)
        oriented_x = -raw_x if direction == "lower_is_better" else raw_x
        for outcome in LOCO_OUTCOMES:
            if outcome not in summary.columns:
                raise KeyError(f"missing LOCO outcome: {outcome}")
            y = summary[outcome].to_numpy(dtype=float)
            mask = np.isfinite(raw_x) & np.isfinite(y)
            raw = spearmanr(raw_x[mask], y[mask])
            oriented = spearmanr(oriented_x[mask], y[mask])
            rows.append({
                "panel_metric": panel_metric,
                "panel_direction": direction,
                "loco_outcome": outcome,
                "n_models": int(mask.sum()),
                "spearman_rho_raw": float(raw.statistic),
                "spearman_rho_robustness_oriented": float(oriented.statistic),
                "pvalue_two_sided": float(oriented.pvalue),
            })
    return pd.DataFrame(rows)


def old_vs_new_table(parent: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    common = parent[["model"]].copy()
    mappings = [
        ("loco_worst_fold_bacc", "loco_overall_worst_bacc", "overall_worst_bacc"),
        ("loco_leave_rp_spec", "loco_leave_rp_spec", "leave_rp_spec"),
        ("loco_type_mean_bacc", "loco_type_mean_bacc", "type_mean_bacc"),
    ]
    for old_col, new_col, label in mappings:
        if old_col in parent.columns:
            old = parent[["model", old_col]].rename(columns={old_col: f"old_{label}"})
            common = common.merge(old, on="model", how="left")
        new = current[["model", new_col]].rename(columns={new_col: f"std20260911_{label}"})
        common = common.merge(new, on="model", how="left")
        if f"old_{label}" in common.columns:
            common[f"delta_std_minus_old_{label}"] = (
                common[f"std20260911_{label}"] - common[f"old_{label}"]
            )
    return common


def fmt(v) -> str:
    return "n/a" if pd.isna(v) else f"{float(v):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-stage2", required=True, type=Path)
    ap.add_argument("--std-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--std-tag", default="std-20260911")
    ap.add_argument("--source-server", default="138")
    ap.add_argument("--source-result-root", required=True)
    ap.add_argument("--source-git", required=True)
    ap.add_argument("--analysis-git", required=True)
    args = ap.parse_args()

    parent = args.panel_stage2.resolve()
    std_json = args.std_json.resolve()
    out = args.out.resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if not (parent / "pfm_summary.csv").is_file():
        raise FileNotFoundError(parent / "pfm_summary.csv")
    if not std_json.is_file():
        raise FileNotFoundError(std_json)
    out.mkdir(parents=True, exist_ok=False)

    parent_df = pd.read_csv(parent / "pfm_summary.csv")
    require_models(parent_df["model"].astype(str).tolist(), "parent Panel A summary")
    with std_json.open("r", encoding="utf-8") as f:
        raw_rows = json.load(f)
    if not isinstance(raw_rows, list):
        raise TypeError("std collector JSON must be a list")

    loco = build_loco_matrix(raw_rows, args.std_tag, args.source_server)
    safe_csv(out / "loco_std_matrix.csv", loco)
    safe_json(out / "std_bench_source.json", raw_rows)

    panel_base = parent_df.loc[:, [c for c in parent_df.columns if not c.startswith("loco_")]].copy()
    summary = panel_base.merge(loco, on="model", how="inner", validate="one_to_one")
    require_models(summary["model"].tolist(), "refreshed Panel A summary")
    safe_csv(out / "pfm_summary_std20260911.csv", summary)

    corr = correlation_table(summary)
    safe_csv(out / "loco_linkage_correlations.csv", corr)

    old_new = old_vs_new_table(parent_df, loco)
    safe_csv(out / "loco_old_vs_std20260911.csv", old_new)

    ranking_cols = [
        "model",
        "consistency_ratio_matched_region",
        "domain_cv_balanced_accuracy_matched_region",
        "label_case_cv_auc_matched_region",
        "loco_internal_mean_bacc",
        "loco_type_mean_bacc",
        "loco_leave_rp_bacc",
        "loco_leave_rp_spec",
        "loco_overall_worst_bacc",
    ]
    ranking = summary[ranking_cols].copy()
    ranking["rank_loco_overall_worst"] = ranking["loco_overall_worst_bacc"].rank(method="min", ascending=False)
    ranking["rank_leave_rp_spec"] = ranking["loco_leave_rp_spec"].rank(method="min", ascending=False)
    ranking["rank_low_matched_center_signal"] = ranking["domain_cv_balanced_accuracy_matched_region"].rank(method="min", ascending=True)
    ranking = ranking.sort_values(["rank_loco_overall_worst", "rank_leave_rp_spec", "model"])
    safe_csv(out / "ranking_std20260911.csv", ranking)

    key_pairs = [
        ("consistency_ratio_matched_region", "loco_overall_worst_bacc"),
        ("domain_cv_balanced_accuracy_matched_region", "loco_overall_worst_bacc"),
        ("label_case_cv_auc_matched_region", "loco_overall_worst_bacc"),
        ("consistency_ratio_matched_region", "loco_leave_rp_spec"),
        ("domain_cv_balanced_accuracy_matched_region", "loco_leave_rp_spec"),
        ("label_case_cv_auc_matched_region", "loco_leave_rp_spec"),
    ]

    lines = [
        "# Panel A × standardized 8-PFM LOCO linkage refresh",
        "",
        f"- Parent Panel A stage2: `{parent}`",
        f"- Standardized LOCO tag: `{args.std_tag}`",
        f"- LOCO source server: `{args.source_server}`",
        f"- LOCO source root: `{args.source_result_root}`",
        f"- LOCO source Git: `{args.source_git}`",
        f"- Analysis Git (195): `{args.analysis_git}`",
        f"- Source JSON SHA256: `{sha256_file(std_json)}`",
        "- This is a derived linkage refresh only; no Panel A feature extraction, registration, or model inference was repeated.",
        "- Legacy five-site/external columns are intentionally not carried forward because they are not part of the standardized 5-fold core matrix.",
        "",
        "## Standardized LOCO matrix",
        "",
        loco[[
            "model", "loco_internal_mean_bacc", "loco_type_mean_bacc", "loco_worst_type_bacc",
            "loco_leave_rp_bacc", "loco_leave_rp_spec", "loco_overall_worst_bacc",
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-fold bACC",
        "",
        loco[[
            "model", "loco_leave_shengli_bacc", "loco_leave_xinchang_bacc",
            "loco_leave_cnb_bacc", "loco_leave_rp_bacc", "loco_leave_turp_bacc",
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Panel representation vs downstream robustness",
        "",
        "Spearman correlations are exploratory across only 8 PFMs. For Panel metrics where lower values mean better robustness, the reported robustness-oriented rho negates the Panel metric so positive rho means the two axes agree. P-values are two-sided and unadjusted for multiplicity.",
        "",
    ]
    for pm, outcome in key_pairs:
        r = corr[(corr.panel_metric == pm) & (corr.loco_outcome == outcome)].iloc[0]
        lines.append(
            f"- `{pm}` → `{outcome}`: robustness-oriented rho={r.spearman_rho_robustness_oriented:.3f}, "
            f"p={r.pvalue_two_sided:.3f}, n={int(r.n_models)}."
        )

    old_domain_rho = None
    old_cons_rho = None
    if "loco_worst_fold_bacc" in parent_df.columns:
        old_domain_rho = spearmanr(
            -parent_df["domain_cv_balanced_accuracy_matched_region"],
            parent_df["loco_worst_fold_bacc"],
        ).statistic
        old_cons_rho = spearmanr(
            -parent_df["consistency_ratio_matched_region"],
            parent_df["loco_worst_fold_bacc"],
        ).statistic
    new_domain = corr[
        (corr.panel_metric == "domain_cv_balanced_accuracy_matched_region")
        & (corr.loco_outcome == "loco_overall_worst_bacc")
    ].iloc[0]
    new_cons = corr[
        (corr.panel_metric == "consistency_ratio_matched_region")
        & (corr.loco_outcome == "loco_overall_worst_bacc")
    ].iloc[0]

    lines += [
        "",
        "## Change from legacy linkage",
        "",
    ]
    if old_domain_rho is not None:
        lines.append(
            f"- Matched-region center bACC vs worst-fold LOCO: legacy robustness-oriented rho={old_domain_rho:.3f} "
            f"→ standardized rho={new_domain.spearman_rho_robustness_oriented:.3f}."
        )
        lines.append(
            f"- Matched-region consistency ratio vs worst-fold LOCO: legacy robustness-oriented rho={old_cons_rho:.3f} "
            f"→ standardized rho={new_cons.spearman_rho_robustness_oriented:.3f}."
        )
    lines += [
        "- Therefore the earlier apparent strong association between lower residual center signal and downstream LOCO robustness is not stable after replacing the mixed/legacy LOCO values with the standardized `std-20260911` matrix.",
        "- The current result supports treating anatomy-controlled laboratory/domain sensitivity and downstream specimen/center generalization as related but distinct reliability axes; it does not establish that center separability alone predicts clinical failure.",
        "",
        "## Old vs standardized values (audit only)",
        "",
        old_new.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        "- `loco_std_matrix.csv`: latest standardized 8-PFM LOCO matrix linked to Panel A.",
        "- `pfm_summary_std20260911.csv`: parent Panel A metrics with legacy LOCO columns removed and standardized LOCO columns added.",
        "- `loco_linkage_correlations.csv`: all pre-specified Panel × LOCO Spearman correlations.",
        "- `loco_old_vs_std20260911.csv`: historical audit of shared legacy vs standardized fields.",
        "- `ranking_std20260911.csv`: descriptive multidimensional ranks; not an overall model verdict.",
        "- `std_bench_source.json`: exact normalized collector output used for linkage.",
        "- `run_meta.json`: provenance and lineage.",
    ]
    safe_text(out / "REPORT.md", "\n".join(lines))

    meta = {
        "study": "panelA_loco_linkage",
        "run_tag": out.name,
        "purpose": "formal_derived_analysis",
        "status": "completed",
        "scientific_status": "active",
        "parent_run": str(parent),
        "parent_artifact": str(parent / "pfm_summary.csv"),
        "std_tag": args.std_tag,
        "std_source_server": args.source_server,
        "std_source_result_root": args.source_result_root,
        "std_source_json": str(std_json),
        "std_source_json_sha256": sha256_file(std_json),
        "source_git_138": args.source_git,
        "analysis_git_195": args.analysis_git,
        "output_path": str(out),
        "n_models": int(len(summary)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Derived linkage refresh only; no Panel A features/registration/inference recomputed. Legacy LOCO columns removed before merge.",
    }
    safe_json(out / "run_meta.json", meta)
    print(f"DONE {out}")


if __name__ == "__main__":
    main()
