#!/usr/bin/env python3
"""Aggregate multi-seed LOCO hard-fold runs.

Scans result/.../AB_MIL_<model>_loco_<mode>/AB_MIL/seed_<S>_*/fold_<k>/Best_Log*.csv
for every seed dir present, maps fold index -> held-out cohort via
datasets/.../AB_MIL_<model>_loco_<mode>/fold_center_map.json, and reports per
(model, held-out) : bACC / AUC / sens / spec  mean +/- SD + bootstrap 95% CI,
plus the derived robustness aggregates. Folds with a single seed are reported
as point values (SD/CI blank). Works on partial data (only the folds that were
run) — aggregates that need absent folds are marked "n/a (missing folds)".

  python loco_multiseed_collect.py --repo <MIL_BASELINE> \
      --models gigapath h-optimus-1 mstar gpfm --out <dir>
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

MODES = ("internal", "type", "fivesite")
RP_KEYS = {"RP", "留RP", "根治", "RP(根治)"}


def parse_cm(s):
    n = list(map(int, re.findall(r"-?\d+", s or "")))
    if len(n) != 4:
        return None
    tn, fp, fn, tp = n
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def last_row(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def bootstrap_ci(xs, iters=5000, alpha=0.05, seed=0):
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    import random
    rng = random.Random(seed)
    means = []
    n = len(xs)
    for _ in range(iters):
        means.append(sum(rng.choice(xs) for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


def msd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = sum(xs) / len(xs)
    s = st.pstdev(xs) if len(xs) > 1 else float("nan")
    return (m, s, len(xs))


def collect_model(repo, model):
    out = {}
    for mode in MODES:
        ds = os.path.join(repo, "datasets/ProstateDiagnosis/DataAnalysis",
                          f"AB_MIL_{model}_loco_{mode}")
        mapf = os.path.join(ds, "fold_center_map.json")
        if not os.path.exists(mapf):
            continue
        fmap = json.load(open(mapf))
        base = os.path.join(repo, "result/ProstateDiagnosis/DataAnalysis",
                            f"AB_MIL_{model}_loco_{mode}", "AB_MIL")
        for seeddir in sorted(glob.glob(os.path.join(base, "seed_*"))):
            m = re.match(r"seed_(\d+)_", os.path.basename(seeddir))
            seed = int(m.group(1)) if m else -1
            for foldp in sorted(glob.glob(os.path.join(seeddir, "fold_*"))):
                fk = os.path.basename(foldp)
                held = fmap.get(fk, {}).get("held_out", fk)
                cands = glob.glob(os.path.join(foldp, "Best_Log*.csv"))
                if not cands:
                    continue
                r = last_row(cands[0])
                if not r:
                    continue
                sp = parse_cm(r.get("test_confusion_mat"))
                rec = out.setdefault((mode, held), {"seeds": {}})
                rec["seeds"][seed] = {
                    "bacc": float(r["test_bacc"]),
                    "auc": float(r["test_macro_auc"]),
                    "sens": sp[0] if sp else float("nan"),
                    "spec": sp[1] if sp else float("nan"),
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="multiseed_summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    long_rows = []
    md_lines = ["# Multi-seed LOCO hard-fold aggregation\n"]
    for model in a.models:
        data = collect_model(a.repo, model)
        if not data:
            md_lines.append(f"## {model}\n\n_no runs found_\n")
            continue
        md_lines.append(f"## {model}\n")
        md_lines.append("| held-out | mode | n_seed | bACC mean±SD | bACC 95%CI | "
                        "AUC mean±SD | sens mean±SD | spec mean±SD |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        per_held_bacc = {}
        for (mode, held), rec in sorted(data.items()):
            seeds = rec["seeds"]
            baccs = [v["bacc"] for v in seeds.values()]
            aucs = [v["auc"] for v in seeds.values()]
            sens = [v["sens"] for v in seeds.values()]
            spec = [v["spec"] for v in seeds.values()]
            bm, bs, bn = msd(baccs)
            lo, hi = bootstrap_ci(baccs)
            am, as_, _ = msd(aucs)
            sm, ss, _ = msd(sens)
            pm, ps, _ = msd(spec)
            per_held_bacc[(mode, held)] = bm
            md_lines.append(
                f"| {held} | {mode} | {bn} | {bm:.3f} ± {bs:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | {am:.3f} ± {as_:.3f} | "
                f"{sm:.3f} ± {ss:.3f} | {pm:.3f} ± {ps:.3f} |")
            for s, v in sorted(seeds.items()):
                long_rows.append([model, mode, held, s, v["bacc"], v["auc"],
                                  v["sens"], v["spec"]])
        # derived aggregates (only if the needed folds are present)
        type_folds = {h: b for (mo, h), b in per_held_bacc.items() if mo == "type"}
        int_folds = {h: b for (mo, h), b in per_held_bacc.items() if mo == "internal"}
        rp = next((b for (mo, h), b in per_held_bacc.items()
                   if mo == "type" and h in RP_KEYS), None)
        rp_spec = None
        for (mode, held), rec in data.items():
            if mode == "type" and held in RP_KEYS:
                rp_spec = msd([v["spec"] for v in rec["seeds"].values()])[0]
        agg = []
        agg.append(f"- leave-RP bACC: {rp:.3f}" if rp is not None else
                   "- leave-RP bACC: n/a (fold missing)")
        agg.append(f"- leave-RP specificity: {rp_spec:.3f}" if rp_spec is not None
                   else "- leave-RP specificity: n/a")
        if len(type_folds) == 3:
            agg.append(f"- worst-type bACC: {min(type_folds.values()):.3f}")
            agg.append(f"- type mean bACC: {sum(type_folds.values())/3:.3f}")
        else:
            agg.append(f"- worst-type / type-mean bACC: n/a "
                       f"(have {sorted(type_folds)} of CNB/RP/TURP)")
        allf = {**{('i', k): v for k, v in int_folds.items()},
                **{('t', k): v for k, v in type_folds.items()}}
        if int_folds and type_folds:
            agg.append(f"- LOCO mean bACC (present folds): "
                       f"{sum(allf.values())/len(allf):.3f}")
            agg.append(f"- overall worst-case bACC (present folds): "
                       f"{min(allf.values()):.3f}")
        md_lines.append("\n" + "\n".join(agg) + "\n")

    with open(os.path.join(a.out, "multiseed_summary.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(os.path.join(a.out, "multiseed_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "held_out", "seed", "bacc", "auc", "sens", "spec"])
        w.writerows(long_rows)
    print("wrote", os.path.join(a.out, "multiseed_summary.md"))
    print("wrote", os.path.join(a.out, "multiseed_long.csv"))


if __name__ == "__main__":
    main()
