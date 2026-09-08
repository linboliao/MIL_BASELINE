"""Aggregate the per-encoder LOCO summaries into one 8-PFM robustness matrix.

Reads the plain-text summaries produced by loco_run.sh (loco_collect.py output):
    <dir>/<model>_internal_summary.txt
    <dir>/<model>_type_summary.txt
    <dir>/<model>_fivesite_summary.txt
    <dir>/<model>_internal_external.txt        (optional, "2-model ensemble on external")
Point --dir at one directory holding the summaries from BOTH servers (just copy
138's *_summary.txt / *_internal_external.txt next to 195's).

Emits:
    robustness_matrix_long.csv   one row per (model, mode, held_out)
    robustness_matrix.md         wide table + a worst-case robustness ranking

The single robustness number = min bACC across every held-out fold of internal+type
(worst-case generalization). We also surface 留RP specificity (the cleanest, largest
shift) and the internal->pristine-external ensemble spec on 301.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

ROW = re.compile(
    r"^\s*(?P<ho>\S+)\s*\|\s*(?P<n>\d+)\(\s*(?P<pos>\d+)\)\s*\|"
    r"\s*(?P<auc>[-\d.]+|nan)\s*\|\s*(?P<auprc>[-\d.]+|nan)\s*\|"
    r"\s*(?P<sens>[-\d.]+|nan)\s*\|\s*(?P<spec>[-\d.]+|nan)\s*\|"
    r"\s*(?P<bacc>[-\d.]+|nan)\s*\|\s*(?P<cm>\[\[.*\]\])\s*$")
EXT = re.compile(
    r"^\s*(?P<site>301|云南肿瘤|ynzl|ALL)\s+n=(?P<n>\d+)\s+AUC\s+(?P<auc>[\d.]+)"
    r"(?:\s+AUPRC\s+(?P<auprc>[\d.]+))?\s+sens\s+(?P<sens>[\d.]+)\s+spec\s+(?P<spec>[\d.]+)")

MODES = ["internal", "type", "fivesite"]
PFM_ORDER = ["conch", "uni", "uni2", "virchow2", "h-optimus-1", "gigapath", "gpfm", "mstar"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_summary(path):
    out = []
    for line in open(path, encoding="utf-8"):
        m = ROW.match(line)
        if m:
            d = m.groupdict()
            out.append(dict(held_out=d["ho"], n=int(d["n"]), pos=int(d["pos"]),
                            auc=f(d["auc"]), auprc=f(d["auprc"]), sens=f(d["sens"]),
                            spec=f(d["spec"]), bacc=f(d["bacc"]), cm=d["cm"]))
    return out


def parse_external(path):
    """Return the '2-model ensemble on external' block: {site: {...}}."""
    res, grab = {}, False
    for line in open(path, encoding="utf-8"):
        if "ensemble on external" in line:
            grab = True
            continue
        if grab:
            m = EXT.match(line)
            if m:
                d = m.groupdict()
                site = {"云南肿瘤": "ynzl"}.get(d["site"], d["site"])
                res[site] = dict(n=int(d["n"]), auc=f(d["auc"]), sens=f(d["sens"]), spec=f(d["spec"]))
            elif line.strip() and not line.startswith(" "):
                break
    return res


def main(d, outdir):
    models = sorted(
        {os.path.basename(p).rsplit("_", 2)[0]
         for p in glob.glob(f"{d}/*_summary.txt")},
        key=lambda m: PFM_ORDER.index(m) if m in PFM_ORDER else 99)
    if not models:
        raise SystemExit(f"no *_summary.txt under {d}")

    long_rows, wide = [], {}
    for model in models:
        w = {"model": model}
        allbacc = []
        for mode in MODES:
            sp = f"{d}/{model}_{mode}_summary.txt"
            if not os.path.exists(sp):
                continue
            rows = parse_summary(sp)
            for r in rows:
                long_rows.append(dict(model=model, mode=mode, **r))
            per = {r["held_out"]: r for r in rows}
            for ho, r in per.items():
                w[f"{mode}:{ho}:auc"] = r["auc"]
                w[f"{mode}:{ho}:spec"] = r["spec"]
                w[f"{mode}:{ho}:bacc"] = r["bacc"]
            if mode in ("internal", "type"):
                allbacc += [r["bacc"] for r in rows]
            if rows:
                w[f"{mode}:mean_auc"] = st.mean(r["auc"] for r in rows if r["auc"] == r["auc"])
                w[f"{mode}:mean_bacc"] = st.mean(r["bacc"] for r in rows if r["bacc"] == r["bacc"])
        ep = f"{d}/{model}_internal_external.txt"
        if os.path.exists(ep):
            for site, v in parse_external(ep).items():
                w[f"ext_ens:{site}:auc"] = v["auc"]
                w[f"ext_ens:{site}:spec"] = v["spec"]
                w[f"ext_ens:{site}:sens"] = v["sens"]
        w["worst_fold_bacc"] = min(allbacc) if allbacc else float("nan")
        w["RP_spec"] = w.get("type:RP:spec", float("nan"))
        wide[model] = w

    os.makedirs(outdir, exist_ok=True)
    lp = f"{outdir}/robustness_matrix_long.csv"
    keys = ["model", "mode", "held_out", "n", "pos", "auc", "auprc", "sens", "spec", "bacc", "cm"]
    with open(lp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in long_rows:
            wr.writerow({k: r.get(k, "") for k in keys})

    ranked = sorted(wide.values(),
                    key=lambda w: (-(w["worst_fold_bacc"] if w["worst_fold_bacc"] == w["worst_fold_bacc"] else -9),
                                   -(w["RP_spec"] if w["RP_spec"] == w["RP_spec"] else -9)))
    mp = f"{outdir}/robustness_matrix.md"
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("# 8-PFM domain-shift robustness matrix\n\n")
        fh.write("Plain AB_MIL, seed 42, StratifiedGroupKFold(7) 1/7 patient val, clean cohort.\n")
        fh.write("`worst_fold_bACC` = min bACC over all internal+type held-out folds (worst-case generalization).\n\n")

        fh.write("## 排名（按 worst_fold_bACC，再按 留RP spec）\n\n")
        fh.write("| # | PFM | worst_fold_bACC | 留RP spec | type mean bACC | internal mean AUC | fivesite mean bACC | ext-ens 301 spec |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for i, w in enumerate(ranked, 1):
            g = lambda k: (f"{w[k]:.3f}" if k in w and w[k] == w[k] else "–")
            fh.write(f"| {i} | {w['model']} | {g('worst_fold_bacc')} | {g('RP_spec')} | "
                     f"{g('type:mean_bacc')} | {g('internal:mean_auc')} | {g('fivesite:mean_bacc')} | "
                     f"{g('ext_ens:301:spec')} |\n")

        for mode, cols in (
            ("type", ["CNB", "RP", "TURP"]),
            ("internal", ["省立", "新昌"]),
            ("fivesite", ["省立", "新昌", "301", "云南肿瘤"]),
        ):
            fh.write(f"\n## 留一 {mode}  (AUC / spec)\n\n")
            fh.write("| PFM | " + " | ".join(cols) + " | mean AUC | mean bACC |\n")
            fh.write("|" + "---|" * (len(cols) + 3) + "\n")
            for w in ranked:
                cells = []
                for c in cols:
                    a, s = w.get(f"{mode}:{c}:auc"), w.get(f"{mode}:{c}:spec")
                    cells.append(f"{a:.3f} / {s:.3f}" if a == a else "–")
                ma, mb = w.get(f"{mode}:mean_auc"), w.get(f"{mode}:mean_bacc")
                cells += [f"{ma:.3f}" if ma == ma else "–", f"{mb:.3f}" if mb == mb else "–"]
                fh.write(f"| {w['model']} | " + " | ".join(cells) + " |\n")

        fh.write("\n## 内部 LOCO 模型 → 纯净外部集（301 + ynzl，2 模型 ensemble）\n\n")
        fh.write("| PFM | 301 AUC / sens / spec | ynzl AUC / sens / spec | ALL AUC / sens / spec |\n|---|---|---|---|\n")
        for w in ranked:
            def trip(site):
                a = w.get(f"ext_ens:{site}:auc"); s = w.get(f"ext_ens:{site}:sens"); p = w.get(f"ext_ens:{site}:spec")
                return f"{a:.3f} / {s:.3f} / {p:.3f}" if a == a else "–"
            fh.write(f"| {w['model']} | {trip('301')} | {trip('ynzl')} | {trip('ALL')} |\n")

    print(f"wrote {lp}\n      {mp}")
    print(open(mp, encoding="utf-8").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding *_summary.txt + *_internal_external.txt from both servers")
    ap.add_argument("--out", default=None, help="output dir (default: --dir)")
    a = ap.parse_args()
    main(a.dir, a.out or a.dir)
