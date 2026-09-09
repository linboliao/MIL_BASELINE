"""Summarize held-out-site test metrics for one encoder + mode."""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
R = str(_REPO / "result" / "ProstateDiagnosis" / "DataAnalysis")
DSR = str(_REPO / "datasets" / "ProstateDiagnosis" / "DataAnalysis")


def cm(s):
    n = [int(x) for x in re.findall(r"-?\d+", s)]
    return n[0], n[1], n[2], n[3]


def main(model, mode):
    cmap = json.load(open(f"{DSR}/AB_MIL_{model}_loco_{mode}/fold_center_map.json"))
    print(f"\n===== LOCO {model} / {mode} =====")
    hdr = f"{'held-out':>8} | {'n(pos)':>10} | {'AUC':>6} | {'AUPRC':>6} | {'sens':>6} | {'spec':>6} | {'bACC':>6} | cm"
    print(hdr); print("-" * len(hdr))
    root = f"{R}/AB_MIL_{model}_loco_{mode}"
    # new layout: <root>/AB_MIL/seed_*/fold_<k>/Best*.csv  (one shared seed dir)
    # old layout: <root>/fold_<k>/*/AB_MIL/seed_*/fold_1/Best*.csv
    seed_dirs = sorted(glob.glob(f"{root}/AB_MIL/seed_*"), key=os.path.getmtime)
    rows = []
    for k in sorted(cmap):
        kn = k.split("_")[1]  # "fold_3" -> "3"
        logs = glob.glob(f"{seed_dirs[-1]}/fold_{kn}/Best*.csv") if seed_dirs else []
        if not logs:  # fall back to the old nested layout
            logs = glob.glob(f"{root}/{k}/*/AB_MIL/seed_*/fold_1/Best_Log_*.csv")
        c = cmap[k]
        if not logs:
            print(f"{c['held_out']:>8} | {c['test']}({c['test_pos']}) | (no result)")
            continue
        rr = list(csv.DictReader(open(max(logs, key=os.path.getmtime))))[-1]
        tn, fp, fn, tp = cm(rr["test_confusion_mat"])
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        auc = float(rr["test_macro_auc"]); bacc = float(rr["test_bacc"])
        try:
            auprc = float(rr.get("test_macro_auprc") or rr.get("test_weighted_auprc") or "nan")
        except Exception:
            auprc = float("nan")
        rows.append((auc, sens, spec, bacc))
        print(f"{c['held_out']:>8} | {c['test']:>5}({c['test_pos']:>3}) | {auc:>6.3f} | "
              f"{auprc:>6.3f} | {sens:>6.3f} | {spec:>6.3f} | {bacc:>6.3f} | [[{tn},{fp}],[{fn},{tp}]]")
    if len(rows) > 1:
        print("-" * len(hdr))
        f = lambda i: (st.mean(x[i] for x in rows), st.pstdev(x[i] for x in rows))
        a, s, sp, b = f(0), f(1), f(2), f(3)
        print(f"{'MEAN':>8} |      -     | {a[0]:>6.3f} |      - | {s[0]:>6.3f} | {sp[0]:>6.3f} | {b[0]:>6.3f}")
        print(f"{'STD':>8} |      -     | {a[1]:>6.3f} |      - | {s[1]:>6.3f} | {sp[1]:>6.3f} | {b[1]:>6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["internal","fivesite","type"])
    a = ap.parse_args()
    main(a.model, a.mode)
