#!/usr/bin/env python3
"""Phase-1 standardized benchmark collector — multi-dimensional, NOT a single rank.

Reads, for each PFM, the 5 core held-out folds' standardized summaries
(<seed_42_TAG>/fold_<k>/preds/summary_test.json, falling back to Best_Log CSV),
and reports every robustness dimension side by side:

  IID/internal (if a standardized 5fold_3center run is present, else n/a)
  internal LOCO mean bACC        (留省立 + 留新昌)
  type mean bACC                 (留CNB + 留RP + 留TURP)
  worst-type bACC                (min of the 3 type folds)
  leave-RP bACC / leave-RP spec
  overall worst-case bACC        (min over all 5 folds)
  worst_fold_bACC                (== overall worst-case; kept for continuity, NOT the sole ranker)
  301 / ynzl external            (if preds/summary_external_*.json present)

  python std_collect.py --repo <MIL_BASELINE> --tag std-20260911 \
     --models conch uni uni2 virchow2 h-optimus-1 mstar gigapath gpfm \
     --out ~/mil_runs/std_bench/summary
"""
import argparse, glob, json, os, re, csv, sys

FOLDS = {"internal": {1: "省立", 2: "新昌"},
         "type": {1: "CNB", 2: "RP", 3: "TURP"}}


def from_summary(fd):
    p = os.path.join(fd, "preds", "summary_test.json")
    if os.path.exists(p):
        s = json.load(open(p))["slide_level"]
        return dict(bacc=s["bacc"], auc=s["auc"], auprc=s.get("auprc"),
                    acc=s["acc"], macro_f1=s["macro_f1"],
                    sens=s["sens"], spec=s["spec"])
    bl = glob.glob(os.path.join(fd, "Best_Log_*.csv"))
    if not bl:
        return None
    r = list(csv.DictReader(open(bl[0])))[-1]
    cm = list(map(int, re.findall(r"-?\d+", r.get("test_confusion_mat", ""))))
    sens = spec = None
    if len(cm) == 4:
        tn, fp, fn, tp = cm
        sens = tp / (tp + fn) if tp + fn else None
        spec = tn / (tn + fp) if tn + fp else None
    return dict(bacc=float(r["test_bacc"]), auc=float(r["test_macro_auc"]),
                auprc=None, acc=float(r["test_acc"]),
                macro_f1=float(r["test_macro_f1"]), sens=sens, spec=spec)


def ext_summary(model_root, site):
    for k in (1, 2):
        p = os.path.join(model_root, "AB_MIL")
        # any fold dir
    hits = glob.glob(os.path.join(model_root, "AB_MIL", "seed_*",
                                  "fold_*", "preds", f"summary_external_{site}.json"))
    if not hits:
        return None
    vals = [json.load(open(h))["slide_level"] for h in hits]
    # ensemble-of-folds would need probs; here just mean of per-fold metrics
    import statistics as st
    return {k: st.mean([v[k] for v in vals if v[k] is not None])
            for k in ("bacc", "auc", "sens", "spec") if any(v[k] is not None for v in vals)}


def collect(repo, tag, model):
    row = {"pfm": model, "folds": {}}
    for mode, fdict in FOLDS.items():
        base = os.path.join(repo, "result/ProstateDiagnosis/DataAnalysis",
                            f"AB_MIL_{model}_loco_{mode}", "AB_MIL", f"seed_42_{tag}")
        for k, ho in fdict.items():
            fd = os.path.join(base, f"fold_{k}")
            m = from_summary(fd) if os.path.isdir(fd) else None
            if m:
                row["folds"][ho] = m
    # IID: standardized 5fold_3center if present
    iid_base = os.path.join(repo, "result/ProstateDiagnosis/DataAnalysis",
                            f"AB_MIL_{model}_5fold_3center", "AB_MIL", f"seed_42_{tag}")
    iid = None
    if os.path.isdir(iid_base):
        bs = []
        for fd in sorted(glob.glob(os.path.join(iid_base, "fold_*"))):
            mm = from_summary(fd)
            if mm:
                bs.append(mm["bacc"])
        if bs:
            iid = sum(bs) / len(bs)
    row["iid_bacc"] = iid
    mr = os.path.join(repo, "result/ProstateDiagnosis/DataAnalysis",
                      f"AB_MIL_{model}_loco_internal")
    row["ext_301"] = ext_summary(mr, "301")
    row["ext_ynzl"] = ext_summary(mr, "云南肿瘤") or ext_summary(mr, "ynzl")
    return row


def agg(row):
    f = row["folds"]
    def b(name): return f.get(name, {}).get("bacc")
    internal = [b("省立"), b("新昌")]
    typ = [b("CNB"), b("RP"), b("TURP")]
    allf = [x for x in internal + typ if x is not None]
    out = {}
    out["internal_LOCO_mean_bACC"] = (sum(x for x in internal if x is not None) /
                                      len([x for x in internal if x is not None])
                                      if any(internal) else None)
    out["type_mean_bACC"] = (sum(x for x in typ if x is not None) /
                             len([x for x in typ if x is not None])
                             if any(typ) else None)
    out["worst_type_bACC"] = min([x for x in typ if x is not None], default=None)
    out["leave_RP_bACC"] = b("RP")
    out["leave_RP_spec"] = f.get("RP", {}).get("spec")
    out["overall_worst_case_bACC"] = min(allf, default=None)
    out["worst_fold_bACC"] = out["overall_worst_case_bACC"]
    out["n_folds_present"] = len(allf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="std_bench_summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for m in a.models:
        r = collect(a.repo, a.tag, m)
        r["agg"] = agg(r)
        rows.append(r)

    # long csv
    with open(os.path.join(a.out, "std_bench_long.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pfm", "held_out", "bacc", "auc", "auprc", "acc", "macro_f1", "sens", "spec"])
        for r in rows:
            for ho, m in r["folds"].items():
                w.writerow([r["pfm"], ho, m["bacc"], m["auc"], m["auprc"], m["acc"],
                            m["macro_f1"], m["sens"], m["spec"]])

    dims = ["iid_bacc", "internal_LOCO_mean_bACC", "type_mean_bACC",
            "worst_type_bACC", "leave_RP_bACC", "leave_RP_spec",
            "overall_worst_case_bACC"]
    L = ["# Phase-1 standardized 8-PFM LOCO benchmark\n",
         f"tag `{a.tag}` · fp32 raw · MIL_DETERMINISM=1 · Plain AB_MIL seed 42 · frozen splits\n",
         "## Per-PFM robustness dimensions (NOT a single forced ranking)\n",
         "| PFM | IID bACC | int-LOCO mean | type mean | worst-type | 留RP bACC | 留RP spec | overall worst | folds |",
         "|---|---|---|---|---|---|---|---|---|"]
    def g(v): return "n/a" if v is None else f"{v:.3f}"
    for r in rows:
        A = r["agg"]
        L.append(f"| {r['pfm']} | {g(r['iid_bacc'])} | {g(A['internal_LOCO_mean_bACC'])} "
                 f"| {g(A['type_mean_bACC'])} | {g(A['worst_type_bACC'])} | {g(A['leave_RP_bACC'])} "
                 f"| {g(A['leave_RP_spec'])} | {g(A['overall_worst_case_bACC'])} | {A['n_folds_present']}/5 |")
    L.append("\n## Per-dimension leaderboards (context, not a verdict)\n")
    for d in dims:
        vals = [(r["pfm"], r["agg"].get(d) if d != "iid_bacc" else r["iid_bacc"]) for r in rows]
        vals = [(p, v) for p, v in vals if v is not None]
        vals.sort(key=lambda x: -x[1])
        L.append(f"- **{d}**: " + " · ".join(f"{p} {v:.3f}" for p, v in vals))
    L.append("\n## External (if inferred)\n")
    for r in rows:
        for site in ("ext_301", "ext_ynzl"):
            e = r[site]
            if e:
                L.append(f"- {r['pfm']} {site}: " +
                         " ".join(f"{k} {v:.3f}" for k, v in e.items()))
    L.append("\n## Per-fold slide-level bACC\n")
    L.append("| PFM | 留省立 | 留新昌 | 留CNB | 留RP | 留TURP |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        f = r["folds"]
        L.append(f"| {r['pfm']} | " + " | ".join(
            g(f.get(h, {}).get('bacc')) for h in ("省立", "新昌", "CNB", "RP", "TURP")) + " |")

    open(os.path.join(a.out, "std_bench_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.join(a.out, "std_bench.json"), "w"),
              ensure_ascii=False, indent=1)
    print("wrote", os.path.join(a.out, "std_bench_summary.md"))


if __name__ == "__main__":
    main()
