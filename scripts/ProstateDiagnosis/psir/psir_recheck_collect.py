"""Collate psir_recheck_<model>_<variant>.json from both servers into one
bare-vs-psir-vs-shuf comparison, so the "301 improvement" claim can be judged.

  python psir_recheck_collect.py --dir <dir with the *.json>

Reads {model}_{bare,psir,shuf}.json. Emits psir_recheck_matrix.md + .csv.

How to read it:
  * psir spec@0.5 >> bare spec@0.5  AND  shuf spec@0.5 ~ bare      -> effect is real, panel-driven
  * psir ~ shuf  (both >> bare)                                    -> it's the projection head, NOT panel supervision
  * psir AUC ~ bare AUC, only spec@0.5 moves                       -> mostly a threshold/prior shift
                                                                     (check: does bare spec@sens95 already close the gap?)
"""
import argparse
import csv
import glob
import json
import os

VARIANTS = ["bare", "psir", "shuf"]


def g(d, *ks, default=None):
    for k in ks:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d != {} else default


def main(dd, out):
    # filenames: psir_recheck_<model>_<variant>.json   (variant in bare/psir/shuf)
    models = sorted({os.path.basename(p)[len("psir_recheck_"):].rsplit("_", 1)[0]
                     for p in glob.glob(f"{dd}/psir_recheck_*_*.json")
                     if os.path.basename(p).rsplit("_", 1)[-1].replace(".json", "") in VARIANTS})
    if not models:
        raise SystemExit(f"no psir_recheck_*.json in {dd}")

    rows = []
    for m in models:
        cond = {}
        for v in VARIANTS:
            p = f"{dd}/psir_recheck_{m}_{v}.json"
            cond[v] = json.load(open(p)) if os.path.exists(p) else None
        for site in ("301", "ynzl"):
            r = {"model": m, "site": site}
            for v in VARIANTS:
                c = cond[v]
                if not c:
                    continue
                s = c["sites"].get(site, {})
                for thr in ("thr0.5", "thr_sens95"):
                    mm = s.get(thr, {})
                    r[f"{v}_{thr}_spec"] = round(mm.get("spec", float("nan")), 3)
                    r[f"{v}_{thr}_sens"] = round(mm.get("sens", float("nan")), 3)
                r[f"{v}_auc"] = round(s.get("thr0.5", {}).get("auc", float("nan")), 3)
            rows.append(r)

    os.makedirs(out, exist_ok=True)
    cols = ["model", "site", "bare_auc", "psir_auc", "shuf_auc",
            "bare_thr0.5_spec", "psir_thr0.5_spec", "shuf_thr0.5_spec",
            "bare_thr0.5_sens", "psir_thr0.5_sens", "shuf_thr0.5_sens",
            "bare_thr_sens95_spec", "psir_thr_sens95_spec", "shuf_thr_sens95_spec"]
    with open(f"{out}/psir_recheck_matrix.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    with open(f"{out}/psir_recheck_matrix.md", "w", encoding="utf-8") as fh:
        fh.write("# PSIR 严格复核 — bare vs psir vs shuf\n\n")
        fh.write("同一 5-fold-3-center 划分（seed 42）、同一外部集、同一 AB_MIL。\n")
        fh.write("`shuf` = 投影头配方不变、把 Panel A 的病例分组标签打乱（负控制）。\n")
        fh.write("`@s95` = 阈值在 internal_test 上标定到 sens=0.95 后的 spec（隔离\"判别力 vs 决策边界\"）。\n\n")
        fh.write("| model | site | AUC bare/psir/shuf | spec@0.5 bare/psir/shuf | spec@s95 bare/psir/shuf |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows:
            trip = lambda k: "/".join(f"{r.get(f'{v}_{k}', '–')}" for v in VARIANTS)
            fh.write(f"| {r['model']} | {r['site']} | {trip('auc')} | "
                     f"{trip('thr0.5_spec')} | {trip('thr_sens95_spec')} |\n")
        fh.write("\n## 判读\n")
        fh.write("- psir spec@0.5 明显 > bare，且 shuf ≈ bare → 提升真实、来自 panel 监督\n")
        fh.write("- psir ≈ shuf（都 > bare） → 是投影头本身（降维/非线性正则），**不是** panel 不变性\n")
        fh.write("- psir AUC ≈ bare AUC，只有 spec@0.5 动、且 bare spec@s95 已追平 → 主要是阈值/先验偏移\n")
    print(open(f"{out}/psir_recheck_matrix.md", encoding="utf-8").read())
    print(f"-> {out}/psir_recheck_matrix.{{md,csv}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.dir, a.out or a.dir)
