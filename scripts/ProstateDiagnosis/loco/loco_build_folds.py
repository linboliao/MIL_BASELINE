"""Leave-one-X-out fold builder.

  internal : leave-one-CENTER-out over 省立, 新昌    (迈新 always in train)
  fivesite : leave-one-CENTER-out over 省立, 新昌, 301, 云南肿瘤  (迈新 always in train)
  type     : leave-one-TYPE-out over CNB, RP, TURP   (all 3 internal centers)

train/val split by PATIENT (~1/7 val, stratified by label, seed 42).
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from loco_common import DS, MODE_CFG, pooled_cohort, feat_path

SEED = 42


def main(model, mode):
    col, values, ext = MODE_CFG[mode]
    df = pooled_cohort(include_external=ext)
    df["feat"] = df.apply(lambda r: feat_path(r, model), axis=1)

    outroot = f"{DS}/DataAnalysis/AB_MIL_{model}_loco_{mode}"
    os.makedirs(outroot, exist_ok=True)
    for old in os.listdir(outroot):                       # clear stale fold CSVs
        if old.endswith("fold.csv"):
            os.remove(os.path.join(outroot, old))
    cmap = {}
    print(f"[{model} / {mode}]  cohort {len(df)}  {col}: {df[col].value_counts().to_dict()}")

    for k, v in enumerate(values, start=1):
        test = df[df[col] == v]
        trainval = df[df[col] != v].reset_index(drop=True)

        pat = (trainval.groupby("patient_id")
               .agg(label=("label", lambda x: int(x.max()))).reset_index())
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
        tr_i, va_i = next(iter(sgkf.split(pat, pat["label"], groups=pat["patient_id"])))
        va_pat = set(pat.loc[va_i, "patient_id"])
        tr = trainval[~trainval["patient_id"].isin(va_pat)]
        va = trainval[trainval["patient_id"].isin(va_pat)]

        n = max(len(tr), len(va), len(test))
        out = pd.DataFrame({
            "train_slide_path": tr["feat"].tolist() + [None] * (n - len(tr)),
            "train_label": tr["label"].tolist() + [None] * (n - len(tr)),
            "val_slide_path": va["feat"].tolist() + [None] * (n - len(va)),
            "val_label": va["label"].tolist() + [None] * (n - len(va)),
            "test_slide_path": test["feat"].tolist() + [None] * (n - len(test)),
            "test_label": test["label"].tolist() + [None] * (n - len(test)),
        })
        # all fold CSVs go in outroot/ (not outroot/fold_k/) so ONE train_mil.py
        # invocation's built-in k-fold loop consumes them -> one shared seed_/ dir.
        out.to_csv(f"{outroot}/prostate_loco_{model}_{mode}_{k}fold.csv", index=False)
        cmap[f"fold_{k}"] = {"held_out": v, "test": int(len(test)),
                             "test_pos": int(test["label"].sum()),
                             "train": int(len(tr)), "val": int(len(va))}
        print(f"  fold{k} held-out {col}={v}: train {len(tr)} | val {len(va)} | "
              f"test {len(test)} (pos {int(test['label'].sum())})")

    json.dump(cmap, open(f"{outroot}/fold_center_map.json", "w"), ensure_ascii=False, indent=2)
    print(f"  -> {outroot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=list(MODE_CFG))
    a = ap.parse_args()
    main(a.model, a.mode)
