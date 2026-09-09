"""For mode=internal: each held-out-site model (trained only on internal centers)
is also scored on the pristine external cohorts 301 + ynzl (never in any
internal-LOCO training). Reports per fold + mean."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts" / "ProstateDiagnosis" / "loco"))
from modules.AB_MIL.ab_mil import AB_MIL
from loco_common import CACHE, DS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = str(_REPO / "result" / "ProstateDiagnosis" / "DataAnalysis")


def ext_slides():
    rows = []
    for name, site in (("external_test_301", "301"), ("external_test_ynzl", "云南肿瘤")):
        d = pd.read_csv(f"{DS}/{name}.csv")
        d = d[d["pool"].astype(str).str.startswith("ext")].copy()
        d["stem"] = d["filename"].map(lambda x: os.path.splitext(str(x))[0])
        d["site"] = site
        d["feat"] = d["stem"].map(lambda s: f"{CACHE}/MIL外部测试/feat_0_224/pt_files/{{model}}/{s}.pt")
        rows.append(d[["stem", "label", "type", "site", "feat"]])
    return pd.concat(rows, ignore_index=True)


def metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, p = np.asarray(y), np.asarray(p); yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    return dict(n=len(y),
               auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
               auprc=float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
               sens=tp / (tp + fn) if tp + fn else float("nan"),
               spec=tn / (tn + fp) if tn + fp else float("nan"),
               acc=(tp + tn) / len(y), cm=[[tn, fp], [fn, tp]])


def main(model, in_dim):
    cmap = json.load(open(f"{DS}/DataAnalysis/AB_MIL_{model}_loco_internal/fold_center_map.json"))
    ext = ext_slides()
    ext["feat"] = ext["stem"].map(lambda s: f"{CACHE}/MIL外部测试/feat_0_224/pt_files/{model}/{s}.pt")
    feats = {}
    for _, r in ext.iterrows():
        if os.path.exists(r["feat"]):
            feats[r["stem"]] = torch.load(r["feat"], map_location="cpu", weights_only=True).float()
    ext = ext[ext["stem"].isin(feats)].reset_index(drop=True)

    root = f"{R}/AB_MIL_{model}_loco_internal"
    new_seed = sorted(glob.glob(f"{root}/AB_MIL/seed_*"), key=os.path.getmtime)
    per_fold = {}
    for k in sorted(cmap):
        kn = k.split("_")[1]
        cps = sorted(glob.glob(f"{new_seed[-1]}/fold_{kn}/Best_EPOCH_*.pth"),
                     key=lambda p: int(p.split("_")[-1].split(".")[0])) if new_seed else []
        if not cps:  # old nested layout
            seeds = sorted(glob.glob(f"{root}/{k}/*/AB_MIL/seed_*"), key=os.path.getmtime)
            cps = sorted(glob.glob(f"{seeds[-1]}/fold_1/Best_EPOCH_*.pth"),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))
        m = AB_MIL(L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=in_dim).to(DEV).eval()
        m.load_state_dict(torch.load(cps[-1], map_location=DEV, weights_only=True))
        probs = {}
        with torch.no_grad():
            for stem, f in feats.items():
                probs[stem] = torch.softmax(m(f.to(DEV).unsqueeze(0))["logits"], -1)[0, 1].item()
        ext[f"p_{k}"] = ext["stem"].map(probs)
        ho = cmap[k]["held_out"]
        per_fold[k] = {"held_out": ho}
        print(f"\n[{model}] internal fold {k} (trained w/o {ho}) -> external:")
        for site in ("301", "云南肿瘤", "ALL"):
            sub = ext if site == "ALL" else ext[ext.site == site]
            rr = metrics(sub["label"].values, sub[f"p_{k}"].values)
            per_fold[k][site] = rr
            print(f"  {site:6} n={rr['n']:3d} AUC {rr['auc']:.3f} sens {rr['sens']:.3f} "
                  f"spec {rr['spec']:.3f} cm {rr['cm']}")

    print(f"\n[{model}] internal-LOCO external mean +/- std (2 folds):")
    for site in ("301", "云南肿瘤", "ALL"):
        for mk in ("auc", "sens", "spec"):
            v = [per_fold[k][site][mk] for k in per_fold]
            print(f"  {site:6} {mk:4}: {np.mean(v):.3f} +/- {np.std(v):.3f}")
    # ensemble of the 2 internal models
    ext["p_ens"] = ext[[f"p_{k}" for k in per_fold]].mean(axis=1)
    print(f"\n[{model}] 2-model ensemble on external:")
    for site in ("301", "云南肿瘤", "ALL"):
        sub = ext if site == "ALL" else ext[ext.site == site]
        rr = metrics(sub["label"].values, sub["p_ens"].values)
        print(f"  {site:6} n={rr['n']:3d} AUC {rr['auc']:.3f} AUPRC {rr['auprc']:.3f} "
              f"sens {rr['sens']:.3f} spec {rr['spec']:.3f} cm {rr['cm']}")

    out = f"{R}/AB_MIL_{model}_loco_internal/external_eval.json"
    json.dump(per_fold, open(out, "w"), ensure_ascii=False, indent=2)
    ext.to_csv(f"{R}/AB_MIL_{model}_loco_internal/external_slide_preds.csv", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in_dim", type=int, required=True)
    a = ap.parse_args()
    main(a.model, a.in_dim)
