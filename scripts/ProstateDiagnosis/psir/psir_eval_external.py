"""PSIR (UNI) external test: for each PSIR fold K, apply projection head K to the
external cohorts' raw UNI features, run the 5-CV AB_MIL ensemble, and score
301 + ynzl (slide-level, threshold 0.5). Report per PSIR fold and mean+/-std.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_REPO = str(Path(__file__).resolve().parents[3])  # repo root, inferred from this file's location
sys.path.insert(0, _REPO)
from modules.AB_MIL.ab_mil import AB_MIL

DS = f"{_REPO}/datasets/ProstateDiagnosis"
R = f"{_REPO}/result/ProstateDiagnosis/DataAnalysis"
PSIR = f"{DS}/psir"
_FEAT_ROOT = os.environ.get(
    "PROSTATE_FEAT_ROOT", "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")
UNI_DIR = f"{_FEAT_ROOT}/MIL外部测试/feat_0_224/pt_files/uni"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_DIM, PROJ_DIM = 1024, 256


class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, IN_DIM), nn.ReLU(inplace=True),
                                 nn.Linear(IN_DIM, PROJ_DIM))

    def forward(self, x):
        return self.net(x)


def load_cohorts():
    out = []
    for name in ("external_test_301", "external_test_ynzl"):
        df = pd.read_csv(f"{DS}/{name}.csv")
        df = df[df["pool"].astype(str).str.startswith("ext")].copy()
        df["stem"] = df["filename"].map(lambda x: os.path.splitext(str(x))[0])
        df["cohort"] = name.replace("external_test_", "")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def ab_mil():
    m = AB_MIL(L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=PROJ_DIM)
    return m.to(DEV).eval()


def cv_ckpts(K):
    paths = []
    for cv in range(1, 6):
        seeds = sorted(glob.glob(f"{R}/AB_MIL_uni_psir_fold{K}_5fold_3center/fold_{cv}/*/AB_MIL/seed_*"),
                       key=os.path.getmtime)
        best = sorted(glob.glob(f"{seeds[-1]}/fold_1/Best_EPOCH_*.pth"),
                      key=lambda p: int(p.split("_")[-1].split(".")[0]))
        paths.append(best[-1])
    return paths


def metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p); yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    ap = average_precision_score(y, p) if len(set(y)) > 1 else float("nan")
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    acc = (tp + tn) / len(y)
    return dict(n=len(y), auc=auc, auprc=ap, sens=sens, spec=spec, acc=acc,
               cm=[[tn, fp], [fn, tp]])


def main():
    coh = load_cohorts()
    print(f"external slides: {len(coh)}  ({coh.cohort.value_counts().to_dict()})")

    # load raw UNI features once
    feats = {}
    miss = []
    for stem in coh["stem"].unique():
        p = f"{UNI_DIR}/{stem}.pt"
        if not os.path.exists(p):
            miss.append(stem); continue
        feats[stem] = torch.load(p, map_location="cpu", weights_only=True).float()
    if miss:
        print(f"  missing UNI feat for {len(miss)}: {miss[:5]}")
    coh = coh[coh["stem"].isin(feats)].reset_index(drop=True)

    per_fold = {}
    all_slide_rows = []
    for K in range(1, 6):
        proj = ProjHead().to(DEV).eval()
        proj.load_state_dict(torch.load(f"{PSIR}/proj_heads/fold{K}_proj.pt",
                                        map_location=DEV, weights_only=True))
        models = []
        for cp in cv_ckpts(K):
            m = ab_mil(); m.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True)); models.append(m)

        probs = {}
        with torch.no_grad():
            for stem, f in feats.items():
                z = proj(f.to(DEV))                       # (N, 256)
                ps = []
                for m in models:
                    logit = m(z.unsqueeze(0))["logits"]   # (1, 2)
                    ps.append(torch.softmax(logit, -1)[0, 1].item())
                probs[stem] = float(np.mean(ps))

        coh[f"p_fold{K}"] = coh["stem"].map(probs)
        fold_res = {}
        for cohort in ("301", "ynzl", "ALL"):
            sub = coh if cohort == "ALL" else coh[coh.cohort == cohort]
            fold_res[cohort] = metrics(sub["label"].values, sub[f"p_fold{K}"].values)
        # 301 benign RP+TURP specificity (the magnification-story metric)
        b = coh[(coh.cohort == "301") & (coh.label == 0) & (coh.type.isin(["RP", "TURP"]))]
        yh = (b[f"p_fold{K}"].values >= 0.5).astype(int)
        fold_res["301_benign_RP_TURP_spec"] = float((yh == 0).mean()) if len(b) else float("nan")
        per_fold[K] = fold_res
        print(f"\nPSIR fold {K}:")
        for c in ("301", "ynzl", "ALL"):
            r = fold_res[c]
            print(f"  {c:5} n={r['n']:3d}  AUC {r['auc']:.3f}  AUPRC {r['auprc']:.3f}  "
                  f"sens {r['sens']:.3f}  spec {r['spec']:.3f}  cm {r['cm']}")
        print(f"  301 benign RP+TURP spec: {fold_res['301_benign_RP_TURP_spec']:.3f}")

    # aggregate mean+/-std across PSIR folds
    print("\n" + "=" * 70)
    print("MEAN +/- STD across 5 PSIR folds")
    agg = {}
    for c in ("301", "ynzl", "ALL"):
        agg[c] = {}
        for m in ("auc", "auprc", "sens", "spec", "acc"):
            v = [per_fold[K][c][m] for K in range(1, 6)]
            agg[c][m] = [float(np.mean(v)), float(np.std(v))]
            print(f"  {c:5} {m:5}: {np.mean(v):.3f} +/- {np.std(v):.3f}")
        print()
    v = [per_fold[K]["301_benign_RP_TURP_spec"] for K in range(1, 6)]
    print(f"  301 benign RP+TURP spec: {np.mean(v):.3f} +/- {np.std(v):.3f}")

    # ensemble-of-all-25: average p across the 5 PSIR-fold probs
    coh["p_ens25"] = coh[[f"p_fold{K}" for K in range(1, 6)]].mean(axis=1)
    print("\nENSEMBLE of all 25 models (mean prob):")
    for c in ("301", "ynzl", "ALL"):
        sub = coh if c == "ALL" else coh[coh.cohort == c]
        r = metrics(sub["label"].values, sub["p_ens25"].values)
        print(f"  {c:5} n={r['n']:3d}  AUC {r['auc']:.3f}  AUPRC {r['auprc']:.3f}  "
              f"sens {r['sens']:.3f}  spec {r['spec']:.3f}  acc {r['acc']:.3f}  cm {r['cm']}")

    out = f"{R}/psir_uni_external_summary.json"
    with open(out, "w") as fh:
        json.dump({"per_fold": per_fold, "agg": agg}, fh, ensure_ascii=False, indent=2)
    coh.to_csv(f"{R}/psir_uni_external_slide_preds.csv", index=False)
    print(f"\nwritten -> {out}\n         -> {R}/psir_uni_external_slide_preds.csv")


if __name__ == "__main__":
    main()
