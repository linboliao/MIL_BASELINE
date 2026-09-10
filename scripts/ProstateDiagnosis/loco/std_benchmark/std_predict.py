#!/usr/bin/env python3
"""Standardized inference + prediction dump for one trained fold dir.

Reloads Best_EPOCH_*.pth from a fold dir, runs inference on the requested
split(s), and writes slide- and patient-level prediction CSVs plus a
fold-level summary JSON — identical schema for internal held-out folds AND
for the 301 / ynzl external cohorts.

  python std_predict.py --repo <MIL_BASELINE> \
     --fold_dir result/.../AB_MIL_<m>_loco_<mode>/AB_MIL/seed_42_<ts>/fold_<k> \
     --in_dim 1536 --model gigapath --mode internal --held_out 省立 \
     --splits test val                      # internal held-out + its val
  # external:
  python std_predict.py ... --external 301 云南肿瘤

Outputs (into <fold_dir>/preds/):
  slide_preds_<split>.csv   pfm,seed,held_out,split,patient_id,slide_id,label,
                            prob_pos,pred_label,ckpt_epoch,val_macro_f1,val_loss
  patient_preds_<split>.csv patient_id,label,prob_pos_mean,pred_label,n_slides
  summary_<split>.json      auc,auprc,acc,bacc,macro_f1,sens,spec,cm,n,n_pos
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd, torch, torch.nn as nn


def load_meta(repo):
    DS = os.path.join(repo, "datasets/ProstateDiagnosis")
    frames = []
    for fn in ("dev_clean.csv", "internal_test_clean.csv",
               "external_test_301.csv", "external_test_ynzl.csv"):
        p = os.path.join(DS, fn)
        if os.path.exists(p):
            d = pd.read_csv(p, dtype={"patient_id": str})
            d["stem"] = d["filename"].map(lambda x: os.path.splitext(str(x))[0])
            frames.append(d[["stem", "patient_id", "label"]])
    m = pd.concat(frames, ignore_index=True).drop_duplicates("stem")
    return m.set_index("stem")


def metrics(y, p, thr=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, balanced_accuracy_score)
    y = np.asarray(y); p = np.asarray(p); yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    two = len(set(y.tolist())) > 1
    return dict(n=int(len(y)), n_pos=int(y.sum()),
               auc=float(roc_auc_score(y, p)) if two else None,
               auprc=float(average_precision_score(y, p)) if two else None,
               acc=float((tp + tn) / len(y)),
               bacc=float(balanced_accuracy_score(y, yh)),
               macro_f1=float(f1_score(y, yh, average="macro")),
               sens=float(tp / (tp + fn)) if tp + fn else None,
               spec=float(tn / (tn + fp)) if tn + fp else None,
               cm=[[tn, fp], [fn, tp]])


def best_ckpt(fold_dir):
    cps = glob.glob(os.path.join(fold_dir, "Best_EPOCH_*.pth"))
    if not cps:
        sys.exit(f"no Best_EPOCH_*.pth in {fold_dir}")
    cps.sort(key=lambda p: int(re.search(r"Best_EPOCH_(\d+)", p).group(1)))
    ep = int(re.search(r"Best_EPOCH_(\d+)", cps[-1]).group(1))
    return cps[-1], ep


def best_val_row(fold_dir):
    bl = glob.glob(os.path.join(fold_dir, "Best_Log_*.csv"))
    if not bl:
        return (None, None)
    r = list(csv_reader(bl[0]))[-1]
    return (float(r.get("val_macro_f1", "nan")), float(r.get("val_loss", "nan")))


def csv_reader(p):
    import csv
    return csv.DictReader(open(p))


def fold_csv_for(repo, model, mode):
    # any one of the k fold csvs has all val/test columns for this fold's k;
    # caller passes the fold dir which contains a copied prostate_loco_*_<k>fold.csv
    return None


def read_split_paths(fold_dir, split):
    """the fold dir holds a copy of prostate_loco_<m>_<mode>_<k>fold.csv"""
    cand = glob.glob(os.path.join(fold_dir, "prostate_loco_*fold.csv"))
    if not cand:
        sys.exit(f"no prostate_loco_*fold.csv in {fold_dir}")
    df = pd.read_csv(cand[0])
    col_p, col_l = f"{split}_slide_path", f"{split}_label"
    d = df[[col_p, col_l]].dropna()
    d.columns = ["path", "label"]
    d["stem"] = d["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])
    d["label"] = d["label"].astype(float).astype(int)
    return d


def resolve_feat(stem, cache_roots):
    for cr in cache_roots:
        for sub in ("MIL训练数据", "MIL测试数据", "MIL外部测试"):
            p = os.path.join(cr, sub, "feat_0_224", "pt_files", MODEL[0], f"{stem}.pt")
            if os.path.exists(p):
                return p
    return None


MODEL = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--in_dim", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--held_out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--external", nargs="*", default=[],
                    help="site names (301 云南肿瘤) — infer external cohorts too")
    ap.add_argument("--cache_roots", nargs="*",
                    default=[os.environ.get("PROSTATE_FEAT_ROOT",
                             "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")])
    a = ap.parse_args()
    MODEL[0] = a.model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, a.repo)
    from modules.AB_MIL.ab_mil import AB_MIL

    ckpt, ep = best_ckpt(a.fold_dir)
    vf1, vloss = best_val_row(a.fold_dir)
    m = AB_MIL(L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(),
               in_dim=a.in_dim).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    meta = load_meta(a.repo)
    outd = os.path.join(a.fold_dir, "preds"); os.makedirs(outd, exist_ok=True)

    jobs = []
    for sp in a.splits:
        jobs.append((sp, read_split_paths(a.fold_dir, sp)))
    for site in a.external:
        DS = os.path.join(a.repo, "datasets/ProstateDiagnosis")
        fn = "external_test_301.csv" if site == "301" else "external_test_ynzl.csv"
        d = pd.read_csv(os.path.join(DS, fn), dtype={"patient_id": str})
        d = d[d["pool"].astype(str).str.startswith("ext")].copy()
        d["stem"] = d["filename"].map(lambda x: os.path.splitext(str(x))[0])
        d = d.rename(columns={"label": "label"})[["stem", "label"]]
        d["label"] = d["label"].astype(int)
        jobs.append((f"external_{site}", d))

    for split, d in jobs:
        rows = []
        with torch.no_grad():
            for _, r in d.iterrows():
                fp = resolve_feat(r["stem"], a.cache_roots)
                if fp is None:
                    continue
                feat = torch.load(fp, map_location="cpu", weights_only=True).float()
                prob = torch.softmax(m(feat.to(dev).unsqueeze(0))["logits"], -1)[0, 1].item()
                pid = meta.loc[r["stem"], "patient_id"] if r["stem"] in meta.index else ""
                rows.append(dict(pfm=a.model, seed=a.seed, held_out=a.held_out,
                                 split=split, patient_id=pid, slide_id=r["stem"],
                                 label=int(r["label"]), prob_pos=prob,
                                 pred_label=int(prob >= 0.5), ckpt_epoch=ep,
                                 val_macro_f1=vf1, val_loss=vloss))
        sd = pd.DataFrame(rows)
        sd.to_csv(os.path.join(outd, f"slide_preds_{split}.csv"), index=False)
        pae = (sd.groupby("patient_id")
               .agg(label=("label", "max"),
                    prob_pos_mean=("prob_pos", "mean"),
                    n_slides=("slide_id", "count")).reset_index())
        pae["pred_label"] = (pae["prob_pos_mean"] >= 0.5).astype(int)
        pae.to_csv(os.path.join(outd, f"patient_preds_{split}.csv"), index=False)
        summ = {"pfm": a.model, "seed": a.seed, "mode": a.mode,
                "held_out": a.held_out, "split": split, "ckpt_epoch": ep,
                "val_macro_f1": vf1, "val_loss": vloss,
                "slide_level": metrics(sd["label"], sd["prob_pos"]),
                "patient_level": metrics(pae["label"], pae["prob_pos_mean"])}
        json.dump(summ, open(os.path.join(outd, f"summary_{split}.json"), "w"),
                  ensure_ascii=False, indent=1)
        sl = summ["slide_level"]
        print(f"{a.model:12} {split:14} ep{ep:<3} "
              f"AUC {sl['auc']:.3f} bACC {sl['bacc']:.3f} "
              f"sens {sl['sens']} spec {sl['spec']}")


if __name__ == "__main__":
    main()
