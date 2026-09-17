#!/usr/bin/env python3
"""Source-locked threshold transport and calibration analysis for standardized LOCO.

Primary rules:
- standardized parent runs are seed_42_<tag> and are never modified;
- validation predictions are the ONLY source for threshold selection;
- held-out target labels are used only for evaluation, never threshold fitting;
- missing standardized prediction dumps can be re-inferred into this derived
  run's supplemental_preds/ subtree without writing into parent fold dirs;
- slide-level and patient-level analyses are both produced.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["conch", "uni", "uni2", "virchow2", "h-optimus-1", "mstar", "gigapath", "gpfm"]
HELD = {"internal": {1: "省立", 2: "新昌"}, "type": {1: "CNB", 2: "RP", 3: "TURP"}}
TARGET_SENS = (0.95, 0.98)
EPS = 1e-7


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_json(path: Path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def safe_csv(path: Path, df: pd.DataFrame):
    with path.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)


def safe_text(path: Path, text: str):
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git_info(repo: Path):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    return commit, dirty


def canonical_split(repo: Path, model: str, mode: str, fold: int):
    d = repo / "datasets/ProstateDiagnosis/DataAnalysis" / f"AB_MIL_{model}_loco_{mode}"
    p = d / f"prostate_loco_{model}_{mode}_{fold}fold.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def fold_dir(repo: Path, tag: str, model: str, mode: str, fold: int):
    return repo / "result/ProstateDiagnosis/DataAnalysis" / f"AB_MIL_{model}_loco_{mode}" / "AB_MIL" / f"seed_42_{tag}" / f"fold_{fold}"


def split_frame(csv_path: Path, split: str):
    df = pd.read_csv(csv_path)
    pcol, lcol = f"{split}_slide_path", f"{split}_label"
    d = df[[pcol, lcol]].dropna().copy()
    d.columns = ["path", "label"]
    d["slide_id"] = d["path"].map(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
    d["label"] = d["label"].astype(float).astype(int)
    return d[["slide_id", "label"]]


def load_meta(repo: Path):
    ds = repo / "datasets/ProstateDiagnosis"
    frames = []
    for fn in ("dev_clean.csv", "internal_test_clean.csv"):
        d = pd.read_csv(ds / fn, dtype={"patient_id": str, "slide_id": str})
        d["stem"] = d["filename"].map(lambda x: os.path.splitext(str(x))[0])
        frames.append(d[["stem", "patient_id", "label", "center", "type"]])
    m = pd.concat(frames, ignore_index=True)
    if m["stem"].duplicated().any():
        dup = m.loc[m["stem"].duplicated(keep=False), "stem"].unique().tolist()[:10]
        raise RuntimeError(f"duplicate metadata stems: {dup}")
    return m.set_index("stem")


def validate_group_isolation(meta: pd.DataFrame, d: pd.DataFrame, mode: str, held_out: str, split: str):
    vals = []
    missing = []
    col = "center" if mode == "internal" else "type"
    for stem in d["slide_id"]:
        if stem not in meta.index:
            missing.append(stem)
        else:
            vals.append(str(meta.loc[stem, col]))
    if missing:
        raise RuntimeError(f"metadata missing {len(missing)} stems, examples={missing[:5]}")
    vals = set(vals)
    if split == "test":
        if vals != {str(held_out)}:
            raise RuntimeError(f"{mode}/{held_out} test group mismatch: {sorted(vals)}")
    else:
        if str(held_out) in vals:
            raise RuntimeError(f"{mode}/{held_out} val contains held-out group")


def read_prediction(path: Path):
    if not path.is_file() or path.stat().st_size < 5:
        return None
    try:
        df = pd.read_csv(path, dtype={"patient_id": str, "slide_id": str})
    except pd.errors.EmptyDataError:
        return None
    return df


def validate_prediction(df: pd.DataFrame, expected: pd.DataFrame, label: str):
    req = {"slide_id", "label", "prob_pos"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"{label}: missing prediction columns {sorted(req-set(df.columns))}")
    got = Counter((str(r.slide_id), int(r.label)) for r in df.itertuples())
    exp = Counter((str(r.slide_id), int(r.label)) for r in expected.itertuples())
    if got != exp:
        raise RuntimeError(f"{label}: prediction membership/labels differ from canonical split")
    p = pd.to_numeric(df["prob_pos"], errors="coerce").to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError(f"{label}: invalid probabilities")


def best_checkpoint(fd: Path):
    man = fd / "checkpoint_manifest.json"
    if man.is_file():
        j = json.load(man.open())
        bp = j.get("best_checkpoint", {}).get("checkpoint")
        if bp and (fd / bp).is_file():
            return fd / bp, int(j["best_checkpoint"]["epoch"])
    cps = list(fd.glob("Best_EPOCH_*.pth"))
    if len(cps) != 1:
        raise RuntimeError(f"expected exactly one best checkpoint in {fd}, got {len(cps)}")
    ep = int(re.search(r"Best_EPOCH_(\d+)", cps[0].name).group(1))
    return cps[0], ep


def model_config(fd: Path):
    yams = [p for p in fd.glob("*.yaml") if "preload" not in p.name]
    if not yams:
        yams = list(fd.glob("*.yaml"))
    if not yams:
        raise FileNotFoundError(f"no yaml in {fd}")
    y = yaml.safe_load(yams[0].open())
    m = y["Model"]
    return dict(in_dim=int(m["in_dim"]), L=int(m.get("L", 512)), D=int(m.get("D", 128)), dropout=float(m.get("dropout", 0.1)))


def feature_index(feature_root: Path, model: str):
    idx = {}
    for sub in ("MIL训练数据", "MIL测试数据", "MIL外部测试"):
        d = feature_root / sub / "feat_0_224/pt_files" / model
        if not d.is_dir():
            continue
        for p in d.glob("*.pt"):
            idx.setdefault(p.stem, p)
    return idx


def infer_split(repo: Path, fd: Path, model: str, held_out: str, split: str, expected: pd.DataFrame,
                meta: pd.DataFrame, feature_root: Path, out_path: Path, device: torch.device,
                preload_chunk_gb: float, preload_workers: int):
    """Infer a missing split using bounded chunked WSI_Dataset RAM preload.

    A single Virchow2 held-out split can exceed 200 GiB, so preloading the whole
    split is unsafe even on a 251-GiB host.  We therefore partition the split by
    feature-file bytes, fully preload each bounded chunk with the project's
    WSI_Dataset.parallel_preload(), infer it from RAM, release it, then continue.
    """
    sys_path = str(repo)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from modules.AB_MIL.ab_mil import AB_MIL
    from utils.wsi_utils import WSI_Dataset

    cfg = model_config(fd)
    ckpt, ep = best_checkpoint(fd)
    net = AB_MIL(L=cfg["L"], D=cfg["D"], num_classes=2, dropout=cfg["dropout"],
                 act=nn.ReLU(), in_dim=cfg["in_dim"]).to(device).eval()
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    findex = feature_index(feature_root, model)
    missing = [s for s in expected["slide_id"] if s not in findex]
    if missing:
        raise RuntimeError(f"{model}/{held_out}/{split}: missing {len(missing)} features, examples={missing[:5]}")

    limit = int(preload_chunk_gb * (1024 ** 3))
    if limit <= 0:
        raise ValueError("preload_chunk_gb must be > 0")
    items = []
    for rec in expected.itertuples(index=False):
        fp = findex[rec.slide_id]
        items.append((rec, fp, int(fp.stat().st_size)))

    chunks, cur, cur_bytes = [], [], 0
    for item in items:
        sz = item[2]
        if cur and cur_bytes + sz > limit:
            chunks.append(cur); cur=[]; cur_bytes=0
        cur.append(item); cur_bytes += sz
    if cur:
        chunks.append(cur)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_path.parent / "preload_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[preload-plan] {model} {held_out} {split}: n={len(items)} "
          f"file_bytes={sum(x[2] for x in items)/1024**3:.1f}GiB chunks={len(chunks)} "
          f"target<={preload_chunk_gb:.1f}GiB workers={preload_workers}", flush=True)

    with torch.no_grad():
        for ci, chunk in enumerate(chunks, 1):
            cbytes = sum(x[2] for x in chunk)
            csvp = manifest_dir / f"{split}_chunk_{ci:03d}.csv"
            cdf = pd.DataFrame({
                f"{split}_slide_path": [str(x[1]) for x in chunk],
                f"{split}_label": [int(x[0].label) for x in chunk],
            })
            safe_csv(csvp, cdf)
            # Allow tensor bytes to exceed serialized file bytes modestly, while
            # keeping the chunk itself far below host available RAM.
            mem_cap = max(preload_chunk_gb * 1.35, preload_chunk_gb + 8.0)
            mem_map = {"train": mem_cap, "val": mem_cap, "test": mem_cap}
            ds = WSI_Dataset(str(csvp), split, preload=False, mem_map=mem_map)
            ds.parallel_preload(max(1, int(preload_workers)))
            loaded = sum(x is not None for x in ds.preloaded_data)
            if loaded != len(ds):
                raise RuntimeError(f"{model}/{held_out}/{split} chunk{ci}: preload incomplete {loaded}/{len(ds)}")
            print(f"[preloaded] {model} {held_out} {split} chunk {ci}/{len(chunks)}: "
                  f"n={len(ds)} files={cbytes/1024**3:.1f}GiB tensor={ds.used_memory/1024**3:.1f}GiB", flush=True)
            for j in range(len(ds)):
                feat, lab, slide_id = ds[j]
                slide_id = str(slide_id)
                prob = torch.softmax(net(feat.float().to(device).unsqueeze(0))["logits"], -1)[0, 1].item()
                if slide_id not in meta.index:
                    raise RuntimeError(f"metadata missing {slide_id}")
                rows.append({"pfm": model, "seed": 42, "held_out": held_out, "split": split,
                             "patient_id": str(meta.loc[slide_id, "patient_id"]), "slide_id": slide_id,
                             "label": int(lab.item()), "prob_pos": float(prob), "pred_label": int(prob >= 0.5),
                             "ckpt_epoch": ep, "checkpoint": str(ckpt),
                             "inference_io": "WSI_Dataset_chunked_preload"})
            del ds
            gc.collect()
            print(f"[infer] {model} {held_out} {split}: chunk {ci}/{len(chunks)} complete; "
                  f"rows={len(rows)}/{len(items)}", flush=True)

    df = pd.DataFrame(rows)
    validate_prediction(df, expected, f"inferred {model}/{held_out}/{split}")
    safe_csv(out_path, df)
    return df

def patient_aggregate(slide: pd.DataFrame, meta: pd.DataFrame):
    d = slide[["slide_id", "label", "prob_pos"]].copy()
    d["patient_id"] = d["slide_id"].map(lambda s: str(meta.loc[str(s), "patient_id"]))
    chk = d.groupby("patient_id")["label"].nunique()
    if (chk > 1).any():
        raise RuntimeError("patient has conflicting labels within split")
    return (d.groupby("patient_id", as_index=False)
            .agg(label=("label", "first"), prob_pos=("prob_pos", "mean"), n_slides=("slide_id", "count")))


def clip_prob(p):
    return np.clip(np.asarray(p, float), EPS, 1-EPS)


def logit(p):
    p = clip_prob(p)
    return np.log(p/(1-p))


def ece_equal_width(y, p, bins=10):
    y=np.asarray(y,int); p=np.asarray(p,float)
    edges=np.linspace(0,1,bins+1)
    total=len(y); out=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        mask=(p>=lo)&(p<hi if i<bins-1 else p<=hi)
        if mask.any():
            out += mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(out)


def calibration_intercept_slope(y, p):
    y=np.asarray(y,float); x=logit(p)
    if len(np.unique(y))<2:
        return np.nan,np.nan
    def obj(theta):
        a,b=theta
        z=np.clip(a+b*x,-40,40)
        # stable logistic NLL + negligible ridge for separated fits
        loss=np.sum(np.logaddexp(0,z)-y*z) + 1e-8*(a*a+b*b)
        return float(loss)
    res=minimize(obj, np.array([0.0,1.0]), method="L-BFGS-B", bounds=[(-20,20),(-10,10)])
    if not res.success:
        return np.nan,np.nan
    return float(res.x[0]),float(res.x[1])


def threshold_for_sensitivity(y, p, target):
    y=np.asarray(y,int); p=np.asarray(p,float)
    pos=np.sort(p[y==1])[::-1]
    if len(pos)==0:
        return np.nan
    k=max(1,int(math.ceil(target*len(pos))))
    return float(pos[k-1])


def decision_metrics(y, p, thr):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=thr).astype(int)
    tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum())
    fn=int(((pred==0)&(y==1)).sum()); tp=int(((pred==1)&(y==1)).sum())
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    ppv=tp/(tp+fp) if tp+fp else np.nan; npv=tn/(tn+fn) if tn+fn else np.nan
    return dict(threshold=float(thr), sensitivity=float(sens), specificity=float(spec),
                balanced_accuracy=float((sens+spec)/2), ppv=float(ppv), npv=float(npv),
                predicted_positive_fraction=float(pred.mean()), negative_triage_fraction=float((pred==0).mean()),
                tn=tn, fp=fp, fn=fn, tp=tp)


def probability_metrics(y, p):
    y=np.asarray(y,int); p=np.asarray(p,float); pc=clip_prob(p)
    two=len(np.unique(y))>1
    ci,cs=calibration_intercept_slope(y,p)
    out=dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()),
             auc=float(roc_auc_score(y,p)) if two else np.nan,
             auprc=float(average_precision_score(y,p)) if two else np.nan,
             brier=float(np.mean((p-y)**2)),
             nll=float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))),
             ece10=float(ece_equal_width(y,p,10)),
             mean_prob=float(p.mean()), calibration_intercept=ci, calibration_slope=cs)
    for cls,name in [(0,"neg"),(1,"pos")]:
        q=p[y==cls]
        z=logit(q) if len(q) else np.array([])
        out[f"{name}_prob_mean"]=float(q.mean()) if len(q) else np.nan
        out[f"{name}_prob_median"]=float(np.median(q)) if len(q) else np.nan
        out[f"{name}_logit_median"]=float(np.median(z)) if len(z) else np.nan
    return out


def format_md(df: pd.DataFrame, cols, digits=3):
    d=df[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c]=d[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.{digits}f}")
    return d.to_markdown(index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--tag", default="std-20260911")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--feature-root", default="/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis", type=Path)
    ap.add_argument("--infer-missing", action="store_true")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of PFMs to analyze; default is all standardized PFMs")
    ap.add_argument("--preload-chunk-gb", type=float, default=60.0,
                    help="max serialized feature bytes per RAM-preload chunk")
    ap.add_argument("--preload-workers", type=int, default=32)
    ap.add_argument("--prefetch-batch", type=int, default=16, help=argparse.SUPPRESS)
    ap.add_argument("--io-workers", type=int, default=8, help=argparse.SUPPRESS)
    args=ap.parse_args()
    repo=args.repo.resolve(); out=args.out.resolve(); feature_root=args.feature_root
    models = args.models or MODELS
    bad = sorted(set(models) - set(MODELS))
    if bad:
        raise ValueError(f"unknown models: {bad}")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    commit,dirty=git_info(repo)
    safe_json(out/"run_meta_prelaunch.json", {
        "study":"loco_threshold_transport", "run_tag":out.name, "purpose":"formal_derived_analysis",
        "status":"running", "repository":str(repo), "git_commit":commit, "git_dirty":dirty,
        "std_tag":args.tag, "seed":42, "models":models, "threshold_source":"source validation only",
        "target_sensitivities":list(TARGET_SENS), "feature_root":str(feature_root),
        "output_path":str(out), "started_at":now_iso(),
        "preload_chunk_gb":args.preload_chunk_gb, "preload_workers":args.preload_workers,
        "notes":"Parent standardized LOCO fold dirs are read-only. Missing prediction dumps are stored only under supplemental_preds/. Missing inference uses bounded chunked WSI_Dataset RAM preload."
    })
    meta=load_meta(repo)
    audit=[]; fold_records=[]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device,flush=True)

    for model in models:
        for mode,fdict in HELD.items():
            for fold,held_out in fdict.items():
                fd=fold_dir(repo,args.tag,model,mode,fold)
                if not fd.is_dir(): raise FileNotFoundError(fd)
                splitcsv=canonical_split(repo,model,mode,fold)
                ckpt,ep=best_checkpoint(fd)
                for split in ("val","test"):
                    expected=split_frame(splitcsv,split)
                    validate_group_isolation(meta,expected,mode,held_out,split)
                    parentp=fd/"preds"/f"slide_preds_{split}.csv"
                    pred=read_prediction(parentp)
                    source="parent_preds"
                    actual_path=parentp
                    if pred is not None:
                        validate_prediction(pred,expected,f"parent {model}/{mode}/{fold}/{split}")
                    else:
                        source="supplemental_inference"
                        actual_path=out/"supplemental_preds"/model/mode/f"fold_{fold}"/f"slide_preds_{split}.csv"
                        if not args.infer_missing:
                            raise RuntimeError(f"missing predictions: {model}/{mode}/{fold}/{split}")
                        print(f"[missing] {model} {mode} fold{fold} {held_out} {split} -> infer",flush=True)
                        pred=infer_split(repo,fd,model,held_out,split,expected,meta,feature_root,actual_path,device,args.preload_chunk_gb,args.preload_workers)
                    audit.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"split":split,
                                  "expected_n":len(expected),"pred_n":len(pred),"prediction_source":source,
                                  "prediction_path":str(actual_path),"split_csv":str(splitcsv),
                                  "checkpoint":str(ckpt),"checkpoint_epoch":ep})
                    fold_records.append((model,mode,fold,held_out,split,pred.copy()))
    audit_df=pd.DataFrame(audit)
    safe_csv(out/"prediction_audit.csv",audit_df)

    # index predictions by fold/split
    P={(m,mo,f,sp):d for m,mo,f,h,sp,d in fold_records}
    thresholds=[]; probs=[]; decisions=[]; transports=[]
    for model in models:
        for mode,fdict in HELD.items():
            for fold,held_out in fdict.items():
                for level in ("slide","patient"):
                    val=P[(model,mode,fold,"val")]
                    test=P[(model,mode,fold,"test")]
                    if level=="patient":
                        val=patient_aggregate(val,meta); test=patient_aggregate(test,meta)
                    else:
                        val=val[["slide_id","label","prob_pos"]].copy(); test=test[["slide_id","label","prob_pos"]].copy()
                    yv=val.label.to_numpy(int); pv=val.prob_pos.to_numpy(float)
                    yt=test.label.to_numpy(int); pt=test.prob_pos.to_numpy(float)
                    vp=probability_metrics(yv,pv); tp=probability_metrics(yt,pt)
                    for split_name,pm in (("val_source",vp),("test_target",tp)):
                        probs.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,"split":split_name,**pm})
                    thrmap={"fixed_0.5":0.5}
                    for target in TARGET_SENS:
                        key=f"source_sens_{int(target*100)}"
                        thrmap[key]=threshold_for_sensitivity(yv,pv,target)
                    for rule,thr in thrmap.items():
                        vdm=decision_metrics(yv,pv,thr); tdm=decision_metrics(yt,pt,thr)
                        thresholds.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,
                                           "threshold_rule":rule,"locked_threshold":thr,"source_n_pos":int(yv.sum()),
                                           "source_sensitivity":vdm["sensitivity"],"source_specificity":vdm["specificity"]})
                        decisions.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,
                                          "threshold_rule":rule,"split":"val_source",**vdm})
                        decisions.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,
                                          "threshold_rule":rule,"split":"test_target",**tdm})
                        transports.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,
                                           "threshold_rule":rule,"locked_threshold":thr,
                                           "source_sensitivity":vdm["sensitivity"],"target_sensitivity":tdm["sensitivity"],
                                           "sensitivity_gap_target_minus_source":tdm["sensitivity"]-vdm["sensitivity"],
                                           "source_specificity":vdm["specificity"],"target_specificity":tdm["specificity"],
                                           "specificity_gap_target_minus_source":tdm["specificity"]-vdm["specificity"],
                                           "source_bacc":vdm["balanced_accuracy"],"target_bacc":tdm["balanced_accuracy"],
                                           "bacc_gap_target_minus_source":tdm["balanced_accuracy"]-vdm["balanced_accuracy"],
                                           "target_ppv":tdm["ppv"],"target_npv":tdm["npv"],
                                           "target_negative_triage_fraction":tdm["negative_triage_fraction"],
                                           "target_fn":tdm["fn"],"target_fp":tdm["fp"]})
                    transports[-1]  # keep loop explicit
                    # probability/calibration transport row independent of threshold
                    transports.append({"model":model,"mode":mode,"fold":fold,"held_out":held_out,"level":level,
                                       "threshold_rule":"probability_scale_shift","locked_threshold":np.nan,
                                       "source_sensitivity":np.nan,"target_sensitivity":np.nan,"sensitivity_gap_target_minus_source":np.nan,
                                       "source_specificity":np.nan,"target_specificity":np.nan,"specificity_gap_target_minus_source":np.nan,
                                       "source_bacc":np.nan,"target_bacc":np.nan,"bacc_gap_target_minus_source":np.nan,
                                       "target_ppv":np.nan,"target_npv":np.nan,"target_negative_triage_fraction":np.nan,
                                       "target_fn":np.nan,"target_fp":np.nan,
                                       "auc_source":vp["auc"],"auc_target":tp["auc"],"auc_gap_target_minus_source":tp["auc"]-vp["auc"],
                                       "brier_source":vp["brier"],"brier_target":tp["brier"],"brier_delta_target_minus_source":tp["brier"]-vp["brier"],
                                       "nll_source":vp["nll"],"nll_target":tp["nll"],"nll_delta_target_minus_source":tp["nll"]-vp["nll"],
                                       "ece10_source":vp["ece10"],"ece10_target":tp["ece10"],"ece10_delta_target_minus_source":tp["ece10"]-vp["ece10"],
                                       "target_calibration_intercept":tp["calibration_intercept"],"target_calibration_slope":tp["calibration_slope"],
                                       "neg_logit_median_shift_target_minus_source":tp["neg_logit_median"]-vp["neg_logit_median"],
                                       "pos_logit_median_shift_target_minus_source":tp["pos_logit_median"]-vp["pos_logit_median"]})

    thresholds_df=pd.DataFrame(thresholds); probs_df=pd.DataFrame(probs); decisions_df=pd.DataFrame(decisions); transports_df=pd.DataFrame(transports)
    safe_csv(out/"locked_thresholds.csv",thresholds_df)
    safe_csv(out/"probability_calibration_metrics.csv",probs_df)
    safe_csv(out/"decision_metrics.csv",decisions_df)
    safe_csv(out/"transport_metrics.csv",transports_df)

    # compact RP table: slide + patient, raw 0.5 + source-95/98
    rp=transports_df[(transports_df.held_out=="RP") & (transports_df.threshold_rule!="probability_scale_shift")].copy()
    rp_prob=transports_df[(transports_df.held_out=="RP") & (transports_df.threshold_rule=="probability_scale_shift")].copy()
    safe_csv(out/"leave_rp_threshold_transport.csv",rp)
    safe_csv(out/"leave_rp_probability_shift.csv",rp_prob)

    # per-PFM across the five held-out groups for source-95 slide-level
    q=transports_df[(transports_df.level=="slide") & (transports_df.threshold_rule=="source_sens_95")].copy()
    agg=(q.groupby("model",as_index=False)
         .agg(mean_target_sens=("target_sensitivity","mean"), worst_target_sens=("target_sensitivity","min"),
              mean_target_spec=("target_specificity","mean"), worst_target_spec=("target_specificity","min"),
              mean_spec_gap=("specificity_gap_target_minus_source","mean"), worst_spec_gap=("specificity_gap_target_minus_source","min"),
              mean_target_negative_triage_fraction=("target_negative_triage_fraction","mean")))
    safe_csv(out/"pfm_source95_summary.csv",agg)

    # report
    rps=rp[rp.level=="slide"].merge(rp_prob[rp_prob.level=="slide"][["model","auc_source","auc_target","brier_target","ece10_target","neg_logit_median_shift_target_minus_source","pos_logit_median_shift_target_minus_source"]],on="model",how="left")
    rps95=rps[rps.threshold_rule=="source_sens_95"].copy()
    rpp95=rp[rp.level.eq("patient") & rp.threshold_rule.eq("source_sens_95")].copy()
    fixed=rp[rp.level.eq("slide") & rp.threshold_rule.eq("fixed_0.5")].copy()
    lines=[f"# Standardized {len(models)}-PFM LOCO: source-locked threshold transport and calibration","",
           f"- Standardized parent tag: `{args.tag}`; Plain AB_MIL seed 42; {len(models)} PFM(s), 5 held-out groups per PFM.",
           "- Thresholds are selected **only on each fold's source validation predictions**; target labels never tune a threshold.",
           "- Primary operating point: source validation sensitivity >=95% with the highest admissible threshold; >=98% is sensitivity analysis; 0.5 is the native softmax threshold.",
           "- Calibration intercept/slope are target evaluation statistics only; they are not used to recalibrate predictions.",
           "- Both slide-level WSI triage and patient-level mean-probability aggregation are reported.","",
           "## Leave-RP: native 0.5 threshold (slide level)","",
           format_md(fixed,["model","target_sensitivity","target_specificity","target_bacc","target_negative_triage_fraction"]),"",
           "## Leave-RP: source-locked 95% sensitivity threshold (slide level)","",
           format_md(rps95,["model","locked_threshold","source_sensitivity","source_specificity","target_sensitivity","target_specificity","target_bacc","target_negative_triage_fraction","auc_target","brier_target","ece10_target","neg_logit_median_shift_target_minus_source"]),"",
           "## Leave-RP: source-locked 95% sensitivity threshold (patient level)","",
           format_md(rpp95,["model","locked_threshold","source_sensitivity","source_specificity","target_sensitivity","target_specificity","target_bacc","target_negative_triage_fraction"]),"",
           "## Five-shift summary at source-locked 95% sensitivity (slide level)","",
           format_md(agg,["model","mean_target_sens","worst_target_sens","mean_target_spec","worst_target_spec","mean_spec_gap","worst_spec_gap","mean_target_negative_triage_fraction"]),"",
           "## Interpretation guardrails","",
           "- AUROC/AUPRC describe ranking; locked-threshold sensitivity/specificity describe operating-point transportability; calibration metrics describe probability-scale transportability. These are distinct axes.",
           "- A model can preserve AUROC while suffering a large specificity or calibration shift when the class-conditional score distributions move relative to the fixed decision threshold.",
           "- Patient-level aggregation can attenuate slide-level instability; slide-level and patient-level findings must not be conflated.",
           "- No target-set threshold calibration is performed in this analysis.","",
           "## Outputs","",
           "- `prediction_audit.csv`: canonical split membership, checkpoint lineage, and parent vs supplemental prediction source.",
           "- `supplemental_preds/`: only folds/splits whose standardized probability dumps were missing; parent result dirs are untouched.",
           "- `locked_thresholds.csv`, `decision_metrics.csv`, `probability_calibration_metrics.csv`, `transport_metrics.csv`.",
           "- `leave_rp_threshold_transport.csv`, `leave_rp_probability_shift.csv`, `pfm_source95_summary.csv`.",
           "- `run_meta_prelaunch.json` and `run_meta.json`."
    ]
    safe_text(out/"REPORT.md","\n".join(lines))
    script=Path(__file__).resolve()
    safe_json(out/"run_meta.json",{
        "study":"loco_threshold_transport","run_tag":out.name,"purpose":"formal_derived_analysis","status":"completed","scientific_status":"active",
        "repository":str(repo),"git_commit":commit,"git_dirty":dirty,"std_tag":args.tag,"seed":42,"models":models,
        "levels":["slide","patient"],"threshold_rules":["fixed_0.5","source_sens_95","source_sens_98"],
        "threshold_source":"source validation only; target labels evaluation-only","calibration":"raw probability evaluation only; no target recalibration",
        "parent_run_pattern":str(repo/"result/ProstateDiagnosis/DataAnalysis/AB_MIL_<model>_loco_{internal,type}/AB_MIL"/f"seed_42_{args.tag}"),
        "supplemental_prediction_policy":"infer only when parent prediction dump missing; write only under derived output",
        "supplemental_inference_io":"WSI_Dataset chunked RAM preload", "preload_chunk_gb":args.preload_chunk_gb, "preload_workers":args.preload_workers,
        "feature_root":str(feature_root),"analysis_script":str(script),"analysis_script_sha256":sha256(script),
        "output_path":str(out),"finished_at":now_iso()
    })
    print("DONE",out,flush=True)

if __name__=="__main__":
    main()
