"""PSIR rigorous re-check — one model-parametrized pipeline, subcommands.

Question under test: is PSIR's "301 specificity jump" real, or an artifact of
(a) comparing against a DIFFERENT PFM's bare model, (b) the projection head just
doing dimensionality reduction / nonlinear reg, or (c) a decision-threshold shift?

Per PFM we run three matched conditions on the identical 5-fold-3-center split
(StratifiedGroupKFold, seed 42) and identical external eval:

  bare : raw <model> features (native dim)                 -> AB_MIL_<model>_recheck_bare
  psir : Panel-A SupCon projection head, 5 folds  (dim 256) -> AB_MIL_<model>_recheck_psir_fold{K}
  shuf : SAME projection recipe, case labels SHUFFLED       -> AB_MIL_<model>_recheck_shuf_fold{K}
         (negative control: breaks the "same case across centers" signal;
          if 301 still improves, the effect is NOT panel invariance)

subcommands:  stage | proj | apply | folds | configs | eval
Driven by psir_recheck.sh.  Paths are repo-relative + $PROSTATE_FEAT_ROOT /
$PSIR_CACHE (local fp16 mirror, kept for reuse).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
DS = _REPO / "datasets" / "ProstateDiagnosis"
PSIR_DIR = DS / "psir"
RESULT = _REPO / "result" / "ProstateDiagnosis" / "DataAnalysis"
NAS = os.environ.get("PROSTATE_FEAT_ROOT", "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")
CACHE = os.environ.get("PSIR_CACHE", "/tmp/psir_cache")

DIM = {"conch": 512, "uni": 1024, "uni2": 1536, "virchow2": 2560,
       "h-optimus-1": 1536, "gigapath": 1536, "gpfm": 1024, "mstar": 1024}
PROJ_DIM = 256
POOL_DIR = {"dev": "MIL训练数据", "oldtest": "MIL测试数据", "ext_sl": "MIL外部测试"}
SEED = 42
N_SPLITS = 5
# SupCon projection hyper-params — identical to the original UNI PSIR run
SC_TEMP, SC_LR, SC_EPOCHS, SC_PATCH_CAP = 0.1, 1e-3, 200, 1024


def rr(model, variant, fold=None):
    """result/dataset dir name for a (model, variant[, fold])."""
    if variant == "bare":
        return f"AB_MIL_{model}_recheck_bare"
    return f"AB_MIL_{model}_recheck_{variant}_fold{fold}"


def feat_model_name(model, variant, fold):
    return model if variant == "bare" else f"{model}_recheck_{variant}_fold{fold}"


# ---------------------------------------------------------------- stage
def _stage_one(job):
    """module-level so ProcessPoolExecutor can pickle it (GIL-bound work -> processes)."""
    import torch
    torch.set_num_threads(1)
    src, dst = job
    if os.path.exists(dst):
        return 0
    try:
        t = torch.load(src, map_location="cpu", weights_only=True).half()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        torch.save(t, tmp); os.rename(tmp, dst)
        return 1
    except Exception as e:
        print("FAIL", src, e); return -1


def cmd_stage(a):
    from concurrent.futures import ProcessPoolExecutor
    dev = pd.read_csv(DS / "dev_clean.csv", dtype={"patient_id": str})
    itest = pd.read_csv(DS / "internal_test_clean.csv", dtype={"patient_id": str})
    ext = []
    for fn in ("external_test_301.csv", "external_test_ynzl.csv"):
        d = pd.read_csv(DS / fn)
        d = d[d["pool"].astype(str).str.startswith("ext")].copy()
        d["feat_dir"] = "MIL外部测试"
        ext.append(d)
    main = pd.concat([dev, itest], ignore_index=True)
    main["feat_dir"] = main["pool"].map(POOL_DIR)
    rows = pd.concat([main, *ext], ignore_index=True)
    rows["stem"] = rows["filename"].map(lambda x: os.path.splitext(str(x))[0])
    jobs = [(f"{NAS}/{r.feat_dir}/feat_0_224/pt_files/{a.model}/{r.stem}.pt",
             f"{CACHE}/{r.feat_dir}/feat_0_224/pt_files/{a.model}/{r.stem}.pt")
            for r in rows.dropna(subset=["feat_dir"]).itertuples()]
    # Panel A/B (all slides — projection needs A train-signal, eval needs A held-out + all B)
    for panel in ("SerialPanelA", "SerialPanelB"):
        src_dir = f"{NAS}/{panel}/feat_0_224/pt_files/{a.model}"
        if os.path.isdir(src_dir):
            for fn in os.listdir(src_dir):
                if fn.endswith(".pt"):
                    jobs.append((f"{src_dir}/{fn}",
                                 f"{CACHE}/{panel}/feat_0_224/pt_files/{a.model}/{fn}"))

    print(f"stage {a.model}: {len(jobs)} files -> {CACHE}  ({a.workers} processes)")
    d = s = f = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_stage_one, jobs, chunksize=4), 1):
            d += r == 1; s += r == 0; f += r == -1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} new={d} skip={s} fail={f}")
    print(f"done new={d} skip={s} fail={f}")
    if f:
        sys.exit(1)


# ---------------------------------------------------------------- proj
class ProjHead:
    pass  # placeholder; real nn.Module built inside cmd_proj/apply to defer torch import


def _proj_module(in_dim):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
                         nn.Linear(in_dim, PROJ_DIM))


def _supcon(z, gid, temp):
    import torch
    import torch.nn.functional as F
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / temp
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    gid = gid.view(-1, 1)
    pos = (gid == gid.t()) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1); ok = cnt > 0
    if ok.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    return (-(logp * pos).sum(1)[ok] / cnt[ok]).mean()


def cmd_proj(a):
    import torch
    in_dim = DIM[a.model]
    fdir = f"{CACHE}/SerialPanelA/feat_0_224/pt_files/{a.model}"
    folds = pd.read_csv(PSIR_DIR / "panel_a_case_folds.csv", dtype={"case_id": str})
    slides = pd.read_csv(PSIR_DIR / "panel_a_usable_slides.csv", dtype={"case_id": str})
    tr_cases = set(folds.loc[folds[f"fold{a.fold}"] == "train_signal", "case_id"])
    sl = slides[slides["case_id"].isin(tr_cases)].copy()
    sl["stem"] = sl["filename"].map(lambda x: os.path.splitext(str(x))[0])
    sl = sl[sl["stem"].map(lambda s: os.path.exists(f"{fdir}/{s}.pt"))].reset_index(drop=True)
    cases = sorted(sl["case_id"].unique())
    c2i = {c: i for i, c in enumerate(cases)}
    print(f"[proj {a.model} fold{a.fold}{' SHUF' if a.shuffle else ''}] "
          f"{len(sl)} slides / {len(cases)} cases")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(SEED)
    bags, gid = [], []
    for _, row in sl.iterrows():
        t = torch.load(f"{fdir}/{row['stem']}.pt", map_location="cpu", weights_only=True).float()
        if t.shape[0] > SC_PATCH_CAP:
            t = t[torch.randperm(t.shape[0], generator=g)[:SC_PATCH_CAP]].contiguous()
        bags.append(t); gid.append(c2i[row["case_id"]])
    gid = torch.tensor(gid, dtype=torch.long)
    if a.shuffle:                       # negative control: permute the case grouping
        gid = gid[torch.randperm(len(gid), generator=torch.Generator().manual_seed(SEED + 1))]

    torch.manual_seed(SEED)
    proj = _proj_module(in_dim).to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=SC_LR, weight_decay=1e-5)
    for ep in range(1, SC_EPOCHS + 1):
        proj.train()
        z = torch.stack([proj(b.to(dev)).mean(0) for b in bags], 0)
        loss = _supcon(z, gid.to(dev), SC_TEMP)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 5.0); opt.step()
        if ep % 40 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{SC_EPOCHS} supcon {loss.item():.4f}")

    variant = "shuf" if a.shuffle else "psir"
    out = PSIR_DIR / "recheck_proj" / f"{a.model}_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(proj.state_dict(), out / f"fold{a.fold}_proj.pt")
    json.dump({"model": a.model, "variant": variant, "fold": a.fold, "in_dim": in_dim,
               "proj_dim": PROJ_DIM, "n_cases": len(cases), "n_slides": len(sl),
               "final_loss": float(loss.item())},
              open(out / f"fold{a.fold}_meta.json", "w"), indent=2)
    print(f"  -> {out}/fold{a.fold}_proj.pt")


# ---------------------------------------------------------------- apply
def cmd_apply(a):
    import torch
    in_dim = DIM[a.model]
    proj = _proj_module(in_dim).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    ck = PSIR_DIR / "recheck_proj" / f"{a.model}_{a.variant}" / f"fold{a.fold}_proj.pt"
    dev = next(proj.parameters()).device
    proj.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    dst_model = feat_model_name(a.model, a.variant, a.fold)

    def do(src, dst):
        if os.path.exists(dst):
            return 0
        t = torch.load(src, map_location="cpu", weights_only=True).float()
        with torch.no_grad():
            o = proj(t.to(dev)).cpu()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(o.half(), dst + ".tmp"); os.rename(dst + ".tmp", dst)
        return 1

    n = 0
    # main cohort + externals
    dev_df = pd.read_csv(DS / "dev_clean.csv", dtype={"patient_id": str})
    it_df = pd.read_csv(DS / "internal_test_clean.csv", dtype={"patient_id": str})
    for df in (dev_df, it_df):
        df["feat_dir"] = df["pool"].map(POOL_DIR)
    exts = []
    for fn in ("external_test_301.csv", "external_test_ynzl.csv"):
        d = pd.read_csv(DS / fn); d = d[d["pool"].astype(str).str.startswith("ext")].copy()
        d["feat_dir"] = "MIL外部测试"; exts.append(d)
    for df in (dev_df, it_df, *exts):
        for _, r in df.dropna(subset=["feat_dir"]).iterrows():
            stem = os.path.splitext(str(r["filename"]))[0]
            s = f"{CACHE}/{r['feat_dir']}/feat_0_224/pt_files/{a.model}/{stem}.pt"
            d = f"{CACHE}/{r['feat_dir']}/feat_0_224/pt_files/{dst_model}/{stem}.pt"
            if os.path.exists(s):
                n += do(s, d)
    # Panel A held-out cases of this fold + all Panel B
    folds = pd.read_csv(PSIR_DIR / "panel_a_case_folds.csv", dtype={"case_id": str})
    sld = pd.read_csv(PSIR_DIR / "panel_a_usable_slides.csv", dtype={"case_id": str})
    held = set(folds.loc[folds[f"fold{a.fold}"] == "held_out", "case_id"])
    for _, r in sld[sld["case_id"].isin(held)].iterrows():
        stem = os.path.splitext(str(r["filename"]))[0]
        s = f"{CACHE}/SerialPanelA/feat_0_224/pt_files/{a.model}/{stem}.pt"
        d = f"{CACHE}/SerialPanelA/feat_0_224/pt_files/{dst_model}/{stem}.pt"
        if os.path.exists(s):
            n += do(s, d)
    pb = f"{CACHE}/SerialPanelB/feat_0_224/pt_files/{a.model}"
    if os.path.isdir(pb):
        for fn in os.listdir(pb):
            if fn.endswith(".pt"):
                n += do(f"{pb}/{fn}", f"{CACHE}/SerialPanelB/feat_0_224/pt_files/{dst_model}/{fn}")
    print(f"[apply {dst_model}] projected {n} new files")


# ---------------------------------------------------------------- folds
def cmd_folds(a):
    from sklearn.model_selection import StratifiedGroupKFold
    fold = a.fold
    fmodel = feat_model_name(a.model, a.variant, fold)
    outroot = DS / "DataAnalysis" / rr(a.model, a.variant, fold)
    dev = pd.read_csv(DS / "dev_clean.csv", dtype={"slide_id": str, "patient_id": str})
    it = pd.read_csv(DS / "internal_test_clean.csv", dtype={"slide_id": str, "patient_id": str})
    for df in (dev, it):
        df["feat_dir"] = df["pool"].map(POOL_DIR)
        df["stem"] = df["filename"].map(lambda x: os.path.splitext(str(x))[0])
        df["feat"] = df.apply(
            lambda r: f"{CACHE}/{r['feat_dir']}/feat_0_224/pt_files/{fmodel}/{r['stem']}.pt", axis=1)
        df.drop(df[~df["feat"].map(os.path.exists)].index, inplace=True)
    print(f"[folds {rr(a.model, a.variant, fold)}] dev {len(dev)} | internal_test {len(it)}")
    pat = (dev.groupby("patient_id")
           .agg(label=("label", lambda s: int(s.max())),
                center=("center", lambda s: s.mode().iat[0])).reset_index())
    pat["stratum"] = pat["label"].astype(str) + "|" + pat["center"].astype(str)
    vc = pat["stratum"].value_counts()
    pat.loc[pat["stratum"].isin(vc[vc < N_SPLITS].index), "stratum"] = pat["label"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(pat, pat["stratum"], groups=pat["patient_id"]))
    tp, tl = it["feat"].tolist(), it["label"].tolist()
    outroot.mkdir(parents=True, exist_ok=True)
    for old in outroot.glob("*fold.csv"):
        old.unlink()
    for k, (tri, vai) in enumerate(splits, 1):
        trp = set(pat.loc[tri, "patient_id"]); vap = set(pat.loc[vai, "patient_id"])
        tr = dev[dev["patient_id"].isin(trp)]; va = dev[dev["patient_id"].isin(vap)]
        n = max(len(tr), len(va), len(tp))
        pad = lambda x: x + [None] * (n - len(x))
        out = pd.DataFrame({"train_slide_path": pad(tr["feat"].tolist()),
                            "train_label": pad(tr["label"].tolist()),
                            "val_slide_path": pad(va["feat"].tolist()),
                            "val_label": pad(va["label"].tolist()),
                            "test_slide_path": pad(tp), "test_label": pad(tl)})
        # all CV CSVs in outroot/ (not outroot/fold_k/) -> one train_mil.py k-fold loop
        outroot.mkdir(parents=True, exist_ok=True)
        out.to_csv(outroot / f"prostate_{rr(a.model, a.variant, fold)}_{k}fold.csv", index=False)
        print(f"  cv{k}: train {len(tr)} val {len(va)} test {len(tp)}")


# ---------------------------------------------------------------- configs
_TEMPLATE = """General:
  MODEL_NAME: AB_MIL
  seed: 42
  num_classes: 2
  num_epochs: 50
  device: {gpu}
  num_workers: 2
  best_model_metric: macro_f1
  earlystop: {{use: true, patience: 15, metric: macro_f1}}
Dataset:
  DATASET_NAME: {name}
  dataset_csv_path: null
  dataset_root_dir: datasets/ProstateDiagnosis/DataAnalysis/{name}
  balanced_sampler: {{use: false, replacement: true}}
Logs:
  log_root_dir: result/ProstateDiagnosis/DataAnalysis
Model:
  in_dim: {in_dim}
  L: 512
  D: 128
  dropout: 0.1
  act: relu
  optimizer:
    which: adam
    adam_config: {{lr: 0.0002, weight_decay: 1.0e-05}}
    adamw_config: {{lr: 0.0002, weight_decay: 1.0e-05}}
  criterion: {{loss: ce}}
  scheduler:
    warmup: 2
    which: step
    step_config: {{step_size: 3, gamma: 0.9}}
    multi_step_config: {{milestones: [20, 30, 40], gamma: 0.9}}
    exponential_config: {{gamma: 0.9}}
    cosine_config: {{T_max: 10, eta_min: 0.0001}}
"""


def cmd_configs(a):
    name = rr(a.model, a.variant, a.fold)
    in_dim = DIM[a.model] if a.variant == "bare" else PROJ_DIM
    out = _REPO / "configs" / "ProstateDiagnosis" / "DataAnalysis"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.yaml"
    p.write_text(_TEMPLATE.format(name=name, in_dim=in_dim, gpu=a.gpu))
    print(f"[configs {name}] in_dim {in_dim} gpu {a.gpu} -> {p}")


# ---------------------------------------------------------------- eval
def _metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, p = np.asarray(y), np.asarray(p); yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    m = len(set(y.tolist())) > 1
    return dict(n=len(y), auc=float(roc_auc_score(y, p)) if m else float("nan"),
                auprc=float(average_precision_score(y, p)) if m else float("nan"),
                sens=tp / (tp + fn) if tp + fn else float("nan"),
                spec=tn / (tn + fp) if tn + fp else float("nan"),
                acc=(tp + tn) / len(y), cm=[[tn, fp], [fn, tp]], thr=float(thr))


def _thr_at_sens(y, p, target=0.95):
    """smallest threshold whose sensitivity >= target (calibrated on internal_test)."""
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def _load_feats(model_name, feat_dir, stems):
    import torch
    out = {}
    for s in stems:
        p = f"{CACHE}/{feat_dir}/feat_0_224/pt_files/{model_name}/{s}.pt"
        if os.path.exists(p):
            out[s] = torch.load(p, map_location="cpu", weights_only=True).float()
    return out


def _cv_models(name, in_dim):
    import torch
    import torch.nn as nn
    from modules.AB_MIL.ab_mil import AB_MIL
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = []
    new_seed = sorted(glob.glob(str(RESULT / name / "AB_MIL" / "seed_*")), key=os.path.getmtime)
    for cv in range(1, N_SPLITS + 1):
        # new layout: <name>/AB_MIL/seed_*/fold_<cv>/Best_EPOCH_*.pth
        best = sorted(glob.glob(f"{new_seed[-1]}/fold_{cv}/Best_EPOCH_*.pth"),
                      key=lambda q: int(q.split("_")[-1].split(".")[0])) if new_seed else []
        if not best:  # old nested layout
            seeds = sorted(glob.glob(str(RESULT / name / f"fold_{cv}" / "*" / "AB_MIL" / "seed_*")),
                           key=os.path.getmtime)
            best = (sorted(glob.glob(f"{seeds[-1]}/fold_1/Best_EPOCH_*.pth"),
                           key=lambda q: int(q.split("_")[-1].split(".")[0])) if seeds else [])
        if not best:
            continue
        m = AB_MIL(L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=in_dim).to(dev).eval()
        m.load_state_dict(torch.load(best[-1], map_location=dev, weights_only=True))
        mods.append(m)
    return mods, dev


def cmd_eval(a):
    import torch
    variant, model = a.variant, a.model
    in_dim = DIM[model] if variant == "bare" else PROJ_DIM
    folds_iter = [None] if variant == "bare" else list(range(1, N_SPLITS + 1))

    coh = []
    for fn, site in (("external_test_301", "301"), ("external_test_ynzl", "ynzl")):
        d = pd.read_csv(DS / f"{fn}.csv"); d = d[d["pool"].astype(str).str.startswith("ext")].copy()
        d["stem"] = d["filename"].map(lambda x: os.path.splitext(str(x))[0]); d["site"] = site
        coh.append(d[["stem", "label", "type", "site"]])
    coh = pd.concat(coh, ignore_index=True)
    it = pd.read_csv(DS / "internal_test_clean.csv", dtype={"patient_id": str})
    it["stem"] = it["filename"].map(lambda x: os.path.splitext(str(x))[0])

    prob_cols = []
    for K in folds_iter:
        fmodel = feat_model_name(model, variant, K)
        name = rr(model, variant, K)
        mods, dev = _cv_models(name, in_dim)
        if not mods:
            print(f"  !! no CV models for {name}"); continue
        ext_f = _load_feats(fmodel, "MIL外部测试", coh["stem"].unique())
        it_f = _load_feats(fmodel, "MIL测试数据", it["stem"].unique())
        tag = "p" if K is None else f"p{K}"
        prob_cols.append(tag)

        def predict(fdict, stems):
            out = {}
            with torch.no_grad():
                for s in stems:
                    if s not in fdict:
                        continue
                    x = fdict[s].to(dev).unsqueeze(0)
                    pr = [torch.softmax(m(x)["logits"], -1)[0, 1].item() for m in mods]
                    out[s] = float(np.mean(pr))
            return out
        coh[tag] = coh["stem"].map(predict(ext_f, coh["stem"]))
        it[tag] = it["stem"].map(predict(it_f, it["stem"]))

    if not prob_cols:
        print("no predictions produced"); sys.exit(1)
    coh["p"] = coh[prob_cols].mean(axis=1)      # ensemble over folds (bare: single col)
    it["p"] = it[prob_cols].mean(axis=1)

    thr95 = _thr_at_sens(it["label"].values, it["p"].values, 0.95)
    res = {"model": model, "variant": variant, "in_dim": in_dim,
           "n_cv_folds": len(prob_cols), "thr_sens95_internal": thr95, "sites": {}}
    print(f"\n===== PSIR-recheck {model} / {variant}  (thr@sens95_internal={thr95:.3f}) =====")
    for site in ("301", "ynzl", "ALL"):
        sub = coh if site == "ALL" else coh[coh.site == site]
        m05 = _metrics(sub["label"].values, sub["p"].values, 0.5)
        m95 = _metrics(sub["label"].values, sub["p"].values, thr95)
        res["sites"][site] = {"thr0.5": m05, "thr_sens95": m95}
        print(f"  {site:4} n={m05['n']:3d} | AUC {m05['auc']:.3f} AUPRC {m05['auprc']:.3f} "
              f"| @0.5 sens {m05['sens']:.3f} spec {m05['spec']:.3f} "
              f"| @s95 sens {m95['sens']:.3f} spec {m95['spec']:.3f}  cm05 {m05['cm']}")
    # 301 benign RP+TURP specificity (the magnification-shift metric)
    b = coh[(coh.site == "301") & (coh.label == 0) & (coh.type.isin(["RP", "TURP"]))]
    if len(b):
        res["sites"]["301_benign_RP_TURP"] = {
            "spec@0.5": float((b["p"].values < 0.5).mean()),
            "spec@s95": float((b["p"].values < thr95).mean()), "n": int(len(b))}
        print(f"  301 benign RP+TURP spec: @0.5 {res['sites']['301_benign_RP_TURP']['spec@0.5']:.3f} "
              f"| @s95 {res['sites']['301_benign_RP_TURP']['spec@s95']:.3f}  (n={len(b)})")

    RESULT.mkdir(parents=True, exist_ok=True)
    out = RESULT / f"psir_recheck_{model}_{variant}.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    coh.to_csv(RESULT / f"psir_recheck_{model}_{variant}_slidepreds.csv", index=False)
    print(f"  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--model", required=True); s.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 8))); s.set_defaults(fn=cmd_stage)
    s = sub.add_parser("proj"); s.add_argument("--model", required=True); s.add_argument("--fold", type=int, required=True); s.add_argument("--shuffle", action="store_true"); s.set_defaults(fn=cmd_proj)
    s = sub.add_parser("apply"); s.add_argument("--model", required=True); s.add_argument("--fold", type=int, required=True); s.add_argument("--variant", choices=["psir", "shuf"], required=True); s.set_defaults(fn=cmd_apply)
    s = sub.add_parser("folds"); s.add_argument("--model", required=True); s.add_argument("--variant", choices=["bare", "psir", "shuf"], required=True); s.add_argument("--fold", type=int, default=0); s.set_defaults(fn=cmd_folds)
    s = sub.add_parser("configs"); s.add_argument("--model", required=True); s.add_argument("--variant", choices=["bare", "psir", "shuf"], required=True); s.add_argument("--fold", type=int, default=0); s.add_argument("--gpu", type=int, default=0); s.set_defaults(fn=cmd_configs)
    s = sub.add_parser("eval"); s.add_argument("--model", required=True); s.add_argument("--variant", choices=["bare", "psir", "shuf"], required=True); s.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    if a.cmd in ("folds", "configs") and a.variant != "bare" and not a.fold:
        ap.error("--fold required for psir/shuf")
    sys.path.insert(0, str(_REPO))
    a.fn(a)
