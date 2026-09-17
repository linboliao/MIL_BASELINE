#!/usr/bin/env python
"""Does Macenko stain-norm shrink the 301 <-> training gap in uni2 feature space?

No trained uni2 model for this task yet, so this is the pre-training check:
mean-pool each slide's uni2 features and compare, for 301 (and ynzl control):

    RAW-uni2-301  vs training      vs      MACENKO-uni2-301  vs training

metrics (lower = closer to training):
  * domain classifier CV-AUC  "is this slide 301?"   (all + benign-RP/TURP-only)
  * MMD^2 (RBF) vs training
  * per-dim standardized mean shift vs training
  * t-SNE for eyeballing

If Macenko clearly pulls 301 toward training -> train a uni2 5-fold and do the
full ensemble eval. If not -> stain-norm isn't closing the gap; don't bother.

out -> /NAS2/Data1/lbliao/Code-195/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis/uni2_stainshift/
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
FEAT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
RAW = {p: f'{FEAT}/{p}/feat_0_224/pt_files/uni2' for p in ('MIL训练数据', 'MIL测试数据', 'MIL外部测试')}
MAC = f'{FEAT}/MIL外部测试/feat_0_224_stains/Macenko/pt_files/uni2'
OUT = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/uni2_stainshift'
RNG = np.random.default_rng(42)
N_TRAIN_REF = 450
D = 1536


def find_raw(sid):
    for d in RAW.values():
        p = f'{d}/{sid}.pt'
        if os.path.exists(p):
            return p
    return None


def meanpool(path):
    x = torch.load(path, map_location='cpu')
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.float().mean(0).numpy()


def build():
    rows = []
    # external: both raw + macenko
    for c in ('301', 'ynzl'):
        d = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/external_test_{c}.csv')
        for _, r in d.iterrows():
            sid = str(r.slide_id)
            rp, mp = find_raw(sid), f'{MAC}/{sid}.pt'
            if rp and os.path.exists(mp):
                rows.append(dict(sid=sid, cohort=c, type=r['type'], label=int(r.label),
                                 raw=rp, mac=mp))
    # internal (raw only) - from internal_test.csv
    it = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/internal_test.csv')
    for _, r in it.iterrows():
        sid = str(r.slide_id)
        rp = find_raw(sid)
        if rp:
            rows.append(dict(sid=sid, cohort='internal', type=r['type'], label=int(r.label),
                             raw=rp, mac=None))
    # training reference (raw only)
    dev = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/dev.csv')
    samp = dev.sample(min(N_TRAIN_REF * 2, len(dev)), random_state=42)
    n = 0
    for _, r in samp.iterrows():
        if n >= N_TRAIN_REF:
            break
        rp = find_raw(str(r.slide_id))
        if rp:
            rows.append(dict(sid=str(r.slide_id), cohort='train', type=r['type'],
                             label=int(r.label), raw=rp, mac=None))
            n += 1
    return pd.DataFrame(rows)


def domain_clf(X, is_target, mask=None):
    if mask is not None:
        X, is_target = X[mask], is_target[mask]
    if is_target.sum() < 8 or (~is_target).sum() < 8:
        return None
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegressionCV(Cs=8, cv=4, max_iter=1500, scoring='roc_auc')
    pred = cross_val_predict(lr, Xs, is_target, cv=StratifiedKFold(4, shuffle=True, random_state=0),
                             method='predict_proba')[:, 1]
    return round(float(roc_auc_score(is_target, pred)), 4)


def mmd2(A, B, gamma=None, nperm=150):
    if gamma is None:
        Z = np.vstack([A, B])[:250]
        dd = np.linalg.norm(Z[:, None] - Z[None, :], axis=-1)
        gamma = 1.0 / (2 * np.median(dd[dd > 0]) ** 2)
    k = lambda x, y: np.exp(-gamma * ((x[:, None] - y[None, :]) ** 2).sum(-1))
    m, n = len(A), len(B)
    Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
    stat = ((Kaa.sum() - np.trace(Kaa)) / (m * (m - 1)) + (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
            - 2 * Kab.mean())
    Z = np.vstack([A, B]); ge = 0
    for _ in range(nperm):
        idx = RNG.permutation(m + n)
        a2, b2 = Z[idx[:m]], Z[idx[m:]]
        K2 = ((k(a2, a2).sum() - m) / (m * (m - 1)) + (k(b2, b2).sum() - n) / (n * (n - 1))
              - 2 * k(a2, b2).mean())
        ge += K2 >= stat
    return round(float(stat), 5), round((ge + 1) / (nperm + 1), 4)


def smd(mu_c, mu_t, sd_t):
    s = np.abs((mu_c - mu_t) / (sd_t + 1e-8))
    return {'mean_abs_SMD': round(float(s.mean()), 3), 'frac_>1': round(float((s > 1).mean()), 3)}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build()
    print(df.groupby('cohort').size().to_dict(), flush=True)

    # pooled feature matrix in a single consistent index order
    N = len(df)
    Xraw = np.zeros((N, D), np.float32)
    Xmac = np.full((N, D), np.nan, np.float32)
    for i, r in enumerate(df.itertuples()):
        Xraw[i] = meanpool(r.raw)
        if isinstance(r.mac, str):
            Xmac[i] = meanpool(r.mac)
        if i % 100 == 0:
            print(f'  pooled {i}/{N}', flush=True)

    coh = df.cohort.values
    typ = df.type.values
    lab = df.label.values
    ref = coh == 'train'
    mu_t, sd_t = Xraw[ref].mean(0), Xraw[ref].std(0)

    res = {'n': df.groupby('cohort').size().to_dict()}
    others_m = np.isin(coh, ['train', 'internal'])
    benign_rt = (lab == 0) & np.isin(typ, ['RP', 'TURP'])
    for c in ('301', 'ynzl'):
        cm = coh == c
        entry = {}
        for tag, Xc in [('raw', Xraw), ('macenko', Xmac)]:
            Xcoh_c = Xc[cm]                      # cohort c features (raw or macenko)
            # --- domain classifier: c vs (train+internal) ---
            Xall = np.vstack([Xraw[others_m], Xcoh_c])
            yall = np.r_[np.zeros(others_m.sum()), np.ones(cm.sum())].astype(int)
            ok = ~np.isnan(Xall).any(1)
            entry[f'domain_auc_{tag}'] = domain_clf(Xall[ok], yall[ok])
            # restricted to benign RP/TURP on both sides
            o_brt = others_m & benign_rt
            c_brt = benign_rt[cm]
            Xb = np.vstack([Xraw[o_brt], Xcoh_c[c_brt]])
            yb = np.r_[np.zeros(o_brt.sum()), np.ones(c_brt.sum())].astype(int)
            okb = ~np.isnan(Xb).any(1)
            entry[f'domain_auc_{tag}_benignRPTURP'] = (
                domain_clf(Xb[okb], yb[okb]) if okb.sum() > 20 and yb[okb].sum() > 6 else None)
            # --- MMD + SMD vs training ---
            Xv = Xcoh_c[~np.isnan(Xcoh_c).any(1)]
            entry[f'mmd2_{tag}_vs_train'] = mmd2(Xraw[ref], Xv)
            entry[f'smd_{tag}_vs_train'] = smd(Xv.mean(0), mu_t, sd_t)
        res[c] = entry

    json.dump(res, open(f'{OUT}/shift_metrics.json', 'w'), indent=2, default=float)

    # t-SNE: train, internal, 301(raw), 301(macenko), ynzl(raw), ynzl(macenko)
    pts, labs = [], []
    for c, X, tg in [('train', Xraw[coh == 'train'], 'train'),
                     ('internal', Xraw[coh == 'internal'], 'internal'),
                     ('301', Xraw[coh == '301'], '301_raw'),
                     ('301', Xmac[coh == '301'], '301_macenko'),
                     ('ynzl', Xraw[coh == 'ynzl'], 'ynzl_raw'),
                     ('ynzl', Xmac[coh == 'ynzl'], 'ynzl_macenko')]:
        X = X[~np.isnan(X).any(1)]
        pts.append(X); labs += [tg] * len(X)
    P = np.vstack(pts)
    Ps = StandardScaler().fit_transform(P)
    emb = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(Ps))
    pd.DataFrame({'group': labs, 'x': emb[:, 0], 'y': emb[:, 1]}).to_csv(f'{OUT}/tsne.csv', index=False)

    # readable
    L = ['UNI2 STAIN-SHIFT CHECK  (mean-pooled features; lower = closer to training)', '=' * 74, '']
    L.append(f'slides: {res["n"]}\n')
    for c in ('301', 'ynzl'):
        e = res[c]
        L.append(f'### {c}')
        L.append(f'  domain-classifier CV-AUC  ("is this {c}?", vs train+internal)')
        L.append(f'      raw      {e["domain_auc_raw"]}      macenko  {e["domain_auc_macenko"]}')
        L.append(f'      benign RP+TURP:  raw {e["domain_auc_raw_benignRPTURP"]}   macenko {e["domain_auc_macenko_benignRPTURP"]}')
        L.append(f'  MMD^2 vs training     raw {e["mmd2_raw_vs_train"][0]}   macenko {e["mmd2_macenko_vs_train"][0]}')
        L.append(f'  per-dim |SMD| vs train  raw mean {e["smd_raw_vs_train"]["mean_abs_SMD"]} (>1: {e["smd_raw_vs_train"]["frac_>1"]})'
                 f'   macenko mean {e["smd_macenko_vs_train"]["mean_abs_SMD"]} (>1: {e["smd_macenko_vs_train"]["frac_>1"]})')
        L.append('')
    txt = '\n'.join(L)
    open(f'{OUT}/SUMMARY.txt', 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'saved -> {OUT}/')


if __name__ == '__main__':
    main()
