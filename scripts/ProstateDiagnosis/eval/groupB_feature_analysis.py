#!/usr/bin/env python
"""Group B - characterize the 301 domain shift in virchow2 feature space.

Pools each slide's [N,2560] virchow2 features (mean-pool) and also runs them
through fold-1's trained AB_MIL to get the attention-pooled 512-d rep + the
per-patch attention weights. Then:

  1. 2D embedding (PCA + t-SNE) of slide-level mean-pooled features, by center/type/label
  2. Domain classifier: LogReg 5-fold CV AUC for "301 vs the rest" (and restricted
     to benign RP+TURP only, to rule out label-composition); top driving dims
  3. Per-dimension standardized shift (301 / ynzl / internal mean  vs  training)
  4. MMD^2 (RBF, median-heuristic) training vs each cohort, permutation p
  5. slide score vs #patches and vs feature-norm, per cohort x type
  6. attention concentration (max weight / entropy) 301 vs internal, by type

Reference "training" set = a random 400-slide sample of fold-1's train column.
Outputs -> <run_dir>/groupB_analysis/  (npy vectors, csv tables, SUMMARY.txt)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
sys.path.insert(0, REPO)
os.chdir(REPO)
RES_ROOT = os.path.join(REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
INT_META = os.path.join(REPO, 'datasets/ProstateDiagnosis/internal_test.csv')
EXT_META = {'301': os.path.join(REPO, 'datasets/ProstateDiagnosis/external_test_301.csv'),
            'ynzl': os.path.join(REPO, 'datasets/ProstateDiagnosis/external_test_ynzl.csv')}
RNG = np.random.default_rng(42)
N_TRAIN_REF = 400
DEVICE = 'cuda:0'


def build_slide_list(run_dir):
    rows = []
    # internal
    ip = pd.read_csv(os.path.join(run_dir, 'internal_results', 'slide_predictions.csv'))
    icsv = pd.read_csv(os.path.join(REPO, 'datasets/ProstateDiagnosis/DataAnalysis/internal_test/internal_test_virchow2_fp16.csv'))
    path_by_base = {os.path.basename(p).replace('.pt', ''): p for p in icsv['test_slide_path']}
    ip['slide_id'] = ip['slide_id'].astype(str)
    for _, r in ip.iterrows():
        rows.append(dict(slide_id=r.slide_id, cohort='internal', center=r['center'], type=r['type'],
                         label=int(r.label), prob=float(r.prob_ensemble),
                         path=path_by_base.get(r.slide_id)))
    # external
    ep = pd.read_csv(os.path.join(run_dir, 'external_results', 'slide_predictions.csv'))
    ep['slide_id'] = ep['slide_id'].astype(str)
    for coh in ('301', 'ynzl'):
        ecsv = pd.read_csv(os.path.join(REPO, f'datasets/ProstateDiagnosis/DataAnalysis/external_test/external_test_{coh}_virchow2_fp16.csv'))
        pbb = {os.path.basename(p).replace('.pt', ''): p for p in ecsv['test_slide_path']}
        for _, r in ep[ep.cohort == coh].iterrows():
            rows.append(dict(slide_id=r.slide_id, cohort=f'ext_{coh}', center=r['center'], type=r['type'],
                             label=int(r.label), prob=float(r.prob_ensemble), path=pbb.get(r.slide_id)))
    # training reference
    tr = pd.read_csv(os.path.join(run_dir, 'fold_1', 'prostate_dev_virchow2_fp16_1fold.csv'))
    trp = tr['train_slide_path'].dropna().tolist()
    tm = pd.read_csv(os.path.join(REPO, 'datasets/ProstateDiagnosis/dev.csv'))
    tmeta = {str(s): (c, t) for s, c, t in zip(tm.slide_id, tm.center, tm.type)}
    tl = {str(s): int(l) for s, l in zip(tm.slide_id, tm.label)}
    sample = RNG.choice(len(trp), size=min(N_TRAIN_REF, len(trp)), replace=False)
    for i in sample:
        p = trp[i]
        b = os.path.basename(p).replace('.pt', '')
        c, t = tmeta.get(b, ('?', '?'))
        rows.append(dict(slide_id=b, cohort='train_ref', center=c, type=t,
                         label=tl.get(b, -1), prob=np.nan, path=p))
    df = pd.DataFrame(rows)
    return df[df.path.notna() & df.path.map(lambda p: os.path.exists(str(p)))].reset_index(drop=True)


@torch.no_grad()
def pool_all(df):
    import glob as _glob
    from modules.AB_MIL.ab_mil import AB_MIL
    from utils.process_utils import get_act
    from utils.yaml_utils import read_yaml
    ya = read_yaml(os.path.join(RUN_DIR, 'fold_1', 'fold_1.yaml'))
    model = AB_MIL(L=ya.Model.L, D=ya.Model.D, num_classes=2, dropout=ya.Model.dropout,
                   act=get_act(ya.Model.act), in_dim=ya.Model.in_dim)
    ck = sorted(_glob.glob(os.path.join(RUN_DIR, 'fold_1', 'Best_EPOCH_*.pth')))[-1]
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
    model.to(DEVICE).eval()

    mean2560 = np.zeros((len(df), 2560), np.float32)
    m512 = np.zeros((len(df), ya.Model.L), np.float32)
    npatch = np.zeros(len(df), int)
    fnorm = np.zeros(len(df), np.float32)
    max_attn = np.zeros(len(df), np.float32)
    attn_ent = np.zeros(len(df), np.float32)
    for i, p in enumerate(df.path):
        x = torch.load(p, map_location='cpu')
        if x.dim() == 3:
            x = x.squeeze(0)
        x = x.float()
        npatch[i] = x.shape[0]
        mean2560[i] = x.mean(0).numpy()
        fnorm[i] = x.norm(dim=1).mean().item()
        xg = x.to(DEVICE)
        out = model(xg, return_WSI_attn=True, return_WSI_feature=True)
        m512[i] = out['WSI_feature'].squeeze(0).cpu().numpy()
        a = torch.softmax(out['WSI_attn'].squeeze(-1), dim=-1).cpu().numpy()
        max_attn[i] = a.max()
        p_ = np.clip(a, 1e-12, 1)
        attn_ent[i] = float(-(p_ * np.log(p_)).sum() / np.log(len(p_)))  # normalized 0..1
        if i % 100 == 0:
            print(f'  pooled {i}/{len(df)}', flush=True)
    df = df.copy()
    df['n_patch'] = npatch
    df['feat_norm'] = fnorm
    df['max_attn'] = max_attn
    df['attn_entropy'] = attn_ent
    return df, mean2560, m512


def domain_clf(X, is301, mask=None, tag=''):
    if mask is not None:
        X, is301 = X[mask], is301[mask]
    if is301.sum() < 8 or (~is301).sum() < 8:
        return {'tag': tag, 'note': 'too few', 'n_301': int(is301.sum())}
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, scoring='roc_auc')
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(lr, Xs, is301, cv=cv, method='predict_proba')[:, 1]
    auc = roc_auc_score(is301, pred)
    lr.fit(Xs, is301)
    top = np.argsort(-np.abs(lr.coef_[0]))[:20].tolist()
    return {'tag': tag, 'n': int(len(is301)), 'n_301': int(is301.sum()),
            'cv_auc_301_vs_rest': round(float(auc), 4), 'top_dims': top}


def mmd2_rbf(A, B, gamma=None, nperm=200):
    if gamma is None:
        Z = np.vstack([A, B])
        d = np.linalg.norm(Z[:200, None] - Z[None, :200], axis=-1)
        gamma = 1.0 / (2 * np.median(d[d > 0]) ** 2)

    def k(x, y):
        return np.exp(-gamma * ((x[:, None] - y[None, :]) ** 2).sum(-1))
    m, n = len(A), len(B)
    Kaa, Kbb, Kab = k(A, A), k(B, B), k(A, B)
    stat = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1)) + (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)) - 2 * Kab.mean()
    Z = np.vstack([A, B])
    ge = 0
    for _ in range(nperm):
        idx = RNG.permutation(m + n)
        a2, b2 = Z[idx[:m]], Z[idx[m:]]
        Kaa2, Kbb2, Kab2 = k(a2, a2), k(b2, b2), k(a2, b2)
        s2 = (Kaa2.sum() - np.trace(Kaa2)) / (m * (m - 1)) + (Kbb2.sum() - np.trace(Kbb2)) / (n * (n - 1)) - 2 * Kab2.mean()
        ge += s2 >= stat
    return float(stat), (ge + 1) / (nperm + 1)


def main():
    global RUN_DIR
    RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1]
    out = os.path.join(RUN_DIR, 'groupB_analysis')
    os.makedirs(out, exist_ok=True)
    print('slide list ...', flush=True)
    df = build_slide_list(RUN_DIR)
    print(df.groupby('cohort').size().to_dict(), flush=True)

    df, mean2560, m512 = pool_all(df)
    np.save(os.path.join(out, 'pooled_mean2560.npy'), mean2560)
    np.save(os.path.join(out, 'pooled_attn512.npy'), m512)
    df.to_csv(os.path.join(out, 'slide_meta.csv'), index=False, encoding='utf-8-sig')

    res = {}

    # ---- 1. 2D embedding ----
    Xs = StandardScaler().fit_transform(mean2560)
    p50 = PCA(50, random_state=0).fit_transform(Xs)
    ts = TSNE(2, perplexity=30, random_state=0, init='pca').fit_transform(p50)
    emb = df[['slide_id', 'cohort', 'center', 'type', 'label', 'prob', 'n_patch']].copy()
    emb['tsne_x'], emb['tsne_y'] = ts[:, 0], ts[:, 1]
    pc2 = PCA(2, random_state=0).fit_transform(Xs)
    emb['pc1'], emb['pc2'] = pc2[:, 0], pc2[:, 1]
    emb.to_csv(os.path.join(out, 'embedding_coords.csv'), index=False, encoding='utf-8-sig')

    # ---- 2. domain classifier ----
    is301 = (df.cohort == 'ext_301').values
    benign_rp_turp = df.label.eq(0).values & df.type.isin(['RP', 'TURP']).values
    res['domain_clf'] = {
        'all_slides': domain_clf(mean2560, is301, tag='301 vs rest, all slides'),
        'benign_RP_TURP_only': domain_clf(mean2560, is301, benign_rp_turp, tag='301 vs rest, benign RP+TURP only'),
    }

    # ---- 3. per-dim standardized shift ----
    ref = df.cohort == 'train_ref'
    mu, sd = mean2560[ref].mean(0), mean2560[ref].std(0) + 1e-8
    shift = {}
    for coh in ['ext_301', 'ext_ynzl', 'internal']:
        smd = (mean2560[df.cohort == coh].mean(0) - mu) / sd
        shift[coh] = {'mean_abs_SMD': float(np.abs(smd).mean()),
                      'frac_|SMD|>1': float((np.abs(smd) > 1).mean()),
                      'frac_|SMD|>2': float((np.abs(smd) > 2).mean()),
                      'top10_dims': np.argsort(-np.abs(smd))[:10].tolist(),
                      'top10_SMD': [round(float(x), 2) for x in smd[np.argsort(-np.abs(smd))[:10]]]}
    res['per_dim_shift_vs_training'] = shift

    # ---- 4. MMD ----
    A = mean2560[ref]
    res['mmd2_rbf_vs_training'] = {}
    for coh in ['ext_301', 'ext_ynzl', 'internal']:
        s, pv = mmd2_rbf(A, mean2560[df.cohort == coh])
        res['mmd2_rbf_vs_training'][coh] = {'mmd2': round(s, 5), 'perm_p': pv}

    # ---- 5. score vs size / norm ----
    sc = []
    for (coh, t), g in df[df.cohort.str.startswith(('internal', 'ext'))].groupby(['cohort', 'type']):
        if g.prob.notna().sum() < 5:
            continue
        rho_n, p_n = spearmanr(g.n_patch, g.prob)
        rho_f, p_f = spearmanr(g.feat_norm, g.prob)
        sc.append({'cohort': coh, 'type': t, 'n': len(g),
                   'spearman_prob_vs_npatch': round(float(rho_n), 3), 'p_npatch': round(float(p_n), 4),
                   'spearman_prob_vs_featnorm': round(float(rho_f), 3), 'p_featnorm': round(float(p_f), 4)})
    pd.DataFrame(sc).to_csv(os.path.join(out, 'score_vs_size.csv'), index=False)
    # 301 benign RP/TURP specifically
    g = df[(df.cohort == 'ext_301') & (df.label == 0) & df.type.isin(['RP', 'TURP'])]
    res['score_301_benign_RPTURP'] = {
        'n': len(g), 'median_prob': round(float(g.prob.median()), 3),
        'spearman_prob_vs_npatch': [round(float(x), 3) for x in spearmanr(g.n_patch, g.prob)],
        'spearman_prob_vs_featnorm': [round(float(x), 3) for x in spearmanr(g.feat_norm, g.prob)]}

    # ---- 6. attention concentration ----
    att = {}
    for t in ['RP', 'TURP', 'CNB']:
        gi = df[(df.cohort == 'internal') & (df.type == t)]
        g3 = df[(df.cohort == 'ext_301') & (df.type == t)]
        if len(gi) < 5 or len(g3) < 5:
            continue
        att[t] = {
            'max_attn_internal_median': round(float(gi.max_attn.median()), 4),
            'max_attn_301_median': round(float(g3.max_attn.median()), 4),
            'entropy_internal_median': round(float(gi.attn_entropy.median()), 4),
            'entropy_301_median': round(float(g3.attn_entropy.median()), 4),
            'mwu_p_entropy': round(float(mannwhitneyu(gi.attn_entropy, g3.attn_entropy).pvalue), 5)}
    res['attention_concentration'] = att

    json.dump(res, open(os.path.join(out, 'groupB_metrics.json'), 'w'), indent=2, ensure_ascii=False, default=float)

    # readable
    L = [f'GROUP B - feature-space characterization - run {os.path.basename(RUN_DIR)}', '=' * 78, '']
    L.append(f"slides: {df.groupby('cohort').size().to_dict()}\n")
    L.append('2. DOMAIN CLASSIFIER (LogReg 5-fold CV, predict \"is this slide from 301?\")')
    for k, v in res['domain_clf'].items():
        if 'cv_auc_301_vs_rest' in v:
            L.append(f"   {v['tag']:38s} n={v['n']:4d} n_301={v['n_301']:3d}  CV-AUC = {v['cv_auc_301_vs_rest']}")
        else:
            L.append(f"   {v['tag']:38s} {v.get('note')}")
    L.append("   (AUC ~1.0 => 301 is trivially separable in feature space => strong covariate shift)\n")
    L.append('3. PER-DIMENSION SHIFT vs training (standardized mean diff over 2560 dims)')
    for coh, v in shift.items():
        L.append(f"   {coh:12s} mean|SMD| {v['mean_abs_SMD']:.3f}  |SMD|>1: {v['frac_|SMD|>1']:.1%}  |SMD|>2: {v['frac_|SMD|>2']:.1%}")
    L.append('')
    L.append('4. MMD^2 (RBF) vs training  [larger = more distributional shift]')
    for coh, v in res['mmd2_rbf_vs_training'].items():
        L.append(f"   {coh:12s} MMD^2 {v['mmd2']:.5f}   perm-p {v['perm_p']:.3g}")
    L.append('')
    L.append('5. SLIDE SCORE vs #PATCHES / FEATURE-NORM (Spearman rho)')
    for r in sc:
        L.append(f"   {r['cohort']:12s} {r['type']:5s} n={r['n']:3d}  "
                 f"rho(prob, n_patch)={r['spearman_prob_vs_npatch']:+.2f} (p={r['p_npatch']})  "
                 f"rho(prob, featnorm)={r['spearman_prob_vs_featnorm']:+.2f} (p={r['p_featnorm']})")
    g6 = res['score_301_benign_RPTURP']
    L.append(f"   >> 301 benign RP+TURP (n={g6['n']}, median prob {g6['median_prob']}): "
             f"rho vs n_patch {g6['spearman_prob_vs_npatch']}, rho vs featnorm {g6['spearman_prob_vs_featnorm']}")
    L.append('')
    L.append('6. ATTENTION CONCENTRATION (median; entropy normalized 0..1)')
    for t, v in att.items():
        L.append(f"   {t:5s}  max_attn  internal {v['max_attn_internal_median']}  vs 301 {v['max_attn_301_median']}   "
                 f"| entropy internal {v['entropy_internal_median']} vs 301 {v['entropy_301_median']} (MWU p={v['mwu_p_entropy']})")
    txt = '\n'.join(L)
    open(os.path.join(out, 'SUMMARY.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'\nsaved -> {out}/')


if __name__ == '__main__':
    main()
