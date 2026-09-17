#!/usr/bin/env python
"""Fix attempt #2 - CORAL: align 301 features' full covariance to training's.

Fix #1 (per-dim mean/std) failed -> the shift is higher-order. CORAL matches the
second-order statistics: whiten the source cohort by its own covariance, then
recolour with the training covariance (and recenter the mean).

    A      = Cov_S^(-1/2) @ Cov_T^(1/2)          (2560 x 2560)
    x'     = (x - mu_S) @ A + mu_T

Covariances are shrinkage-regularised:  C <- (1-a)*C + a*(tr(C)/d)*I .
Tried at a few shrinkage levels. Training maps ~identity, so the internal test is
the safety control (should stay near its raw numbers).

Patch stats use up to --patch-cap random patches per slide (covariance estimate
of a 2560-d space needs >> 2560 samples, not every patch). Raw ensemble probs
are reused from groupC_featurenorm/slide_preds.csv when present.

Outputs -> <run_dir>/groupC_coral/
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
sys.path.insert(0, REPO)
os.chdir(REPO)
from modules.AB_MIL.ab_mil import AB_MIL          # noqa: E402
from utils.process_utils import get_act            # noqa: E402
from utils.yaml_utils import read_yaml             # noqa: E402

RES_ROOT = os.path.join(REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
INT_META = os.path.join(REPO, 'datasets/ProstateDiagnosis/internal_test.csv')
EXT_META = {c: os.path.join(REPO, f'datasets/ProstateDiagnosis/external_test_{c}.csv') for c in ('301', 'ynzl')}
DEVICE = 'cuda:0'
RNG = np.random.default_rng(0)
D = 2560


def load_feat(p):
    x = torch.load(p, map_location='cpu')
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.float()


def slide_table(run_dir):
    rows = []
    icsv = pd.read_csv(os.path.join(REPO, 'datasets/ProstateDiagnosis/DataAnalysis/internal_test/internal_test_virchow2_fp16.csv'))
    imeta = pd.read_csv(INT_META)[['slide_id', 'type']].astype(str).set_index('slide_id')['type'].to_dict()
    for p, y in zip(icsv['test_slide_path'], icsv['test_label']):
        b = os.path.basename(p).replace('.pt', '')
        rows.append(('internal', b, imeta.get(b, 'UNK'), int(y), p))
    for c in ('301', 'ynzl'):
        ecsv = pd.read_csv(os.path.join(REPO, f'datasets/ProstateDiagnosis/DataAnalysis/external_test/external_test_{c}_virchow2_fp16.csv'))
        em = pd.read_csv(EXT_META[c])[['slide_id', 'type']].astype(str).set_index('slide_id')['type'].to_dict()
        for p, y in zip(ecsv['test_slide_path'], ecsv['test_label']):
            b = os.path.basename(p).replace('.pt', '')
            rows.append((c, b, em.get(b, 'UNK'), int(y), p))
    tr = pd.read_csv(os.path.join(run_dir, 'fold_1', 'prostate_dev_virchow2_fp16_1fold.csv'))
    for p in tr['train_slide_path'].dropna():
        rows.append(('train', os.path.basename(p).replace('.pt', ''), '', -1, p))
    df = pd.DataFrame(rows, columns=['cohort', 'slide_id', 'type', 'label', 'path'])
    return df[df.path.map(lambda p: isinstance(p, str) and os.path.exists(p))].reset_index(drop=True)


def mean_cov(paths, cap, desc):
    s = torch.zeros(D, dtype=torch.float64, device=DEVICE)
    S = torch.zeros(D, D, dtype=torch.float64, device=DEVICE)
    n = 0
    for i, p in enumerate(paths):
        x = load_feat(p)
        if x.shape[0] > cap:
            idx = torch.from_numpy(RNG.choice(x.shape[0], cap, replace=False))
            x = x[idx]
        xg = x.to(DEVICE, torch.float64)
        s += xg.sum(0)
        S += xg.T @ xg
        n += xg.shape[0]
        if i % 100 == 0:
            print(f'  [{desc}] {i}/{len(paths)}  sampled_patches={n:,}', flush=True)
    mu = s / n
    cov = S / n - torch.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)
    return mu.cpu().numpy(), cov.cpu().numpy(), n


def shrink(C, a):
    d = C.shape[0]
    return (1 - a) * C + a * (np.trace(C) / d) * np.eye(d)


def mat_pow_sym(C, p, eps=1e-8):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    return (V * (w ** p)) @ V.T


def coral_A(mu_s, cov_s, mu_t, cov_t, a):
    Cs = shrink(cov_s, a)
    Ct = shrink(cov_t, a)
    return (mat_pow_sym(Cs, -0.5) @ mat_pow_sym(Ct, 0.5)).astype(np.float32)


def load_models(run_dir):
    ya = read_yaml(os.path.join(run_dir, 'fold_1', 'fold_1.yaml'))
    ms = []
    for k in range(1, 6):
        m = AB_MIL(L=ya.Model.L, D=ya.Model.D, num_classes=2, dropout=ya.Model.dropout,
                   act=get_act(ya.Model.act), in_dim=ya.Model.in_dim)
        ck = sorted(glob.glob(os.path.join(run_dir, f'fold_{k}', 'Best_EPOCH_*.pth')))[-1]
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        ms.append(m.to(DEVICE).eval())
    return ms


@torch.no_grad()
def ens(models, x):
    """x may already be on GPU. Falls back to CPU for a slide that OOMs."""
    try:
        xg = x.to(DEVICE)
        ps = [torch.softmax(m(xg)['logits'].squeeze(0), -1)[1].item() for m in models]
    except RuntimeError:
        torch.cuda.empty_cache()
        xc = x.detach().to('cpu')
        ps = []
        for m in models:
            m.to('cpu')
            ps.append(torch.softmax(m(xc)['logits'].squeeze(0), -1)[1].item())
            m.to(DEVICE)
    return float(np.mean(ps))


def Mrow(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {'n': int(len(y)), 'auc': round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            'sens': round(float(tp / (tp + fn)), 4) if tp + fn else None,
            'spec': round(float(tn / (tn + fp)), 4) if tn + fp else None,
            'acc': round(float((tp + tn) / len(y)), 4), 'cm': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def report_block(sub, col):
    d = sub.rename(columns={col: 'prob'})
    r = {'ALL': Mrow(d.label, d.prob)}
    for t in ['CNB', 'RP', 'TURP']:
        g = d[d.type == t]
        if len(g) >= 5:
            r[t] = Mrow(g.label, g.prob)
    g = d[(d.label == 0) & d.type.isin(['RP', 'TURP'])]
    if len(g):
        r['benign_RPTURP_spec'] = round(float((g.prob < .5).mean()), 4)
    g = d[d.label == 1]
    if len(g):
        r['cancer_sens'] = round(float((g.prob >= .5).mean()), 4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1])
    ap.add_argument('--patch-cap', type=int, default=3000)
    ap.add_argument('--alphas', type=float, nargs='+', default=[0.05, 0.2])
    a = ap.parse_args()
    run_dir = a.run_dir
    out = os.path.join(run_dir, 'groupC_coral')
    os.makedirs(out, exist_ok=True)

    df = slide_table(run_dir)
    print(df.groupby('cohort').size().to_dict(), flush=True)
    models = load_models(run_dir)

    tcache = os.path.join(out, f'train_meancov_cap{a.patch_cap}.npz')
    if os.path.exists(tcache):
        z = np.load(tcache); mu_t, cov_t = z['mu'], z['cov']
        print(f'loaded cached train mean/cov (n={int(z["n"]):,})', flush=True)
    else:
        mu_t, cov_t, nt = mean_cov(df[df.cohort == 'train'].path.tolist(), a.patch_cap, 'train')
        np.savez(tcache, mu=mu_t, cov=cov_t, n=nt)

    coh = {}
    for c in ['internal', '301', 'ynzl']:
        mu, cov, n = mean_cov(df[df.cohort == c].path.tolist(), a.patch_cap, c)
        coh[c] = (mu, cov, n)

    # precompute transforms
    mu_t_g = torch.from_numpy(mu_t.astype(np.float32)).to(DEVICE)
    A = {}
    for c in ['internal', '301', 'ynzl']:
        mu_s, cov_s, _ = coh[c]
        for al in a.alphas:
            A[(c, al)] = (torch.from_numpy(coral_A(mu_s, cov_s, mu_t, cov_t, al)).to(DEVICE),
                          torch.from_numpy(mu_s.astype(np.float32)).to(DEVICE))
    print('CORAL transforms ready', flush=True)

    # reuse raw probs
    raw_csv = os.path.join(run_dir, 'groupC_featurenorm', 'slide_preds.csv')
    raw_map = {}
    if os.path.exists(raw_csv):
        rr = pd.read_csv(raw_csv)
        raw_map = dict(zip(rr.slide_id.astype(str), rr.prob_raw))

    ev = df[df.cohort.isin(['internal', '301', 'ynzl'])].reset_index(drop=True)
    cols = {'prob_raw': []}
    for al in a.alphas:
        cols[f'prob_coral_a{al}'] = []
    for i, row in ev.iterrows():
        x = load_feat(row.path)
        cols['prob_raw'].append(raw_map.get(row.slide_id) if row.slide_id in raw_map else ens(models, x))
        big = x.shape[0] > 120000
        xg = x if big else x.to(DEVICE)
        for al in a.alphas:
            Amat, mu_s_g = A[(row.cohort, al)]
            if big:                                   # transform on CPU for giant slides
                xc = (x - mu_s_g.cpu()) @ Amat.cpu() + mu_t_g.cpu()
            else:
                xc = (xg - mu_s_g) @ Amat + mu_t_g
            cols[f'prob_coral_a{al}'].append(ens(models, xc))
        if i % 50 == 0:
            print(f'  infer {i}/{len(ev)}', flush=True)
    for k, v in cols.items():
        ev[k] = v
    ev.to_csv(os.path.join(out, 'slide_preds.csv'), index=False, encoding='utf-8-sig')

    variants = ['prob_raw'] + [f'prob_coral_a{al}' for al in a.alphas]
    res = {}
    for c in ['301', 'ynzl', 'internal']:
        sub = ev[ev.cohort == c]
        res[c] = {v: report_block(sub, v) for v in variants}
    json.dump({'patch_cap': a.patch_cap, 'alphas': a.alphas, 'results': res},
              open(os.path.join(out, 'results.json'), 'w'), indent=2, default=float)

    L = [f'CORAL FIX - run {os.path.basename(run_dir)}', '=' * 74,
         f'full-covariance alignment to training patch stats; shrinkage alphas={a.alphas}; '
         f'patch cap {a.patch_cap}', '']
    for c in ['301', 'ynzl', 'internal']:
        L.append(f'### {c}')
        for v in variants:
            r = res[c][v]; al = r['ALL']
            tag = v.replace('prob_', '')
            L.append(f'  [{tag:11s}] ALL AUC {al["auc"]}  sens {al["sens"]}  spec {al["spec"]}  acc {al["acc"]}  cm {al["cm"]}')
            ex = []
            if 'benign_RPTURP_spec' in r: ex.append(f'benign RP+TURP spec {r["benign_RPTURP_spec"]}')
            if 'cancer_sens' in r: ex.append(f'cancer sens {r["cancer_sens"]}')
            if ex: L.append('               ' + '  |  '.join(ex))
            for t in ['CNB', 'RP', 'TURP']:
                if t in r:
                    rt = r[t]
                    L.append(f'               {t:5s} n={rt["n"]:3d} AUC {rt["auc"]} sens {rt["sens"]} spec {rt["spec"]}')
        L.append('')
    txt = '\n'.join(L)
    open(os.path.join(out, 'SUMMARY.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'saved -> {out}/')


if __name__ == '__main__':
    main()
