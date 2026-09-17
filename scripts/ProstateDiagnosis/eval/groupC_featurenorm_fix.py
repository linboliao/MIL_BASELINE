#!/usr/bin/env python
"""Fix attempt #1 - align 301 features to training statistics, re-run ensemble.

The shift is broad (43% of 2560 dims move >1 SD, Group B). This applies a
per-dimension affine correction that maps a cohort's feature distribution onto
the TRAINING distribution's per-dim mean/std, then re-runs the frozen 5-fold
AB_MIL ensemble. Training maps to itself (identity), so a cohort that already
matches training (internal) is left ~unchanged - that is the safety property.

correction:   x' = (x - mu_C) / sd_C * sd_train + mu_train      (per feature dim)

Two ways to get mu_C / sd_C:
  - cohort  : pooled over ALL patches of the cohort (transductive / test-time BN)
  - slide   : per slide, from its own patches (inductive instance-norm -> train scale)

Reports raw vs corrected, overall and by specimen type: AUC, sens, spec@0.5,
confusion. The question is whether corrected recovers benign RP/TURP specificity
WITHOUT dropping cancer sensitivity, and without hurting the internal test.

Outputs -> <run_dir>/groupC_featurenorm/
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
EPS = 1e-6


# ----------------------------- data plumbing -----------------------------
def load_feat(path):
    x = torch.load(path, map_location='cpu')
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.float()


def slide_table(run_dir):
    """slide_id, cohort, type, label, path  for internal / 301 / ynzl / train."""
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


def patch_stats(paths, desc=''):
    """streaming per-dim mean/std over all patches of the given slides (fp64)."""
    s = np.zeros(2560, np.float64)
    ss = np.zeros(2560, np.float64)
    n = 0
    for i, p in enumerate(paths):
        x = load_feat(p).numpy().astype(np.float64)
        s += x.sum(0)
        ss += (x * x).sum(0)
        n += x.shape[0]
        if i % 100 == 0:
            print(f'  [{desc}] {i}/{len(paths)}  patches={n:,}', flush=True)
    mu = s / n
    var = np.maximum(ss / n - mu ** 2, 0.0)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32) + EPS, n


# ----------------------------- model / inference -----------------------------
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
def ensemble_prob(models, x):
    try:
        xg = x.to(DEVICE)
        ps = [torch.softmax(m(xg)['logits'].squeeze(0), -1)[1].item() for m in models]
    except RuntimeError:                       # OOM on a giant slide -> CPU
        torch.cuda.empty_cache()
        cpu = [m.to('cpu') for m in models]
        ps = [torch.softmax(m(x)['logits'].squeeze(0), -1)[1].item() for m in cpu]
        for m in cpu:
            m.to(DEVICE)
    return float(np.mean(ps))


def apply_affine(x, mu_c, sd_c, mu_t, sd_t):
    return (x - mu_c) / sd_c * sd_t + mu_t


# ----------------------------- metrics -----------------------------
def M(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {'n': int(len(y)), 'prev': round(float(y.mean()), 3),
            'auc': round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            'sens': round(float(tp / (tp + fn)), 4) if tp + fn else None,
            'spec': round(float(tn / (tn + fp)), 4) if tn + fp else None,
            'acc': round(float((tp + tn) / len(y)), 4),
            'cm': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def strata_report(df):
    r = {'ALL': M(df.label, df.prob)}
    for t, g in df.groupby('type'):
        if len(g) >= 5:
            r[f'type_{t}'] = M(g.label, g.prob)
    g = df[(df.label == 0) & df.type.isin(['RP', 'TURP'])]
    if len(g):
        r['benign_RP+TURP_spec'] = round(float((g.prob < 0.5).mean()), 4)
    g = df[df.label == 1]
    if len(g):
        r['cancer_sens'] = round(float((g.prob >= 0.5).mean()), 4)
    return r


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1])
    ap.add_argument('--modes', nargs='+', default=['cohort', 'slide'],
                    choices=['cohort', 'slide'])
    a = ap.parse_args()
    run_dir = a.run_dir
    out = os.path.join(run_dir, 'groupC_featurenorm')
    os.makedirs(out, exist_ok=True)

    df = slide_table(run_dir)
    print(df.groupby('cohort').size().to_dict(), flush=True)
    models = load_models(run_dir)

    # --- training stats (cached) ---
    stcache = os.path.join(out, 'train_patch_stats.npz')
    if os.path.exists(stcache):
        z = np.load(stcache); mu_t, sd_t = z['mu'], z['sd']
        print(f'loaded cached training stats (patches={int(z["n"])})', flush=True)
    else:
        mu_t, sd_t, nt = patch_stats(df[df.cohort == 'train'].path.tolist(), 'train')
        np.savez(stcache, mu=mu_t, sd=sd_t, n=nt)

    # --- per-cohort stats ---
    coh_stats = {}
    for c in ['internal', '301', 'ynzl']:
        mu, sd, n = patch_stats(df[df.cohort == c].path.tolist(), c)
        coh_stats[c] = (mu, sd, n)

    mu_t_t = torch.from_numpy(mu_t); sd_t_t = torch.from_numpy(sd_t)

    # --- inference: raw + each mode ---
    variants = ['raw'] + a.modes
    preds = {v: [] for v in variants}
    ev = df[df.cohort.isin(['internal', '301', 'ynzl'])].reset_index(drop=True)
    for i, row in ev.iterrows():
        x = load_feat(row.path)
        preds['raw'].append(ensemble_prob(models, x))
        if 'cohort' in a.modes:
            mu_c, sd_c, _ = coh_stats[row.cohort]
            xc = apply_affine(x, torch.from_numpy(mu_c), torch.from_numpy(sd_c), mu_t_t, sd_t_t)
            preds['cohort'].append(ensemble_prob(models, xc))
        if 'slide' in a.modes:
            mu_s = x.mean(0); sd_s = x.std(0) + EPS
            xs = apply_affine(x, mu_s, sd_s, mu_t_t, sd_t_t)
            preds['slide'].append(ensemble_prob(models, xs))
        if i % 50 == 0:
            print(f'  infer {i}/{len(ev)}', flush=True)

    for v in variants:
        ev[f'prob_{v}'] = preds[v]
    ev.to_csv(os.path.join(out, 'slide_preds.csv'), index=False, encoding='utf-8-sig')

    # --- report ---
    report = {'raw_patch_level_shift_vs_training': {}}
    for c in ['internal', '301', 'ynzl']:
        mu_c, sd_c, _ = coh_stats[c]
        smd_raw = np.abs((mu_c - mu_t) / (sd_t + EPS))
        report['raw_patch_level_shift_vs_training'][c] = {
            'frac_|SMD|>1': round(float((smd_raw > 1).mean()), 3),
            'mean_|SMD|': round(float(smd_raw.mean()), 3)}

    res = {}
    for c in ['301', 'ynzl', 'internal']:
        sub = ev[ev.cohort == c]
        res[c] = {}
        for v in variants:
            d = sub.rename(columns={f'prob_{v}': 'prob'})
            res[c][v] = strata_report(d[['type', 'label', 'prob']])
    report['results'] = res
    json.dump(report, open(os.path.join(out, 'results.json'), 'w'), indent=2, default=float)

    # --- readable ---
    L = [f'FEATURE-NORM FIX  -  run {os.path.basename(run_dir)}', '=' * 74,
         'affine per-dim correction to TRAINING patch statistics, frozen 5-fold ensemble', '']
    for c in ['301', 'ynzl', 'internal']:
        L.append(f'### {c}')
        for v in variants:
            r = res[c][v]
            a0 = r['ALL']
            L.append(f'  [{v:6s}] ALL  AUC {a0["auc"]}  sens {a0["sens"]}  spec {a0["spec"]}  acc {a0["acc"]}  cm {a0["cm"]}')
            extra = []
            if 'benign_RP+TURP_spec' in r:
                extra.append(f'benign RP+TURP spec {r["benign_RP+TURP_spec"]}')
            if 'cancer_sens' in r:
                extra.append(f'cancer sens {r["cancer_sens"]}')
            if extra:
                L.append(f'           {"  |  ".join(extra)}')
            for t in ['CNB', 'RP', 'TURP']:
                if f'type_{t}' in r:
                    rt = r[f'type_{t}']
                    L.append(f'           {t:5s} n={rt["n"]:3d}  AUC {rt["auc"]}  sens {rt["sens"]}  spec {rt["spec"]}')
        L.append('')
    txt = '\n'.join(L)
    open(os.path.join(out, 'SUMMARY.txt'), 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'saved -> {out}/')


if __name__ == '__main__':
    main()
