#!/usr/bin/env python
"""Stratified internal-test results + side-by-side vs the external cohorts.

Internal test = the fixed 332-slide held-out set (internal_test.csv metadata:
3 training centers 省立/新昌/迈新, specimen types CNB/RP/TURP). Answers: does the
model over-call benign RP/TURP on data from its OWN centers, or only on 301?

Writes <run_dir>/internal_results/ : per_stratum_metrics.{json,csv},
slide_predictions.csv, SUMMARY.txt  (and appends the internal-vs-external
benign-score comparison to SUMMARY.txt).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
RES_ROOT = os.path.join(
    REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
INT_META = os.path.join(REPO, 'datasets/ProstateDiagnosis/internal_test.csv')


def m(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yh = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, yh, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'n': int(len(y)), 'n_pos': int(y.sum()), 'prevalence': round(float(y.mean()), 3),
        'auc': round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
        'sens': round(float(tp / (tp + fn)), 4) if tp + fn else None,
        'spec': round(float(tn / (tn + fp)), 4) if tn + fp else None,
        'acc': round(float((tp + tn) / len(y)), 4),
        'ppv': round(float(tp / (tp + fp)), 4) if tp + fp else None,
        'cm': cm.tolist(),
    }


def load_folds(d):
    fp = {}
    for k in range(1, 6):
        f = os.path.join(d, f'fold_{k}', 'Infer_Result.csv')
        if os.path.exists(f):
            fp[k] = pd.read_csv(f)[['slide_id', 'label', 'prob_1']].set_index('slide_id')
    return fp


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1]
    out = os.path.join(run_dir, 'internal_results')
    os.makedirs(out, exist_ok=True)

    fp = load_folds(os.path.join(run_dir, 'internal_test'))
    if not fp:
        sys.exit('no internal_test Infer_Result.csv - run run_virchow2_internal_eval.sh first')

    meta = pd.read_csv(INT_META)[['slide_id', 'label', 'type', 'center']].copy()
    meta['slide_id'] = meta['slide_id'].astype(str)
    base = next(iter(fp.values())).reset_index()[['slide_id', 'label']].copy()
    base['slide_id'] = base['slide_id'].astype(str)
    tab = base.merge(meta[['slide_id', 'type', 'center']], on='slide_id', how='left')
    tab['type'] = tab['type'].fillna('UNK')
    tab['center'] = tab['center'].fillna('UNK')
    for k, df in fp.items():
        tab[f'prob_fold{k}'] = tab['slide_id'].map(df['prob_1'])
    pcols = [f'prob_fold{k}' for k in fp]
    tab['prob_ensemble'] = tab[pcols].mean(axis=1)
    tab['pred_ensemble'] = (tab['prob_ensemble'] >= 0.5).astype(int)
    tab.to_csv(os.path.join(out, 'slide_predictions.csv'), index=False, encoding='utf-8-sig')

    res = {'overall': {}, 'by_type': {}, 'by_center': {}, 'by_center_type': {}}
    res['overall']['ensemble'] = m(tab['label'], tab['prob_ensemble'])
    res['overall']['per_fold'] = {k: m(tab['label'], tab[f'prob_fold{k}']) for k in fp}

    for t, g in tab.groupby('type'):
        res['by_type'][t] = m(g['label'], g['prob_ensemble'])
    for c, g in tab.groupby('center'):
        res['by_center'][c] = m(g['label'], g['prob_ensemble'])
    for (c, t), g in tab.groupby(['center', 'type']):
        res['by_center_type'][f'{c}/{t}'] = m(g['label'], g['prob_ensemble'])

    # benign-score comparison internal vs external
    ext_sp = os.path.join(run_dir, 'external_results', 'slide_predictions.csv')
    benign_cmp = {}
    tab_b = tab[tab.label == 0]
    for t, g in tab_b.groupby('type'):
        benign_cmp.setdefault(t, {})['internal(3-center)'] = {
            'n': int(len(g)), 'median_prob': round(float(g['prob_ensemble'].median()), 3),
            'frac_called_cancer': round(float((g['prob_ensemble'] >= .5).mean()), 3)}
    if os.path.exists(ext_sp):
        e = pd.read_csv(ext_sp)
        for coh in ['301', 'ynzl']:
            eb = e[(e.cohort == coh) & (e.label == 0)]
            for t, g in eb.groupby('type'):
                benign_cmp.setdefault(t, {})[f'external_{coh}'] = {
                    'n': int(len(g)), 'median_prob': round(float(g['prob_ensemble'].median()), 3),
                    'frac_called_cancer': round(float((g['prob_ensemble'] >= .5).mean()), 3)}

    res['benign_score_internal_vs_external'] = benign_cmp
    with open(os.path.join(out, 'per_stratum_metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    rows = []
    for grp, dd in [('type', res['by_type']), ('center', res['by_center']), ('center/type', res['by_center_type'])]:
        for name, mm in dd.items():
            rows.append({'stratum_kind': grp, 'stratum': name, **mm})
    rows.append({'stratum_kind': 'overall', 'stratum': 'ALL', **res['overall']['ensemble']})
    pd.DataFrame(rows).to_csv(os.path.join(out, 'per_stratum_metrics.csv'), index=False)

    L = [f'INTERNAL held-out test - run {os.path.basename(run_dir)}  (ensemble of available folds, thr 0.5)',
         '=' * 78, '']
    o = res['overall']['ensemble']
    L.append(f"OVERALL  n={o['n']}  prev={o['prevalence']}  AUC {o['auc']}  sens {o['sens']}  spec {o['spec']}  acc {o['acc']}")
    L.append(f"  confusion [[TN,FP],[FN,TP]] = {o['cm']}")
    L.append('\nby specimen type:')
    for t, mm in res['by_type'].items():
        L.append(f"  {t:5s} n={mm['n']:3d} prev={mm['prevalence']:<5}  AUC {mm['auc']}  sens {mm['sens']}  spec {mm['spec']}")
    L.append('\nby center:')
    for c, mm in res['by_center'].items():
        L.append(f"  {c:8s} n={mm['n']:3d} prev={mm['prevalence']:<5}  AUC {mm['auc']}  sens {mm['sens']}  spec {mm['spec']}")
    L.append('\nby center / type:')
    for name, mm in res['by_center_type'].items():
        L.append(f"  {name:16s} n={mm['n']:3d} prev={mm['prevalence']:<5}  AUC {mm['auc']}  sens {mm['sens']}  spec {mm['spec']}")
    L.append('\n\nBENIGN-slide score: does the model over-call benign RP/TURP on its OWN centers?')
    L.append(f"  {'type':6s} {'source':22s} {'n':>4s} {'median_prob':>12s} {'frac_called_cancer':>18s}")
    for t, srcs in benign_cmp.items():
        for src, v in srcs.items():
            L.append(f"  {t:6s} {src:22s} {v['n']:>4d} {v['median_prob']:>12.3f} {v['frac_called_cancer']:>18.3f}")
    txt = '\n'.join(L)
    open(os.path.join(out, 'SUMMARY.txt'), 'w').write(txt + '\n')
    print(txt)
    print(f'\nsaved -> {out}/')


if __name__ == '__main__':
    main()
