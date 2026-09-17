#!/usr/bin/env python
"""Per-center external-test results for the virchow2 5-fold run.

Reads the 5 fold Infer_Result.csv per external cohort (301 = center "301",
ynzl = center "云南肿瘤"), joins slide-level metadata (label / specimen type /
patient id), and reports, per center:
  * each of the 5 fold checkpoints, scored independently
  * the 5-fold probability ENSEMBLE (mean of prob_1) - the headline number
  * a specimen-type breakdown (CNB / RP / TURP) for the ensemble
All at the 0.5 decision threshold; AUC is threshold-free.

Writes, under <run_dir>/external_results/ :
  per_center_metrics.json      full numbers (per fold + ensemble + by type)
  per_center_metrics.csv       flat table, one row per (center, model)
  slide_predictions.csv        every slide: metadata + 5 fold probs + ensemble
  SUMMARY.txt                  human-readable tables

Usage: summarize_external_eval.py [RUN_DIR]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             f1_score, confusion_matrix, cohen_kappa_score)

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
RES_ROOT = os.path.join(
    REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
META = {
    '301': (os.path.join(REPO, 'datasets/ProstateDiagnosis/external_test_301.csv'), '301'),
    'ynzl': (os.path.join(REPO, 'datasets/ProstateDiagnosis/external_test_ynzl.csv'), '云南肿瘤'),
}


def metrics(y, p1):
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float)
    yhat = (p1 >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'n': int(len(y)),
        'n_pos': int(y.sum()),
        'prevalence': float(y.mean()),
        'auc': float(roc_auc_score(y, p1)) if len(set(y)) > 1 else None,
        'acc': float(accuracy_score(y, yhat)),
        'balanced_acc': float(balanced_accuracy_score(y, yhat)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) else None,
        'specificity': float(tn / (tn + fp)) if (tn + fp) else None,
        'ppv': float(tp / (tp + fp)) if (tp + fp) else None,
        'npv': float(tn / (tn + fn)) if (tn + fn) else None,
        'macro_f1': float(f1_score(y, yhat, average='macro')),
        'quadratic_kappa': float(cohen_kappa_score(y, yhat, weights='quadratic')),
        'confusion_matrix_[[TN,FP],[FN,TP]]': cm.tolist(),
    }


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(
        (os.path.join(RES_ROOT, d) for d in os.listdir(RES_ROOT) if d.startswith('run_')))[-1]
    ext = os.path.join(run_dir, 'external_test')
    out = os.path.join(run_dir, 'external_results')
    os.makedirs(out, exist_ok=True)
    print(f'run_dir: {run_dir}')

    summary = {}
    flat_rows = []
    all_sltables = []

    for cohort, (meta_csv, center_name) in META.items():
        meta = pd.read_csv(meta_csv)[['slide_id', 'label', 'type', 'patient_id']]

        fold_p = {}
        for k in range(1, 6):
            f = os.path.join(ext, cohort, f'fold_{k}', 'Infer_Result.csv')
            if not os.path.exists(f):
                print(f'  WARNING missing {f}')
                continue
            d = pd.read_csv(f)[['slide_id', 'label', 'prob_1']]
            fold_p[k] = d.set_index('slide_id')['prob_1']

        if not fold_p:
            print(f'{cohort}: no fold results, skipping')
            continue

        tab = meta.set_index('slide_id').copy()
        for k, s in fold_p.items():
            tab[f'prob_fold{k}'] = s
        pcols = [f'prob_fold{k}' for k in fold_p]
        tab['prob_ensemble'] = tab[pcols].mean(axis=1)
        tab['pred_ensemble'] = (tab['prob_ensemble'] >= 0.5).astype(int)
        tab = tab.dropna(subset=pcols)
        tab.insert(0, 'center', center_name)
        tab.insert(1, 'cohort', cohort)
        all_sltables.append(tab.reset_index())

        center_res = {'center': center_name, 'cohort': cohort, 'per_fold': {}, 'ensemble': None, 'ensemble_by_type': {}}
        for k in fold_p:
            m = metrics(tab['label'], tab[f'prob_fold{k}'])
            center_res['per_fold'][k] = m
            flat_rows.append({'center': center_name, 'cohort': cohort, 'model': f'fold{k}', **m})
        # mean +- std over folds
        agg = {}
        for key in ['auc', 'acc', 'balanced_acc', 'sensitivity', 'specificity', 'macro_f1', 'quadratic_kappa']:
            vals = [center_res['per_fold'][k][key] for k in fold_p if center_res['per_fold'][k][key] is not None]
            agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        center_res['per_fold_mean_std'] = agg

        ens = metrics(tab['label'], tab['prob_ensemble'])
        center_res['ensemble'] = ens
        flat_rows.append({'center': center_name, 'cohort': cohort, 'model': 'ensemble_5fold', **ens})

        for t, g in tab.groupby('type'):
            if len(set(g['label'])) > 1:
                center_res['ensemble_by_type'][t] = metrics(g['label'], g['prob_ensemble'])
            else:
                center_res['ensemble_by_type'][t] = {'n': int(len(g)), 'note': 'single-class, metrics skipped'}

        summary[cohort] = center_res

    # combined (pooled) external
    if all_sltables:
        allt = pd.concat(all_sltables, ignore_index=True)
        allt.to_csv(os.path.join(out, 'slide_predictions.csv'), index=False, encoding='utf-8-sig')
        pooled = metrics(allt['label'], allt['prob_ensemble'])
        summary['_pooled_all_external'] = {'ensemble': pooled}
        flat_rows.append({'center': 'ALL', 'cohort': 'pooled', 'model': 'ensemble_5fold', **pooled})

    pd.DataFrame(flat_rows).to_csv(os.path.join(out, 'per_center_metrics.csv'), index=False)
    with open(os.path.join(out, 'per_center_metrics.json'), 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # readable
    lines = [f'External test - per center - run {os.path.basename(run_dir)}', '=' * 70, '']
    for cohort, r in summary.items():
        if cohort.startswith('_'):
            continue
        e = r['ensemble']
        a = r['per_fold_mean_std']
        lines += [
            f"### {r['center']}  (cohort={cohort}, n={e['n']}, cancer prevalence={e['prevalence']:.2%})",
            f"  5-fold ENSEMBLE:  AUC {e['auc']:.4f} | acc {e['acc']:.4f} | bal-acc {e['balanced_acc']:.4f} "
            f"| sens {e['sensitivity']:.4f} | spec {e['specificity']:.4f} | PPV {e['ppv']:.4f} | NPV {e['npv']:.4f} "
            f"| macroF1 {e['macro_f1']:.4f} | qkappa {e['quadratic_kappa']:.4f}",
            f"  confusion [[TN,FP],[FN,TP]] = {e['confusion_matrix_[[TN,FP],[FN,TP]]']}",
            f"  per-fold  AUC {a['auc']['mean']:.4f}+-{a['auc']['std']:.4f} | "
            f"acc {a['acc']['mean']:.4f}+-{a['acc']['std']:.4f} | "
            f"sens {a['sensitivity']['mean']:.4f}+-{a['sensitivity']['std']:.4f} | "
            f"spec {a['specificity']['mean']:.4f}+-{a['specificity']['std']:.4f}",
            "  by specimen type (ensemble):",
        ]
        for t, m in r['ensemble_by_type'].items():
            if 'auc' in m:
                lines.append(f"    {t:5s} n={m['n']:3d}  AUC {m['auc']:.4f} | sens {m['sensitivity']:.4f} | spec {m['specificity']:.4f}")
            else:
                lines.append(f"    {t:5s} n={m['n']:3d}  ({m['note']})")
        lines.append('')
    if '_pooled_all_external' in summary:
        p = summary['_pooled_all_external']['ensemble']
        lines += [f"### POOLED (301 + ynzl)  n={p['n']}",
                  f"  ensemble AUC {p['auc']:.4f} | acc {p['acc']:.4f} | bal-acc {p['balanced_acc']:.4f} "
                  f"| sens {p['sensitivity']:.4f} | spec {p['specificity']:.4f}", '']
    txt = '\n'.join(lines)
    with open(os.path.join(out, 'SUMMARY.txt'), 'w') as fh:
        fh.write(txt + '\n')
    print('\n' + txt)
    print(f'saved -> {out}/  (per_center_metrics.json, per_center_metrics.csv, slide_predictions.csv, SUMMARY.txt)')


if __name__ == '__main__':
    main()
