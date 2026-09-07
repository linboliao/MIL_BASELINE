"""Feasibility check: does the serial-section panel carry enough signal to be
worth building a study on?

Uses the PRIOR model's per-centre predictions that were already stored in the
spreadsheets, so this costs nothing. The question is not "what is each centre's
accuracy" (n=60 gives +/-5.5pp, hopelessly overlapping) but the paired one:
on identical tissue, how often does the SAME case flip between centres? Paired
comparisons are far better powered than comparing two accuracy estimates.
"""
import itertools
import os

import numpy as np
import pandas as pd

SER = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/serial_sections'

preds = pd.read_csv(f'{SER}/prior_model_predictions.csv', dtype={'case_id': str})
labels = pd.read_csv(f'{SER}/serial_case_labels.csv', dtype={'case_id': str})
lab = dict(zip(labels['case_id'], labels['label']))

# the per-centre columns of 结果汇总 (raw and stain-normalised variants)
sm = preds[preds['source'].str.startswith('结果汇总')].copy()
sm['normalised'] = sm['source'].str.contains('归一化')
sm['centre'] = sm['center']
sm['label'] = sm['case_id'].map(lab)
sm = sm.dropna(subset=['label'])
print(f'per-centre prediction rows usable: {len(sm)}')
print(sm.groupby(['centre', 'normalised']).size().to_string())
print()


def wilson_half(n, p=0.95, z=1.96):
    if n == 0:
        return float('nan')
    return z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n) * 100


for norm_flag in [False, True]:
    tag = 'stain-normalised' if norm_flag else 'raw'
    d = sm[sm['normalised'] == norm_flag]
    if d.empty:
        continue
    print(f'================ {tag} ================')
    wide = d.pivot_table(index='case_id', columns='centre', values='prediction')
    wide = wide.dropna(axis=1, thresh=30)          # keep centres with enough coverage
    wide = wide.dropna()                            # cases scored by every kept centre
    if wide.empty or wide.shape[1] < 2:
        print('  not enough overlapping coverage\n')
        continue
    y = pd.Series({c: lab[c] for c in wide.index})
    print(f'cases scored by all {wide.shape[1]} centres: {len(wide)}  centres: {list(wide.columns)}')
    print()

    print('per-centre accuracy (unpaired, with CI):')
    for c in wide.columns:
        acc = (wide[c] == y).mean()
        print(f'  {c:8s} acc={acc:.3f}  95% CI +/-{wilson_half(len(wide), acc):.1f} pp')
    print()

    print('PAIRED disagreement between centres on identical tissue:')
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        disagree = (wide[a] != wide[b]).sum()
        # McNemar counts
        b_fixes = ((wide[a] != y) & (wide[b] == y)).sum()
        a_fixes = ((wide[b] != y) & (wide[a] == y)).sum()
        rows.append({'pair': f'{a} vs {b}', 'n': len(wide), 'disagree': int(disagree),
                     'disagree_rate': round(disagree / len(wide), 3),
                     f'only_{a}_right': int(a_fixes), f'only_{b}_right': int(b_fixes)})
        print(f'  {a:8s} vs {b:8s}: {disagree:3d}/{len(wide)} disagree '
              f'({disagree / len(wide) * 100:4.1f}%)')
    pd.DataFrame(rows).to_csv(f'{SER}/panel_pairwise_{"norm" if norm_flag else "raw"}.csv',
                              index=False, encoding='utf-8-sig')
    print()

    flips = (wide.nunique(axis=1) > 1)
    print(f'cases where the prediction FLIPS across at least one centre pair: '
          f'{flips.sum()}/{len(wide)} ({flips.mean() * 100:.1f}%)')
    print(f'  of those, truly positive: {int(y[flips].sum())}, negative: {int((1 - y[flips]).sum())}')
    print()

    # Fleiss' kappa across centres
    k = wide.shape[1]
    n_pos = wide.sum(axis=1)
    n_neg = k - n_pos
    P_i = ((n_pos ** 2 + n_neg ** 2) - k) / (k * (k - 1))
    P_bar = P_i.mean()
    p_pos = n_pos.sum() / (len(wide) * k)
    P_e = p_pos ** 2 + (1 - p_pos) ** 2
    kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else float('nan')
    print(f"Fleiss' kappa across the {k} centres: {kappa:.3f}")
    print(f'mean per-case agreement P_bar={P_bar:.3f}, chance P_e={P_e:.3f}')
    print()
