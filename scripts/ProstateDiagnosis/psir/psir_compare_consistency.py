"""Compare CONCH-baseline vs PSIR-fold1 cross-center prediction consistency
on the exact same held-out slides (Panel A fold-1 held-out cases + Panel B).

Each variant's 5 fold-models' probabilities are averaged (ensemble) per
slide, then predictions are grouped by case and compared across centers.
"""
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DS = str(ROOT / "datasets/ProstateDiagnosis")
PSIR_DIR = f"{DS}/psir"
PANEL_EVAL = os.environ.get("PSIR_PANEL_EVAL_DIR", f"{PSIR_DIR}/panel_eval")


def load_ensemble_preds(prefix, n_folds, fold_dir_tmpl):
    frames = []
    for k in range(1, n_folds + 1):
        d = fold_dir_tmpl.format(k=k)
        f = None
        for root, _, files in os.walk(d):
            for fn in files:
                if fn == 'Infer_Result.csv':
                    f = os.path.join(root, fn)
        df = pd.read_csv(f)
        df['slide_id'] = df['slide_id'].astype(str)
        frames.append(pd.DataFrame({'slide_id': df['slide_id'], f'prob1_f{k}': df['prob_1'].values}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on='slide_id', how='inner')
    prob_cols = [c for c in merged.columns if c.startswith('prob1_')]
    merged['prob1_ens'] = merged[prob_cols].mean(axis=1)
    merged['pred_ens'] = (merged['prob1_ens'] >= 0.5).astype(int)
    return merged[['slide_id', 'prob1_ens', 'pred_ens']]


def wilson_half(n, p, z=1.96):
    if n == 0:
        return float('nan')
    denom = 1 + z * z / n
    return z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom * 100


def analyze(name, manifest, preds):
    df = manifest.merge(preds, on='slide_id', how='inner')
    print(f'\n================ {name}: {len(df)} slides matched ================')

    print('per-slide acc vs manifest label:', round((df['pred_ens'] == df['label']).mean(), 4))

    results = []
    for panel_tag, sub in df.groupby('panel'):
        wide = sub.pivot_table(index='case_id', columns='center', values='pred_ens', aggfunc='first')
        wide = wide.dropna(axis=1, thresh=max(2, int(0.5 * wide.shape[0])))
        wide_full = wide.dropna()
        if wide_full.empty or wide_full.shape[1] < 2:
            print(f'  panel {panel_tag}: not enough overlapping centers to compare')
            continue
        y = sub.groupby('case_id')['label'].first().reindex(wide_full.index)

        # pairwise disagreement across all center pairs, per case, averaged
        disagree_fracs = []
        for _, row in wide_full.iterrows():
            vals = row.values
            pairs = list(itertools.combinations(vals, 2))
            disagree_fracs.append(np.mean([a != b for a, b in pairs]))
        mean_disagree = np.mean(disagree_fracs)

        flips = (wide_full.nunique(axis=1) > 1)
        flip_rate = flips.mean()

        k = wide_full.shape[1]
        n_pos = wide_full.sum(axis=1)
        n_neg = k - n_pos
        P_i = ((n_pos ** 2 + n_neg ** 2) - k) / (k * (k - 1))
        P_bar = P_i.mean()
        p_pos = n_pos.sum() / (len(wide_full) * k)
        P_e = p_pos ** 2 + (1 - p_pos) ** 2
        kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else float('nan')

        n_cases = len(wide_full)
        print(f'  panel {panel_tag}: {n_cases} cases x {k} centers | '
              f'mean pairwise disagreement={mean_disagree:.3f} | '
              f'flip_rate={flip_rate:.3f} ({int(flips.sum())}/{n_cases}) | '
              f"Fleiss' kappa={kappa:.3f}")
        results.append({'name': name, 'panel': panel_tag, 'n_cases': n_cases, 'n_centers': k,
                         'mean_pairwise_disagreement': round(mean_disagree, 4),
                         'flip_rate': round(flip_rate, 4), 'fleiss_kappa': round(kappa, 4)})
    return pd.DataFrame(results)


manifest_a = pd.read_csv(f'{PANEL_EVAL}/manifest_conch_baseline.csv', dtype={'slide_id': str, 'case_id': str})
manifest_b = pd.read_csv(f'{PANEL_EVAL}/manifest_conch_psir_fold1.csv', dtype={'slide_id': str, 'case_id': str})

baseline_preds = load_ensemble_preds('conch_baseline', 5,
    f'{PANEL_EVAL}/infer_conch_baseline_fold{{k}}')
psir_preds = load_ensemble_preds('conch_psir_fold1', 5,
    f'{PANEL_EVAL}/infer_conch_psir_fold1_cvfold{{k}}')

r1 = analyze('CONCH baseline', manifest_a, baseline_preds)
r2 = analyze('PSIR fold1', manifest_b, psir_preds)

summary = pd.concat([r1, r2], ignore_index=True)
summary.to_csv(f'{PANEL_EVAL}/consistency_comparison.csv', index=False, encoding='utf-8-sig')
print('\n================ SUMMARY ================')
print(summary.to_string(index=False))
print(f'\nwritten -> {PANEL_EVAL}/consistency_comparison.csv')
