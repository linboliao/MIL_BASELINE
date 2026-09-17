#!/usr/bin/env python3
"""Finalize full 60-case SerialPanelA registration-controlled analysis.

Reads only the raw full-run outputs plus existing 8-PFM .pt features.
No feature extraction, no source-data modification, no silent case exclusion.
Produces a separate final directory with standardized QC, matched-patch,
matched-slide, pathology-preservation, whole-slide comparison, and LOCO linkage.
"""
from __future__ import annotations

import argparse
import math
import os
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path('/NAS2/Data1/lbliao/Code-195/MIL_BASELINE')
PT_ROOT = Path('/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis/SerialPanelA/feat_0_224/pt_files')
WHOLE_SUMMARY = REPO / 'result/ProstateDiagnosis/DataAnalysis/panel8pfm_feature_space_20260912_172024/summary.csv'
MODELS = ['conch','uni','uni2','virchow2','h-optimus-1','mstar','gigapath','gpfm']
RANDOM_SEED = 42

LOCO = pd.DataFrame([
    ['h-optimus-1',0.869,0.840,0.949,0.978,0.949,0.958,'current consolidated matrix; GigaPath has separate reproducibility caveat'],
    ['mstar',      0.858,0.752,0.921,0.968,0.941,0.958,'current consolidated matrix'],
    ['gigapath',   0.823,0.908,0.945,0.958,0.943,0.857,'195 self-consistent matrix; known cross-environment best-epoch variance'],
    ['gpfm',       0.818,0.663,0.927,0.961,0.934,0.966,'current consolidated matrix'],
    ['uni2',       0.811,0.660,0.915,0.969,0.937,0.798,'current consolidated matrix'],
    ['virchow2',   0.781,0.571,0.912,0.982,0.967,0.983,'current consolidated matrix'],
    ['uni',        0.747,0.507,0.898,0.968,0.928,0.924,'current consolidated matrix'],
    ['conch',      0.653,0.316,0.842,0.937,0.932,0.983,'current consolidated matrix'],
], columns=['model','loco_worst_fold_bacc','loco_leave_rp_spec','loco_type_mean_bacc','loco_internal_mean_auc','loco_fivesite_mean_bacc','loco_ext301_spec','loco_note'])


def safe_csv(path: Path, df: pd.DataFrame):
    with open(path, 'x', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)


def safe_text(path: Path, text: str):
    with open(path, 'x', encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')


def deterministic_pos(n, max_n):
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_n, dtype=np.int64)


def load_rows(model, slide_id, idx):
    t = torch.load(str(PT_ROOT / model / f'{slide_id}.pt'), map_location='cpu', mmap=True, weights_only=True)
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    sidx = idx[order]
    v = t[torch.from_numpy(sidx)].float().numpy()
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    v = v[inv]
    if not np.isfinite(v).all():
        raise RuntimeError(f'nonfinite {model}/{slide_id}')
    return v


def parse_match_file(path, max_locations):
    df = pd.read_csv(path)
    pos = deterministic_pos(len(df), max_locations)
    df = df.iloc[pos].reset_index(drop=True)
    rows = []
    for c in df.columns:
        if c == 'target_idx':
            continue
        _, center, slide_id = c.split('__', 2)
        rows.append((center, slide_id, df[c].to_numpy(np.int64)))
    return rows


def make_repr(X, max_pc=50):
    Xs = StandardScaler().fit_transform(X)
    npc = min(max_pc, Xs.shape[0] - 1, Xs.shape[1])
    return PCA(n_components=npc, random_state=RANDOM_SEED).fit_transform(Xs) if npc >= 2 else Xs


def cosine_pair_metrics(X, cases, domains):
    Z = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    D = 1 - np.clip(Z @ Z.T, -1, 1)
    same, diff = [], []
    for a, b in combinations(sorted(set(domains)), 2):
        ia = np.where(domains == a)[0]
        ib = np.where(domains == b)[0]
        for i in ia:
            for j in ib:
                (same if cases[i] == cases[j] else diff).append(float(D[i, j]))
    y = np.r_[np.ones(len(same), int), np.zeros(len(diff), int)]
    score = -np.r_[same, diff]
    return {
        'same_case_cosine_median': float(np.median(same)),
        'same_case_cosine_mean': float(np.mean(same)),
        'diff_case_crossdomain_cosine_median': float(np.median(diff)),
        'consistency_ratio': float(np.median(same) / max(np.median(diff), 1e-12)),
        'same_case_retrieval_auc': float(roc_auc_score(y, score)),
        'n_same_pairs': len(same), 'n_diff_pairs': len(diff),
    }


def domain_classifier(X, y, groups):
    classes = np.array(sorted(pd.unique(y)))
    n_splits = min(5, len(np.unique(groups)))
    min_train = len(y) - math.ceil(len(y) / n_splits)
    npc = min(30, max(2, min_train - 1), X.shape[1])
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=npc, random_state=RANDOM_SEED)),
        ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', solver='lbfgs')),
    ])
    cv = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method='predict')
    prob = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method='predict_proba')
    out = {
        'domain_n_classes': len(classes), 'domain_cv_splits': n_splits,
        'domain_cv_accuracy': float(accuracy_score(y, pred)),
        'domain_cv_balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'domain_chance_balanced_accuracy': float(1 / len(classes)),
    }
    try:
        out['domain_cv_macro_ovr_auc'] = float(roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=classes))
    except Exception:
        out['domain_cv_macro_ovr_auc'] = np.nan
    return out


def label_classifier_caselevel(X, cases, labels):
    meta = pd.DataFrame({'case_id': cases, 'label': labels})
    rows = []
    for case, g in meta.groupby('case_id', sort=True):
        idx = g.index.to_numpy()
        rows.append((str(case), int(g['label'].iloc[0]), X[idx].mean(0)))
    y = np.asarray([r[1] for r in rows], dtype=int)
    CX = np.stack([r[2] for r in rows])
    counts = np.bincount(y, minlength=2)
    min_class = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    n_splits = min(5, min_class)
    out = {'label_n_cases': len(y), 'label_n_pos': int((y == 1).sum()), 'label_n_neg': int((y == 0).sum()), 'label_cv_splits': n_splits}
    if len(np.unique(y)) < 2 or n_splits < 2:
        return out
    min_train = len(y) - math.ceil(len(y) / n_splits)
    npc = min(10, max(2, min_train - 1), CX.shape[1])
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=npc, random_state=RANDOM_SEED)),
        ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', solver='lbfgs')),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    prob = cross_val_predict(pipe, CX, y, cv=cv, method='predict_proba')[:, 1]
    pred = (prob >= 0.5).astype(int)
    out['label_case_cv_auc'] = float(roc_auc_score(y, prob))
    out['label_case_cv_balanced_accuracy'] = float(balanced_accuracy_score(y, pred))
    CR = make_repr(CX, max_pc=min(20, len(y) - 1))
    if int(counts[counts > 0].min()) >= 2:
        out['label_case_silhouette'] = float(silhouette_score(CR, y))
    return out


def rbf_kernel(Z):
    d2 = squareform(pdist(Z, metric='sqeuclidean'))
    nz = d2[d2 > 0]
    med = float(np.median(nz)) if len(nz) else 1.0
    gamma = 1 / max(2 * med, 1e-12)
    return np.exp(-gamma * d2)


def mmd_from_K(K, ia, ib):
    return float(K[np.ix_(ia, ia)].mean() + K[np.ix_(ib, ib)].mean() - 2 * K[np.ix_(ia, ib)].mean())


def paired_mmd(R, cases, domains):
    vals = []
    for a, b in combinations(sorted(set(domains)), 2):
        ia_all = np.where(domains == a)[0]
        ib_all = np.where(domains == b)[0]
        ma = {cases[i]: i for i in ia_all}
        mb = {cases[i]: i for i in ib_all}
        common = sorted(set(ma) & set(mb))
        if len(common) < 4:
            continue
        ia = np.array([ma[c] for c in common])
        ib = np.array([mb[c] for c in common])
        Z = np.vstack([R[ia], R[ib]])
        K = rbf_kernel(Z)
        n = len(common)
        vals.append(mmd_from_K(K, np.arange(n), np.arange(n, 2 * n)))
    return {
        'paired_mmd2_mean': float(np.mean(vals)) if vals else np.nan,
        'paired_mmd2_median': float(np.median(vals)) if vals else np.nan,
        'paired_mmd2_max': float(np.max(vals)) if vals else np.nan,
        'mmd_pairs': len(vals),
    }


def review_reason_case(row, pair_flag_cases):
    reasons = []
    if float(row['min_pair_dice']) < 0.75:
        reasons.append('min_pair_dice_lt_0.75')
    if float(row['common_sixway_fraction_of_target']) < 0.20:
        reasons.append('common_coverage_lt_0.20')
    if int(row['common_sixway_locations']) < 1000:
        reasons.append('common_locations_lt_1000')
    if str(row['case_id']) in pair_flag_cases and 'min_pair_dice_lt_0.75' not in reasons:
        reasons.append('pair_registration_review')
    return ';'.join(reasons)


def pareto_flags(domain_signal, pathology):
    out = []
    for i in range(len(domain_signal)):
        dominated = False
        for j in range(len(domain_signal)):
            if i == j:
                continue
            if domain_signal[j] <= domain_signal[i] and pathology[j] >= pathology[i] and (domain_signal[j] < domain_signal[i] or pathology[j] > pathology[i]):
                dominated = True
                break
        out.append(not dominated)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_out', required=True)
    ap.add_argument('--final_out', required=True)
    ap.add_argument('--max_locations_per_case', type=int, default=512)
    a = ap.parse_args()
    raw = Path(a.raw_out)
    final = Path(a.final_out)
    if not raw.is_absolute(): raw = REPO / raw
    if not final.is_absolute(): final = REPO / final
    if not raw.exists(): raise FileNotFoundError(raw)
    final.mkdir(parents=True, exist_ok=False)
    os.symlink(str(raw / 'qc'), str(final / 'qc'), target_is_directory=True)
    safe_text(final / 'source_run.txt', f'raw_registration_run={raw}\nfeature_reextraction=False\nmax_locations_per_case={a.max_locations_per_case}')

    pair = pd.read_csv(raw / 'registration_pair_qc.csv', dtype={'case_id': str})
    pair['review_required'] = pair['final_dice'].astype(float) < 0.75
    pair['review_reason'] = np.where(pair['review_required'], 'final_dice_lt_0.75', '')
    pair['ecc_fallback'] = pair['method'].astype(str) != 'coarse+ecc_affine'
    safe_csv(final / 'registration_qc.csv', pair)

    case = pd.read_csv(raw / 'case_qc.csv', dtype={'case_id': str})
    pair_flag_cases = set(pair.loc[pair['review_required'], 'case_id'].astype(str))
    case['review_reason'] = case.apply(lambda r: review_reason_case(r, pair_flag_cases), axis=1)
    case['review_required'] = case['review_reason'].astype(str).str.len() > 0
    case['excluded_from_analysis'] = False
    safe_csv(final / 'case_qc.csv', case)

    patch = pd.read_csv(raw / 'model_metrics.csv')
    safe_csv(final / 'matched_patch_metrics.csv', patch)

    case_ids = case['case_id'].astype(str).tolist()
    label_map = {str(r.case_id): int(float(r.label)) for r in case.itertuples()}
    slide_rows = []
    for model in MODELS:
        print('[matched-slide-full]', model, flush=True)
        X, domains, groups, labels = [], [], [], []
        for cid in case_ids:
            rows = parse_match_file(raw / 'matches' / f'{cid}.csv', a.max_locations_per_case)
            if len(rows) != 6:
                raise RuntimeError(f'{cid}: expected 6 mapped centers, got {len(rows)}')
            for center, slide_id, idx in rows:
                f = load_rows(model, slide_id, idx)
                X.append(f.mean(axis=0))
                domains.append(center)
                groups.append(cid)
                labels.append(label_map[cid])
        X = np.stack(X)
        domains = np.asarray(domains)
        groups = np.asarray(groups)
        labels = np.asarray(labels, dtype=int)
        R = make_repr(X, 50)
        row = {'model': model, 'n_cases': len(case_ids), 'n_slides': len(X), 'matched_locations_per_case_cap': a.max_locations_per_case}
        row.update(cosine_pair_metrics(X, groups, domains))
        row.update(domain_classifier(X, domains, groups))
        row.update(paired_mmd(R, groups, domains))
        row.update(label_classifier_caselevel(X, groups, labels))
        row['domain_silhouette'] = float(silhouette_score(R, domains))
        row['case_silhouette'] = float(silhouette_score(R, groups))
        slide_rows.append(row)
    slide = pd.DataFrame(slide_rows).sort_values(['consistency_ratio','domain_cv_balanced_accuracy'])
    safe_csv(final / 'matched_slide_metrics.csv', slide)

    whole = pd.read_csv(WHOLE_SUMMARY)
    whole = whole[whole['panel'].astype(str) == 'A'].copy()
    metrics = [
        'consistency_ratio','same_case_retrieval_auc','domain_cv_balanced_accuracy','domain_cv_macro_ovr_auc',
        'domain_silhouette','case_silhouette','paired_mmd2_mean','label_case_cv_auc',
        'label_case_cv_balanced_accuracy','label_case_silhouette'
    ]
    cmp = whole[['model'] + metrics].merge(slide[['model'] + metrics], on='model', suffixes=('_whole_slide','_matched_region'))
    for c in metrics:
        cmp[c + '_delta_matched_minus_whole'] = cmp[c + '_matched_region'] - cmp[c + '_whole_slide']
    safe_csv(final / 'matched_vs_whole_slide.csv', cmp)

    psel = patch[['model','same_location_cosine_mean','same_location_cosine_median','different_location_cosine_mean','different_location_cosine_median','same_over_different_ratio','center_prototype_loo_bacc']].copy()
    psel = psel.rename(columns={
        'same_location_cosine_mean':'matched_patch_same_location_cosine_mean',
        'same_location_cosine_median':'matched_patch_same_location_cosine_median',
        'different_location_cosine_mean':'matched_patch_different_location_cosine_mean',
        'different_location_cosine_median':'matched_patch_different_location_cosine_median',
        'same_over_different_ratio':'matched_patch_same_over_different_ratio',
        'center_prototype_loo_bacc':'matched_patch_center_prototype_bacc',
    })
    summary = psel.merge(cmp, on='model').merge(LOCO, on='model', how='left')
    summary['morphology_effect_consistency_abs'] = summary['consistency_ratio_whole_slide'] - summary['consistency_ratio_matched_region']
    summary['morphology_effect_consistency_relative'] = summary['morphology_effect_consistency_abs'] / summary['consistency_ratio_whole_slide'].clip(lower=1e-12)
    summary['morphology_effect_center_bacc_abs'] = summary['domain_cv_balanced_accuracy_whole_slide'] - summary['domain_cv_balanced_accuracy_matched_region']
    summary['pathology_auc_change_matched_minus_whole'] = summary['label_case_cv_auc_matched_region'] - summary['label_case_cv_auc_whole_slide']
    summary['matched_center_signal_above_chance'] = summary['domain_cv_balanced_accuracy_matched_region'] - (1/6)
    safe_csv(final / 'pfm_summary.csv', summary)

    rank = summary[['model','matched_patch_same_over_different_ratio','consistency_ratio_matched_region','domain_cv_balanced_accuracy_matched_region','label_case_cv_auc_matched_region','loco_worst_fold_bacc']].copy()
    rank['rank_anatomical_patch_consistency'] = rank['matched_patch_same_over_different_ratio'].rank(method='min', ascending=True)
    rank['rank_matched_region_consistency'] = rank['consistency_ratio_matched_region'].rank(method='min', ascending=True)
    rank['rank_low_center_signal'] = rank['domain_cv_balanced_accuracy_matched_region'].rank(method='min', ascending=True)
    rank['rank_pathology_preservation'] = rank['label_case_cv_auc_matched_region'].rank(method='min', ascending=False)
    rank['rank_loco_worstcase'] = rank['loco_worst_fold_bacc'].rank(method='min', ascending=False)
    rank['pareto_low_domain_high_pathology'] = pareto_flags(rank['domain_cv_balanced_accuracy_matched_region'].to_numpy(), rank['label_case_cv_auc_matched_region'].to_numpy())
    rank = rank.sort_values(['rank_low_center_signal','rank_pathology_preservation','rank_anatomical_patch_consistency'])
    safe_csv(final / 'ranking.csv', rank)

    rho_cons = spearmanr(summary['consistency_ratio_whole_slide'], summary['consistency_ratio_matched_region']).statistic
    rho_center = spearmanr(summary['domain_cv_balanced_accuracy_whole_slide'], summary['domain_cv_balanced_accuracy_matched_region']).statistic
    rho_loco_cons = spearmanr(-summary['consistency_ratio_matched_region'], summary['loco_worst_fold_bacc']).statistic
    rho_loco_domain = spearmanr(-summary['domain_cv_balanced_accuracy_matched_region'], summary['loco_worst_fold_bacc']).statistic
    rho_loco_label = spearmanr(summary['label_case_cv_auc_matched_region'], summary['loco_worst_fold_bacc']).statistic

    domain_sensitive = summary.sort_values('domain_cv_balanced_accuracy_matched_region', ascending=False)[['model','domain_cv_balanced_accuracy_matched_region','consistency_ratio_matched_region']].head(4)
    domain_low = summary.sort_values('domain_cv_balanced_accuracy_matched_region', ascending=True)[['model','domain_cv_balanced_accuracy_matched_region','label_case_cv_auc_matched_region']].head(4)
    pathology = summary.sort_values('label_case_cv_auc_matched_region', ascending=False)[['model','label_case_cv_auc_matched_region','domain_cv_balanced_accuracy_matched_region']]
    pareto = rank[rank['pareto_low_domain_high_pathology']][['model','domain_cv_balanced_accuracy_matched_region','label_case_cv_auc_matched_region']]

    lines = []
    lines.append('# Panel A full 60-case registration-controlled analysis')
    lines.append('')
    lines.append('This report separates **domain sensitivity**, **anatomical/content consistency**, **pathology preservation**, and **downstream LOCO robustness**. It does not treat a simple composite mean as a calibrated domain-robustness score.')
    lines.append('')
    lines.append('## Registration / matching QC')
    lines.append(f'- Cases: {len(case)}; source→target registrations: {len(pair)}; QC overlays: 300.')
    lines.append(f'- Pair Dice mean/median/min: {pair.final_dice.mean():.3f} / {pair.final_dice.median():.3f} / {pair.final_dice.min():.3f}.')
    lines.append(f'- Median common six-center locations: {case.common_sixway_locations.median():.0f}; median target-grid coverage: {case.common_sixway_fraction_of_target.median():.1%}.')
    lines.append(f'- review_required: {int(case.review_required.sum())}/{len(case)} cases. No case was silently excluded.')
    lines.append('- Review rules: pair Dice < 0.75, case common coverage < 20%, or common matched locations < 1000. Low coverage is flagged for review and is not equated with registration failure.')
    lines.append('')
    lines.append('## Morphology / scan-position confounding')
    lines.append(f'- Mean consistency ratio: whole-slide {summary.consistency_ratio_whole_slide.mean():.3f} → matched-region {summary.consistency_ratio_matched_region.mean():.3f}; mean absolute reduction {summary.morphology_effect_consistency_abs.mean():.3f} ({summary.morphology_effect_consistency_relative.mean():.1%} relative).')
    lines.append(f'- Mean center-classifier bACC: whole-slide {summary.domain_cv_balanced_accuracy_whole_slide.mean():.3f} → matched-region {summary.domain_cv_balanced_accuracy_matched_region.mean():.3f}; mean absolute reduction {summary.morphology_effect_center_bacc_abs.mean():.3f}. Chance = 0.167.')
    lines.append(f'- Whole-slide vs matched-region rank stability: Spearman rho={rho_cons:.3f} for consistency ratio and rho={rho_center:.3f} for center bACC.')
    lines.append('- Interpretation: morphology/coverage mismatch inflates absolute cross-center separation, while residual center signal after anatomy control estimates PFM-specific stain/domain sensitivity more directly.')
    lines.append('')
    lines.append('## Domain sensitivity after anatomy control')
    lines.append(domain_sensitive.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('Lowest residual center signal:')
    lines.append(domain_low.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('## Pathology preservation')
    lines.append('Case-level label CV uses one case representation averaged across its six centers, followed by the same StandardScaler→PCA→balanced LogisticRegression design with stratified 5-fold CV.')
    lines.append(pathology.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('Pareto set for lower matched-region center signal and higher pathology AUC:')
    lines.append(pareto.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('## Relationship with downstream LOCO robustness')
    lines.append(f'- Spearman between LOCO worst-fold bACC and lower matched-region consistency ratio: rho={rho_loco_cons:.3f}.')
    lines.append(f'- Spearman between LOCO worst-fold bACC and lower matched-region center bACC: rho={rho_loco_domain:.3f}.')
    lines.append(f'- Spearman between LOCO worst-fold bACC and matched-region pathology AUC: rho={rho_loco_label:.3f}.')
    lines.append('- LOCO values are linked as a separate downstream generalization axis; GigaPath has a known cross-environment best-epoch reproducibility caveat and should not be over-interpreted from a single worst-fold value.')
    lines.append('')
    lines.append('## PFM summary')
    show = ['model','matched_patch_same_over_different_ratio','consistency_ratio_matched_region','domain_cv_balanced_accuracy_matched_region','label_case_cv_auc_matched_region','loco_worst_fold_bacc']
    lines.append(summary[show].sort_values('consistency_ratio_matched_region').to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('## Output files')
    lines.append('- registration_qc.csv')
    lines.append('- case_qc.csv')
    lines.append('- matched_patch_metrics.csv')
    lines.append('- matched_slide_metrics.csv')
    lines.append('- matched_vs_whole_slide.csv')
    lines.append('- pfm_summary.csv')
    lines.append('- ranking.csv')
    lines.append('- qc/ (symlink to the raw run QC overlays)')
    safe_text(final / 'REPORT.md', '\n'.join(lines))
    print('DONE', final, flush=True)


if __name__ == '__main__':
    main()
