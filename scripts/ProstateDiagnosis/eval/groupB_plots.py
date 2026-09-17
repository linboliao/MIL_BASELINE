#!/usr/bin/env python
"""Render Group-B figures from the saved analysis outputs."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = sys.argv[1] if len(sys.argv) > 1 else sorted(
    'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local/' + d
    for d in os.listdir('result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
    if d.startswith('run_'))[-1]
G = os.path.join(RUN, 'groupB_analysis')
emb = pd.read_csv(os.path.join(G, 'embedding_coords.csv'))
mean2560 = np.load(os.path.join(G, 'pooled_mean2560.npy'))
meta = pd.read_csv(os.path.join(G, 'slide_meta.csv'))

COH_C = {'train_ref': '#9aa4b0', 'internal': '#3b7dd8', 'ext_ynzl': '#2ca25f', 'ext_301': '#d62728'}
COH_L = {'train_ref': 'training (ref)', 'internal': 'internal test', 'ext_ynzl': 'external ynzl', 'ext_301': 'external 301'}

# --- fig 1: t-SNE by cohort ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
for c in ['train_ref', 'internal', 'ext_ynzl', 'ext_301']:
    m = emb.cohort == c
    ax[0].scatter(emb.tsne_x[m], emb.tsne_y[m], s=14, c=COH_C[c], label=COH_L[c],
                  alpha=.75, edgecolors='none')
ax[0].legend(frameon=False, fontsize=10); ax[0].set_title('t-SNE of slide-level virchow2 features — by cohort')
ax[0].set_xticks([]); ax[0].set_yticks([])
TY = {'CNB': '#1f77b4', 'RP': '#ff7f0e', 'TURP': '#9467bd', 'UNK': '#cccccc', '?': '#cccccc'}
for t in emb.type.dropna().unique():
    m = emb.type == t
    ax[1].scatter(emb.tsne_x[m], emb.tsne_y[m], s=14, c=TY.get(t, '#999'), label=str(t), alpha=.7, edgecolors='none')
# outline 301
m = emb.cohort == 'ext_301'
ax[1].scatter(emb.tsne_x[m], emb.tsne_y[m], s=42, facecolors='none', edgecolors='#d62728', linewidths=.8, label='301 (outlined)')
ax[1].legend(frameon=False, fontsize=10); ax[1].set_title('same embedding — by specimen type (301 outlined red)')
ax[1].set_xticks([]); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(G, 'fig_tsne.png'), dpi=95, bbox_inches='tight'); plt.close()

# --- fig 2: per-dim SMD ---
ref = meta.cohort == 'train_ref'
mu, sd = mean2560[ref].mean(0), mean2560[ref].std(0) + 1e-8
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for c, col in [('ext_301', '#d62728'), ('ext_ynzl', '#2ca25f'), ('internal', '#3b7dd8')]:
    smd = (mean2560[meta.cohort == c].mean(0) - mu) / sd
    ax[0].hist(smd, bins=80, histtype='step', lw=1.8, color=col, label=COH_L[c])
ax[0].axvline(0, color='k', lw=.6); ax[0].set_xlabel('standardized mean diff vs training (per feature dim)')
ax[0].set_ylabel('# of dims'); ax[0].legend(frameon=False); ax[0].set_title('Per-dimension feature shift (2560 dims)')
labels = ['internal', 'ext_ynzl', 'ext_301']
vals = []
for c in labels:
    smd = np.abs((mean2560[meta.cohort == c].mean(0) - mu) / sd)
    vals.append([(smd > 1).mean() * 100, (smd > 2).mean() * 100])
vals = np.array(vals)
x = np.arange(3)
ax[1].bar(x - .17, vals[:, 0], .34, label='|SMD| > 1', color='#d62728')
ax[1].bar(x + .17, vals[:, 1], .34, label='|SMD| > 2', color='#7a1416')
ax[1].set_xticks(x); ax[1].set_xticklabels([COH_L[c] for c in labels])
ax[1].set_ylabel('% of 2560 feature dims'); ax[1].legend(frameon=False)
ax[1].set_title('Fraction of feature dims strongly shifted vs training')
plt.tight_layout(); plt.savefig(os.path.join(G, 'fig_smd.png'), dpi=95, bbox_inches='tight'); plt.close()

# --- fig 3: score vs feature norm, 301 benign RP/TURP ---
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
d = meta[meta.cohort.isin(['internal', 'ext_301'])].copy()
for c, col, lab in [('internal', '#3b7dd8', 'internal'), ('ext_301', '#d62728', '301')]:
    g = d[(d.cohort == c) & (d.label == 0) & d.type.isin(['RP', 'TURP'])]
    ax[0].scatter(g.feat_norm, g.prob, s=22, c=col, alpha=.7, label=f'{lab} benign RP/TURP (n={len(g)})')
ax[0].axhline(.5, color='k', ls='--', lw=.7); ax[0].set_xlabel('mean patch feature L2-norm')
ax[0].set_ylabel('cancer probability (ensemble)'); ax[0].legend(frameon=False)
ax[0].set_title('Benign RP/TURP: feature magnitude vs cancer score')
for c, col, lab in [('internal', '#3b7dd8', 'internal'), ('ext_301', '#d62728', '301')]:
    g = d[(d.cohort == c) & (d.type == 'TURP')]
    ax[1].scatter(g.n_patch, g.prob, s=22, c=col, alpha=.7, label=f'{lab} TURP (n={len(g)})')
ax[1].axhline(.5, color='k', ls='--', lw=.7); ax[1].set_xlabel('# patches (slide size)')
ax[1].set_ylabel('cancer probability'); ax[1].legend(frameon=False)
ax[1].set_title('TURP: slide size vs cancer score')
plt.tight_layout(); plt.savefig(os.path.join(G, 'fig_score_drivers.png'), dpi=95, bbox_inches='tight'); plt.close()

print('wrote fig_tsne.png, fig_smd.png, fig_score_drivers.png ->', G)
