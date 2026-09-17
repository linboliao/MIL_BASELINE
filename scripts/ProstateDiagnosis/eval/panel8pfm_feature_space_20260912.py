#!/usr/bin/env python3
"""8-PFM SerialPanel A/B feature-space mechanism analysis.

Safety/design:
- Reads existing patch-level .pt features only; does not modify source data/features.
- Uses formal non-excluded panel metadata CSVs.
- Panel A domain = named center; Panel B domain = anonymous unit_id U1..U6.
- Deterministic approximate mean pooling from <=4096 patches/slide using 8 spatially
  distributed contiguous blocks with torch.load(..., mmap=True).
- Grouped CV prevents the same case appearing in train and test for domain classification.
- Label classification is performed at CASE level after averaging across centers.
- Pairwise MMD uses matched cases and within-case swap permutations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

REPO = Path('/NAS2/Data1/lbliao/Code-195/MIL_BASELINE')
FEAT_ROOT = Path('/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis')
META_ROOT = REPO / 'datasets/ProstateDiagnosis/serial_sections'
MODELS = ['conch', 'uni', 'uni2', 'virchow2', 'h-optimus-1', 'mstar', 'gigapath', 'gpfm']
PANELS = {
    'A': {'csv': 'panel_A_6center.csv', 'pool': 'SerialPanelA', 'domain_col': 'center'},
    'B': {'csv': 'panel_B_20case_anon.csv', 'pool': 'SerialPanelB', 'domain_col': 'unit_id'},
}
RANDOM_SEED = 42
MAX_PATCHES = 4096
N_BLOCKS = 8
MMD_PERM = 199


def safe_json(path: Path, obj):
    if path.exists():
        raise FileExistsError(path)
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write('\n')


def safe_csv(path: Path, df: pd.DataFrame):
    if path.exists():
        raise FileExistsError(path)
    with open(path, 'x', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def sha256(path: Path, block=4*1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def load_panel(panel: str) -> pd.DataFrame:
    cfg = PANELS[panel]
    d = pd.read_csv(META_ROOT / cfg['csv'], dtype={'case_id': str})
    d['slide_id'] = d['filename'].map(lambda x: Path(str(x)).stem)
    d['label'] = pd.to_numeric(d['label'], errors='coerce')
    d['domain'] = d[cfg['domain_col']].astype(str)
    if d['case_id'].isna().any() or d['label'].isna().any() or d['domain'].isna().any():
        raise ValueError(f'Panel {panel}: missing case/label/domain in formal metadata')
    # each case must have a single label
    if d.groupby('case_id')['label'].nunique().max() != 1:
        raise ValueError(f'Panel {panel}: inconsistent case labels')
    if d['slide_id'].duplicated().any():
        raise ValueError(f'Panel {panel}: duplicated slide_id')
    return d.reset_index(drop=True)


def block_mean_pt(path: Path, max_patches=MAX_PATCHES, n_blocks=N_BLOCKS):
    x = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'{path}: not Tensor')
    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f'{path}: bad shape {tuple(x.shape)}')
    n, dim = int(x.shape[0]), int(x.shape[1])
    take = min(n, int(max_patches))
    if take == n:
        sample = x.float()
    else:
        nb = min(int(n_blocks), take)
        base = take // nb
        rem = take % nb
        parts = []
        for j in range(nb):
            bs = base + (1 if j < rem else 0)
            if bs <= 0: continue
            # deterministic block anchors spanning the slide order
            if nb == 1:
                a = max(0, (n-bs)//2)
            else:
                a = int(round(j * max(0, n-bs) / (nb-1)))
            parts.append(x[a:a+bs].float())
        sample = torch.cat(parts, dim=0)
    if not torch.isfinite(sample).all().item():
        raise ValueError(f'{path}: non-finite values in sampled features')
    mu = sample.mean(0).numpy().astype(np.float32, copy=False)
    return mu, n, int(sample.shape[0]), dim, str(x.dtype)


def make_repr(X: np.ndarray, max_pc=50):
    Xs = StandardScaler().fit_transform(X)
    npc = min(max_pc, Xs.shape[0]-1, Xs.shape[1])
    if npc >= 2:
        return PCA(n_components=npc, random_state=RANDOM_SEED).fit_transform(Xs)
    return Xs


def cosine_pair_metrics(X, cases, domains):
    # normalized slide representations; distance = 1-cosine
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Z = X / np.clip(norm, 1e-12, None)
    D = 1.0 - np.clip(Z @ Z.T, -1.0, 1.0)
    same, diff = [], []
    pair_rows = []
    uniq_domains = sorted(set(domains))
    for a, b in combinations(uniq_domains, 2):
        ia = np.where(domains == a)[0]; ib = np.where(domains == b)[0]
        vals_same=[]; vals_diff=[]
        for i in ia:
            for j in ib:
                if cases[i] == cases[j]: vals_same.append(float(D[i,j]))
                else: vals_diff.append(float(D[i,j]))
        if vals_same:
            same.extend(vals_same); diff.extend(vals_diff)
            pair_rows.append({'domain_a':a,'domain_b':b,'n_matched_cases':len(vals_same),
                              'same_case_cosine_median':float(np.median(vals_same)),
                              'diff_case_cosine_median':float(np.median(vals_diff)),
                              'consistency_ratio':float(np.median(vals_same)/max(np.median(vals_diff),1e-12))})
    y = np.r_[np.ones(len(same),dtype=int), np.zeros(len(diff),dtype=int)]
    score = -np.r_[same,diff]
    auc = roc_auc_score(y, score) if len(set(y))==2 else np.nan
    return {
        'same_case_cosine_median': float(np.median(same)),
        'same_case_cosine_mean': float(np.mean(same)),
        'diff_case_crossdomain_cosine_median': float(np.median(diff)),
        'consistency_ratio': float(np.median(same)/max(np.median(diff),1e-12)),
        'same_case_retrieval_auc': float(auc),
        'n_same_pairs': len(same), 'n_diff_pairs': len(diff),
    }, pair_rows


def domain_classifier(X, y, groups):
    classes = np.array(sorted(pd.unique(y)))
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2 or len(classes) < 2:
        return {'domain_n_classes':len(classes), 'domain_cv_splits':n_splits}
    min_train = len(y) - math.ceil(len(y)/n_splits)
    npc = min(30, max(2, min_train-1), X.shape[1])
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=npc, random_state=RANDOM_SEED)),
        ('clf', LogisticRegression(max_iter=4000, class_weight='balanced', solver='lbfgs')),
    ])
    cv = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method='predict')
    prob = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method='predict_proba')
    # columns follow sorted string classes because all folds contain all domains in these paired panels
    out = {
        'domain_n_classes': len(classes), 'domain_cv_splits': n_splits,
        'domain_cv_accuracy': float(accuracy_score(y,pred)),
        'domain_cv_balanced_accuracy': float(balanced_accuracy_score(y,pred)),
        'domain_chance_balanced_accuracy': float(1.0/len(classes)),
    }
    try:
        out['domain_cv_macro_ovr_auc'] = float(roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=classes))
    except Exception as e:
        out['domain_cv_macro_ovr_auc'] = np.nan
        out['domain_auc_note'] = repr(e)
    return out


def label_classifier_caselevel(X, meta):
    # one vector per case, averaged across centers/units; avoids pseudo-replication
    rows=[]
    for case, g in meta.groupby('case_id', sort=True):
        idx=g.index.to_numpy()
        rows.append((str(case), int(g['label'].iloc[0]), X[idx].mean(0)))
    cases=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows],dtype=int); CX=np.stack([r[2] for r in rows])
    counts=np.bincount(y, minlength=2)
    min_class=int(counts[counts>0].min()) if (counts>0).any() else 0
    n_splits=min(5,min_class)
    out={'label_n_cases':len(y),'label_n_pos':int((y==1).sum()),'label_n_neg':int((y==0).sum()),'label_cv_splits':n_splits}
    if len(np.unique(y))<2 or n_splits<2:
        return out
    min_train=len(y)-math.ceil(len(y)/n_splits)
    npc=min(10,max(2,min_train-1),CX.shape[1])
    pipe=Pipeline([
        ('scale',StandardScaler()),
        ('pca',PCA(n_components=npc,random_state=RANDOM_SEED)),
        ('clf',LogisticRegression(max_iter=4000,class_weight='balanced',solver='lbfgs')),
    ])
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=RANDOM_SEED)
    prob=cross_val_predict(pipe,CX,y,cv=cv,method='predict_proba')[:,1]
    pred=(prob>=0.5).astype(int)
    out.update({'label_case_cv_auc':float(roc_auc_score(y,prob)),
                'label_case_cv_balanced_accuracy':float(balanced_accuracy_score(y,pred))})
    # descriptive label silhouette in a common case-level representation
    CR=make_repr(CX,max_pc=min(20,len(y)-1))
    if min(counts[counts>0])>=2:
        out['label_case_silhouette']=float(silhouette_score(CR,y))
    return out


def rbf_kernel(Z):
    d2=squareform(pdist(Z,metric='sqeuclidean'))
    nz=d2[d2>0]
    med=float(np.median(nz)) if len(nz) else 1.0
    gamma=1.0/max(2*med,1e-12)
    return np.exp(-gamma*d2),gamma


def mmd_from_K(K, ia, ib):
    return float(K[np.ix_(ia,ia)].mean()+K[np.ix_(ib,ib)].mean()-2*K[np.ix_(ia,ib)].mean())


def paired_mmd_metrics(R, meta, nperm=MMD_PERM):
    rng=np.random.default_rng(RANDOM_SEED)
    domains=sorted(meta['domain'].unique())
    rows=[]
    for a,b in combinations(domains,2):
        ga=meta[meta.domain==a].set_index('case_id'); gb=meta[meta.domain==b].set_index('case_id')
        common=sorted(set(ga.index)&set(gb.index))
        if len(common)<4: continue
        ia0=np.array([ga.loc[c].name for c in common])  # placeholder to satisfy linter
        # indices in original meta
        ia=np.array([meta.index[(meta.case_id==c)&(meta.domain==a)][0] for c in common],dtype=int)
        ib=np.array([meta.index[(meta.case_id==c)&(meta.domain==b)][0] for c in common],dtype=int)
        Z=np.vstack([R[ia],R[ib]])
        K,gamma=rbf_kernel(Z)
        n=len(common); A=np.arange(n); B=np.arange(n,2*n)
        stat=mmd_from_K(K,A,B); ge=0
        for _ in range(nperm):
            swap=rng.random(n)<0.5
            Ap=A.copy(); Bp=B.copy(); Ap[swap],Bp[swap]=Bp[swap].copy(),Ap[swap].copy()
            ge += mmd_from_K(K,Ap,Bp) >= stat-1e-15
        rows.append({'domain_a':a,'domain_b':b,'n_matched_cases':n,'paired_mmd2':stat,
                     'paired_mmd_perm_p':float((ge+1)/(nperm+1)),'rbf_gamma':gamma})
    vals=[r['paired_mmd2'] for r in rows]
    return {'paired_mmd2_mean':float(np.mean(vals)) if vals else np.nan,
            'paired_mmd2_median':float(np.median(vals)) if vals else np.nan,
            'paired_mmd2_max':float(np.max(vals)) if vals else np.nan,
            'mmd_pairs':len(vals),'mmd_permutations':nperm}, rows


def analyze_model(model: str, out_dir: str):
    out=Path(out_dir)
    summaries=[]; pairwise=[]; coverage=[]
    for panel in ['A','B']:
        meta=load_panel(panel)
        pool=PANELS[panel]['pool']
        pdir=FEAT_ROOT/pool/'feat_0_224'/'pt_files'/model
        feats=[]; dims=set(); dtypes=set(); patch_counts=[]; sample_counts=[]
        t0=time.time()
        for k,row in meta.iterrows():
            f=pdir/(row.slide_id+'.pt')
            if not f.exists(): raise FileNotFoundError(f)
            mu,n,ns,dim,dtype=block_mean_pt(f)
            feats.append(mu); dims.add(dim); dtypes.add(dtype); patch_counts.append(n); sample_counts.append(ns)
            if (k+1)%60==0:
                print(f'[{model}/{panel}] pooled {k+1}/{len(meta)}',flush=True)
        if len(dims)!=1: raise ValueError(f'{model}/{panel}: mixed dims {dims}')
        X=np.stack(feats)
        if not np.isfinite(X).all(): raise ValueError(f'{model}/{panel}: nonfinite pooled matrix')
        R=make_repr(X,50)
        cases=meta.case_id.astype(str).to_numpy(); domains=meta.domain.astype(str).to_numpy()
        cos,cos_rows=cosine_pair_metrics(X,cases,domains)
        dom=domain_classifier(X,domains,cases)
        lab=label_classifier_caselevel(X,meta)
        # descriptive silhouettes; case silhouette requires >=2 slides/case
        desc={}
        try: desc['domain_silhouette']=float(silhouette_score(R,domains))
        except Exception as e: desc['domain_silhouette_note']=repr(e)
        try: desc['case_silhouette']=float(silhouette_score(R,cases))
        except Exception as e: desc['case_silhouette_note']=repr(e)
        mmd,mmd_rows=paired_mmd_metrics(R,meta)
        summaries.append({'model':model,'panel':panel,'n_slides':len(meta),'n_cases':meta.case_id.nunique(),
                          'feature_dim':next(iter(dims)),'source_dtype':'|'.join(sorted(dtypes)),
                          'patch_median':float(np.median(patch_counts)),'sampled_patch_median':float(np.median(sample_counts)),
                          'pooling_seconds':time.time()-t0,**cos,**dom,**lab,**desc,**mmd})
        coverage.append({'model':model,'panel':panel,'n_slides':len(meta),'n_cases':meta.case_id.nunique(),
                         'n_domains':meta.domain.nunique(),'feature_dim':next(iter(dims)),
                         'missing_features':0,'min_patches':int(min(patch_counts)),'median_patches':float(np.median(patch_counts)),
                         'max_patches':int(max(patch_counts)),'max_sampled_patches':int(max(sample_counts))})
        for r in cos_rows: pairwise.append({'model':model,'panel':panel,'metric_family':'cosine',**r})
        for r in mmd_rows: pairwise.append({'model':model,'panel':panel,'metric_family':'paired_mmd',**r})
        npz=out/'pooled'/f'panel_{panel}_{model}.npz'
        if npz.exists(): raise FileExistsError(npz)
        np.savez(npz,X=X,case_id=cases,domain=domains,label=meta.label.to_numpy(dtype=int),
                 slide_id=meta.slide_id.astype(str).to_numpy(),patch_count=np.array(patch_counts),sample_count=np.array(sample_counts))
    return summaries,pairwise,coverage


def rank_panel(s: pd.DataFrame, panel: str):
    d=s[s.panel==panel].copy()
    # higher rank score is better; individual raw metrics remain primary outputs
    specs=[('consistency_ratio',True),('same_case_retrieval_auc',False),('domain_cv_balanced_accuracy',True),
           ('domain_silhouette',True),('paired_mmd2_mean',True),('case_silhouette',False)]
    ranks=[]
    for col,lower_better in specs:
        if col not in d or d[col].isna().all(): continue
        rc=col+'_rank'
        d[rc]=d[col].rank(method='average',ascending=lower_better)
        ranks.append(rc)
    d['domain_robustness_mean_rank']=d[ranks].mean(axis=1)
    return d.sort_values('domain_robustness_mean_rank').reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--models',nargs='+',default=MODELS)
    a=ap.parse_args()
    models=a.models
    if any(m not in MODELS for m in models): raise ValueError(models)
    out=Path(a.out)
    if out.exists(): raise FileExistsError(f'Output exists; refusing to overwrite: {out}')
    out.mkdir(parents=True,exist_ok=False); (out/'pooled').mkdir(exist_ok=False)
    meta={
        'created':time.strftime('%Y-%m-%d %H:%M:%S'), 'hostname':platform.node(),
        'repo':str(REPO), 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'models':models,'panels':PANELS,'max_patches_per_slide':MAX_PATCHES,'n_blocks':N_BLOCKS,
        'mmd_permutations':MMD_PERM,'random_seed':RANDOM_SEED,'workers':a.workers,
        'python':platform.python_version(),'torch':torch.__version__,
        'metadata_sha256':{k:sha256(META_ROOT/v['csv']) for k,v in PANELS.items()},
        'note':'Panel B domain labels are anonymous unit_id U1..U6 because several true center identities are unresolved.'
    }
    safe_json(out/'run_meta.json',meta)
    summaries=[]; pairwise=[]; coverage=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,len(models))) as ex:
        fut={ex.submit(analyze_model,m,str(out)):m for m in models}
        for f in as_completed(fut):
            m=fut[f]
            s,p,c=f.result(); summaries+=s; pairwise+=p; coverage+=c
            print(f'=== MODEL DONE {m} ===',flush=True)
    sdf=pd.DataFrame(summaries).sort_values(['panel','model']).reset_index(drop=True)
    pdf=pd.DataFrame(pairwise).sort_values(['panel','model','metric_family','domain_a','domain_b']).reset_index(drop=True)
    cdf=pd.DataFrame(coverage).sort_values(['panel','model']).reset_index(drop=True)
    safe_csv(out/'summary.csv',sdf); safe_csv(out/'pairwise_domain_metrics.csv',pdf); safe_csv(out/'coverage.csv',cdf)
    ra=rank_panel(sdf,'A'); rb=rank_panel(sdf,'B')
    safe_csv(out/'panel_A_ranking.csv',ra); safe_csv(out/'panel_B_ranking.csv',rb)

    L=['# 8-PFM Panel A/B feature-space mechanism analysis','',
       f"Run: `{out.name}`",f"Git: `{meta['git_head']}`",'',
       '## Design','',
       f'- Deterministic mean pooling: at most {MAX_PATCHES} patches/slide in {N_BLOCKS} distributed contiguous blocks.',
       '- Panel A: 60 matched cases × 6 named centers; Panel B: 19 valid cases, domains represented by anonymous U1–U6.',
       '- Domain classifier CV is grouped by case; label classifier is case-level after averaging across centers.',
       f'- Pairwise MMD is case-matched with {MMD_PERM} within-case swap permutations.','',
       '## Panel A ranking (exploratory composite; raw metrics are primary)','']
    show=['model','domain_robustness_mean_rank','consistency_ratio','same_case_retrieval_auc','domain_cv_balanced_accuracy','domain_silhouette','paired_mmd2_mean','case_silhouette','label_case_cv_auc']
    L.append(ra[[c for c in show if c in ra]].to_markdown(index=False,floatfmt='.4f'))
    L += ['', '## Panel B ranking (supportive; anonymous units, smaller n)','']
    L.append(rb[[c for c in show if c in rb]].to_markdown(index=False,floatfmt='.4f'))
    L += ['', '## Interpretation guide','',
          '- Better domain robustness: lower consistency_ratio, lower domain classifier balanced accuracy, lower domain silhouette, lower paired MMD; higher same-case retrieval AUC and case silhouette.',
          '- Higher label_case_cv_auc is desirable disease-label separability but is not included in the domain-robustness composite.',
          '- Panel A should be treated as the primary paired-domain experiment; Panel B is supportive because several unit identities are anonymous and n is smaller.','']
    report=out/'REPORT.md'
    if report.exists(): raise FileExistsError(report)
    with open(report,'x',encoding='utf-8') as f: f.write('\n'.join(L))
    print('DONE',out,flush=True)
    print(ra[['model','domain_robustness_mean_rank','consistency_ratio','domain_cv_balanced_accuracy','paired_mmd2_mean','label_case_cv_auc']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
