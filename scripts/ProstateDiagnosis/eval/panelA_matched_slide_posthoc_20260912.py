#!/usr/bin/env python3
"""Recompute the original slide-level feature-space metrics using only registration-matched Panel A patches.

This is the apples-to-apples companion to panel8pfm_feature_space_20260912.py:
- same StandardScaler -> PCA -> grouped LogisticRegression center classifier;
- same slide-level cosine consistency metrics;
- same representation-level silhouettes and paired MMD;
- but each slide embedding is the mean of the same six-way matched tissue locations.
No features are re-extracted.
"""
from __future__ import annotations
import argparse, math
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO=Path("/NAS2/Data1/lbliao/Code-195/MIL_BASELINE")
PT_ROOT=Path("/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis/SerialPanelA/feat_0_224/pt_files")
OLD_SUMMARY=REPO/"result/ProstateDiagnosis/DataAnalysis/panel8pfm_feature_space_20260912_172024/summary.csv"
MODELS=["conch","uni","uni2","virchow2","h-optimus-1","mstar","gigapath","gpfm"]
RANDOM_SEED=42
MMD_PERM=199

def deterministic_pos(n,max_n):
    if n<=max_n: return np.arange(n,dtype=np.int64)
    return np.linspace(0,n-1,max_n,dtype=np.int64)

def load_rows(model, slide_id, idx):
    t=torch.load(str(PT_ROOT/model/(slide_id+".pt")),map_location="cpu",mmap=True,weights_only=True)
    idx=np.asarray(idx,dtype=np.int64); order=np.argsort(idx); sidx=idx[order]
    v=t[torch.from_numpy(sidx)].float().numpy(); inv=np.empty_like(order); inv[order]=np.arange(len(order))
    v=v[inv]
    if not np.isfinite(v).all(): raise RuntimeError(f"nonfinite {model}/{slide_id}")
    return v

def make_repr(X,max_pc=50):
    Xs=StandardScaler().fit_transform(X); npc=min(max_pc,Xs.shape[0]-1,Xs.shape[1])
    return PCA(n_components=npc,random_state=RANDOM_SEED).fit_transform(Xs) if npc>=2 else Xs

def cosine_pair_metrics(X,cases,domains):
    Z=X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-12,None); D=1-np.clip(Z@Z.T,-1,1)
    same=[]; diff=[]
    for a,b in combinations(sorted(set(domains)),2):
        ia=np.where(domains==a)[0]; ib=np.where(domains==b)[0]
        for i in ia:
            for j in ib:
                (same if cases[i]==cases[j] else diff).append(float(D[i,j]))
    y=np.r_[np.ones(len(same),int),np.zeros(len(diff),int)]; score=-np.r_[same,diff]
    return {"same_case_cosine_median":float(np.median(same)),"same_case_cosine_mean":float(np.mean(same)),
            "diff_case_crossdomain_cosine_median":float(np.median(diff)),
            "consistency_ratio":float(np.median(same)/max(np.median(diff),1e-12)),
            "same_case_retrieval_auc":float(roc_auc_score(y,score)),"n_same_pairs":len(same),"n_diff_pairs":len(diff)}

def domain_classifier(X,y,groups):
    classes=np.array(sorted(pd.unique(y))); n_splits=min(5,len(np.unique(groups)))
    min_train=len(y)-math.ceil(len(y)/n_splits); npc=min(30,max(2,min_train-1),X.shape[1])
    pipe=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=npc,random_state=RANDOM_SEED)),
                   ("clf",LogisticRegression(max_iter=4000,class_weight="balanced",solver="lbfgs"))])
    cv=GroupKFold(n_splits=n_splits)
    pred=cross_val_predict(pipe,X,y,groups=groups,cv=cv,method="predict")
    prob=cross_val_predict(pipe,X,y,groups=groups,cv=cv,method="predict_proba")
    out={"domain_cv_accuracy":float(accuracy_score(y,pred)),"domain_cv_balanced_accuracy":float(balanced_accuracy_score(y,pred)),
         "domain_chance_balanced_accuracy":float(1/len(classes)),"domain_cv_splits":n_splits}
    try: out["domain_cv_macro_ovr_auc"]=float(roc_auc_score(y,prob,multi_class="ovr",average="macro",labels=classes))
    except Exception: out["domain_cv_macro_ovr_auc"]=np.nan
    return out

def rbf_kernel(Z):
    d2=squareform(pdist(Z,metric="sqeuclidean")); nz=d2[d2>0]; med=float(np.median(nz)) if len(nz) else 1.0
    gamma=1/max(2*med,1e-12); return np.exp(-gamma*d2),gamma

def mmd_from_K(K,ia,ib):
    return float(K[np.ix_(ia,ia)].mean()+K[np.ix_(ib,ib)].mean()-2*K[np.ix_(ia,ib)].mean())

def paired_mmd(R,cases,domains,nperm=MMD_PERM):
    rng=np.random.default_rng(RANDOM_SEED); vals=[]
    for a,b in combinations(sorted(set(domains)),2):
        ia_all=np.where(domains==a)[0]; ib_all=np.where(domains==b)[0]
        ma={cases[i]:i for i in ia_all}; mb={cases[i]:i for i in ib_all}; common=sorted(set(ma)&set(mb))
        if len(common)<4: continue
        ia=np.array([ma[c] for c in common]); ib=np.array([mb[c] for c in common]); Z=np.vstack([R[ia],R[ib]])
        K,_=rbf_kernel(Z); n=len(common); A=np.arange(n); B=np.arange(n,2*n); stat=mmd_from_K(K,A,B); vals.append(stat)
    return {"paired_mmd2_mean":float(np.mean(vals)) if vals else np.nan,"paired_mmd2_median":float(np.median(vals)) if vals else np.nan,
            "paired_mmd2_max":float(np.max(vals)) if vals else np.nan,"mmd_pairs":len(vals)}

def parse_match_file(path,max_locations):
    df=pd.read_csv(path); pos=deterministic_pos(len(df),max_locations); df=df.iloc[pos].reset_index(drop=True)
    rows=[]
    for c in df.columns:
        if c=="target_idx": continue
        _,center,slide_id=c.split("__",2); rows.append((center,slide_id,df[c].to_numpy(np.int64)))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pilot_out",required=True); ap.add_argument("--max_locations_per_case",type=int,default=256); a=ap.parse_args()
    out=Path(a.pilot_out); out=out if out.is_absolute() else REPO/out
    case_ids=pd.read_csv(out/"case_qc.csv").case_id.astype(str).tolist()
    summaries=[]
    for model in MODELS:
        print("[matched-slide]",model,flush=True); X=[]; domains=[]; groups=[]
        for case in case_ids:
            rows=parse_match_file(out/"matches"/(case+".csv"),a.max_locations_per_case)
            if len(rows)!=6: raise RuntimeError(f"{case}: expected 6 mapped centers")
            for center,slide_id,idx in rows:
                f=load_rows(model,slide_id,idx); X.append(f.mean(axis=0)); domains.append(center); groups.append(case)
        X=np.stack(X); domains=np.asarray(domains); groups=np.asarray(groups); R=make_repr(X,50)
        row={"model":model,"n_cases":len(case_ids),"n_slides":len(X),"matched_locations_per_case_cap":a.max_locations_per_case,
             **cosine_pair_metrics(X,groups,domains),**domain_classifier(X,domains,groups),**paired_mmd(R,groups,domains)}
        row["domain_silhouette"]=float(silhouette_score(R,domains)); row["case_silhouette"]=float(silhouette_score(R,groups)); summaries.append(row)
    df=pd.DataFrame(summaries).sort_values(["consistency_ratio","domain_cv_balanced_accuracy"])
    path=out/"matched_slide_metrics.csv"
    if path.exists(): raise FileExistsError(path)
    df.to_csv(path,index=False)
    old=pd.read_csv(OLD_SUMMARY); old=old[old.panel=="A"]
    cols=["model","consistency_ratio","same_case_retrieval_auc","domain_cv_balanced_accuracy","domain_silhouette","case_silhouette","paired_mmd2_mean"]
    cmp=old[cols].merge(df[cols],on="model",suffixes=("_whole_slide","_matched_region"))
    for c in cols[1:]: cmp[c+"_delta_matched_minus_whole"]=cmp[c+"_matched_region"]-cmp[c+"_whole_slide"]
    cp=out/"matched_vs_whole_slide.csv"
    if cp.exists(): raise FileExistsError(cp)
    cmp.to_csv(cp,index=False)
    rp=out/"POSTHOC_REPORT.md"
    if rp.exists(): raise FileExistsError(rp)
    with open(rp,"x",encoding="utf-8") as f:
        f.write("# Matched-region slide-level posthoc\n\n")
        f.write("Uses the same grouped StandardScaler→PCA→LogisticRegression center classifier as the original WSI-level feature-space analysis.\n\n")
        f.write(df.to_markdown(index=False,floatfmt=".4f")); f.write("\n\n## Whole-slide comparison\n\n"); f.write(cmp.to_markdown(index=False,floatfmt=".4f")); f.write("\n")
    print(df.to_string(index=False),flush=True)
    print("DONE",path,cp,rp,flush=True)
if __name__=="__main__": main()
