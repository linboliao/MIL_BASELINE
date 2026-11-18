import argparse, hashlib, json, platform
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()
def ci(x):
    x=np.asarray(x,dtype=float)
    return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
def exact(k,n):
    return [0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),1.0 if k==n else float(beta.ppf(.975,k+1,n-k))]
def wr(df,path):
    df.to_csv(path,index=False,encoding="utf-8")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True,type=Path); ap.add_argument("--predictions",required=True,type=Path); ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    package=a.config.resolve().parents[1]
    cfg=json.loads(a.config.read_text())
    src=a.predictions.resolve()
    out=a.out.resolve()
    provenance=package.parent/"provenance"
    if provenance not in out.parents:
        ap.error("Output must be below project provenance")
    pred=pd.read_csv(src)
    pairs=pd.read_csv(package/cfg["pairs"])
    slides=pd.read_csv(package/cfg["slides"])
    mm=pd.read_csv(package/cfg["models"])
    expected = slides.rename(columns={"cohort":"panel"})
    truth = expected.set_index(["panel","stem"])
    assert truth.index.is_unique
    for model_id, group in pred.groupby("model_id"):
        model = mm.set_index("model_id").loc[model_id]
        indexed = group.set_index(["panel","stem"])
        assert indexed.index.is_unique and set(indexed.index)==set(truth.index)
        assert indexed.checkpoint_sha256.eq(model.checkpoint_sha256).all()
        for field in ("manual_label","case_id","identity_union_group","quality_usable_for_primary"):
            assert indexed[field].reindex(truth.index).equals(truth[field]), (model_id, field)
    assert set(pred.model_id)==set(mm.model_id)
    assert cfg["threshold"]==.5 and cfg["bootstrap_draws"]==5000 and cfg["bootstrap_seed"]==20261103
    assert len(pred)==2838 and len(pairs)==1180 and len(slides)==473 and len(mm)==6
    assert not pred.duplicated(["model_id","panel","stem"]).any()
    assert pred["threshold"].eq(.5).all() and ((pred.prob_pos>=.5).astype(int)==pred.decision).all()
    assert pred.groupby(["model_id","panel"]).size().eq([360,113]*6).all()
    assert pairs.groupby("panel").size().to_dict()=={"A":900,"B":280}
    assert (pairs.left_identity_union_group==pairs.right_identity_union_group).all()
    assert pred.loc[pred.manual_label.notna()].shape[0]==2832
    assert pred.groupby("panel").identity_union_group.nunique().to_dict()=={"A":52,"B":19}
    assert pairs.groupby(["panel","manual_truth_relation"]).size().to_dict()=={("A","different"):5,("A","same"):895,("B","same"):275,("B","unavailable"):5}
    out.mkdir(parents=True,exist_ok=False)
    root=out/"outputs"; root.mkdir()
    manifest=[dict(path=str(f),sha256=sha(f)) for f in
              (src,a.config,package/cfg["models"],package/cfg["pairs"],package/cfg["slides"])]
    wr(pd.DataFrame(manifest),out/"inputs_manifest.csv")
    model_ids=list(pred.model_id.unique()); panels=["A","B"]
    pred["error"]=np.where(pred.manual_label.notna(),(pred.decision !=pred.manual_label.astype("Int64")).astype(float),np.nan)
    lookup=pred.set_index(["model_id","panel","stem"])
    rows=[]
    for p in pairs.itertuples(index=False):
        for m in model_ids:
            l=lookup.loc[(m,p.panel,p.left_stem)]
            r=lookup.loc[(m,p.panel,p.right_stem)]
            assert l.case_id==r.case_id==p.case_id
            assert l.identity_union_group==r.identity_union_group==p.left_identity_union_group
            assert l.manual_label_available==bool(p.left_manual_label_available)
            assert r.manual_label_available==bool(p.right_manual_label_available)
            eligible=p.manual_truth_relation=="same" and bool(p.manual_pair_quality_eligible)
            if eligible:
                assert pd.notna(l.manual_label) and pd.notna(r.manual_label) and l.manual_label==r.manual_label
            rows.append(dict(model_id=m,panel=p.panel,case_id=p.case_id,identity_union_group=l.identity_union_group,pair_id=p.pair_id,left_stem=p.left_stem,right_stem=p.right_stem,truth_relation=p.manual_truth_relation,primary_eligible=eligible,left_label=l.manual_label,right_label=r.manual_label,left_prediction=int(l.decision),right_prediction=int(r.decision),left_prob=l.prob_pos,right_prob=r.prob_pos,left_error=l.error,right_error=r.error,disagree=int(l.decision!=r.decision),abs_prob_diff=abs(l.prob_pos-r.prob_pos)))
    pp=pd.DataFrame(rows); wr(pp,root/"pair_results.csv")
    assert pp.groupby(["model_id","panel","primary_eligible"]).size().unique().tolist() and len(pp)==7080
    valid=pp[pp.primary_eligible].copy()
    assert valid.groupby(["model_id","panel"]).size().to_dict()=={(m,p):895 if p=="A" else 275 for m in model_ids for p in panels}
    case_pair=valid.groupby(["model_id","panel","identity_union_group","case_id"],as_index=False).agg(disagreement=("disagree","mean"),prob_diff=("abs_prob_diff","mean"),valid_pairs=("pair_id","size"),flipped_pairs=("disagree","sum"))
    group_pair=case_pair.groupby(["model_id","panel","identity_union_group"],as_index=False).agg(disagreement=("disagreement","mean"),prob_diff=("prob_diff","mean"),case_count=("case_id","size"),valid_pairs=("valid_pairs","sum"),flipped_pairs=("flipped_pairs","sum"))
    wr(case_pair,root/"case_pair_metrics.csv");wr(group_pair,root/"patient_group_metrics.csv")
    diagnostic=pred[pred.manual_label.notna() & pred.quality_usable_for_primary.astype(bool)].copy()
    assert len(diagnostic)==2832
    case_error=diagnostic.groupby(["model_id","panel","case_id","identity_union_group"],as_index=False).agg(slides=("stem","size"),errors=("error","sum"),error_fraction=("error","mean"),distinct_labels=("manual_label","nunique"),distinct_predictions=("decision","nunique"),prob_range=("prob_pos",lambda x:x.max()-x.min()))
    case_error["correctness_state"]=np.select([case_error.errors.eq(0),case_error.errors.eq(case_error.slides)],["all_correct","all_wrong"],default="partly_wrong")
    case_error["any_prediction_flip"]=case_error.distinct_predictions.gt(1)
    case_error["uniform_but_wrong"]=case_error.distinct_predictions.eq(1)&case_error.errors.gt(0)
    wr(case_error,root/"case_metrics.csv")
    group_err=case_error.groupby(["model_id","panel","identity_union_group"],as_index=False).agg(error_rate=("error_fraction","mean"),any_wrong=("errors",lambda x: int((x>0).any())),all_wrong_cases=("correctness_state",lambda x: int((x=="all_wrong").all())),any_flip=("any_prediction_flip","max"))
    wr(group_err,root/"patient_group_correctness.csv")
    from sklearn.metrics import roc_auc_score
    lab=[]
    for (m,p,u),g in diagnostic.groupby(["model_id","panel","unit_id"]):
        y=g.manual_label.astype(int); d=g.decision.astype(int); npos=int(y.sum());nneg=int((1-y).sum())
        sens=float((d[y==1]==1).mean()) if npos else np.nan
        spec=float((d[y==0]==0).mean()) if nneg else np.nan
        lab.append(dict(model_id=m,panel=p,unit_id=u,slides=len(g),positive=npos,negative=nneg,sensitivity=sens,specificity=spec,bACC=(sens+spec)/2 if npos and nneg else np.nan,AUROC=roc_auc_score(y,g.prob_pos) if npos and nneg else np.nan))
    wr(pd.DataFrame(lab),root/"laboratory_metrics.csv")
    summary=[]
    for (m,p),g in group_pair.groupby(["model_id","panel"]):
        e=group_err[(group_err.model_id==m)&(group_err.panel==p)]
        assert len(g)==len(e)==(52 if p=="A" else 19)
        v=valid[(valid.model_id==m)&(valid.panel==p)]
        cs=case_error[(case_error.model_id==m)&(case_error.panel==p)]
        summary.append(dict(model_id=m,panel=p,patient_groups=len(g),cases=cs.case_id.nunique(),valid_pairs=len(v),flipped_pairs=int(v.disagree.sum()),raw_pair_disagreement=float(v.disagree.mean()),patient_equal_disagreement=float(g.disagreement.mean()),patient_equal_prob_diff=float(g.prob_diff.mean()),patient_equal_error_rate=float(e.error_rate.mean()),any_flip_patient_groups=int(g.flipped_pairs.gt(0).sum()),any_flip_group_ci_low=exact(int(g.flipped_pairs.gt(0).sum()),len(g))[0],any_flip_group_ci_high=exact(int(g.flipped_pairs.gt(0).sum()),len(g))[1],all_correct_cases=int((cs.correctness_state=="all_correct").sum()),partly_wrong_cases=int((cs.correctness_state=="partly_wrong").sum()),all_wrong_cases=int((cs.correctness_state=="all_wrong").sum()),uniform_but_wrong_cases=int(cs.uniform_but_wrong.sum())))
    S=pd.DataFrame(summary)
    rng=np.random.default_rng(20261103); bootstrap_indices=[]; drawrows=[]
    for p in panels:
        gids=sorted(group_pair[group_pair.panel==p].identity_union_group.unique())
        assert len(gids)==(52 if p=="A" else 19)
        ix=rng.integers(0,len(gids),size=(5000,len(gids)))
        for j in range(len(ix)):
            bootstrap_indices.append(dict(panel=p,draw=j,group_indices_json=json.dumps(ix[j].tolist()),group_names_json=json.dumps([gids[k] for k in ix[j]])))
        mats={}
        for field,table in [("disagreement",group_pair),("error_rate",group_err),("prob_diff",group_pair)]:
            mats[field]=np.stack([table[(table.model_id==m)&(table.panel==p)].set_index("identity_union_group").reindex(gids)[field].to_numpy() for m in model_ids])
            assert np.isfinite(mats[field]).all()
        arrays={field:mat[:,ix].mean(axis=2) for field,mat in mats.items()}
        for j in range(5000):
            for i,m in enumerate(model_ids):
                drawrows.append(dict(panel=p,draw=j,model_id=m,disagreement=arrays["disagreement"][i,j],error_rate=arrays["error_rate"][i,j],prob_diff=arrays["prob_diff"][i,j]))
        for i,m in enumerate(model_ids):
            row=S.index[(S.model_id==m)&(S.panel==p)][0]
            for field in arrays:
                lo,hi=ci(arrays[field][i]);S.loc[row,field+"_ci_low"]=lo;S.loc[row,field+"_ci_high"]=hi
    wr(pd.DataFrame(bootstrap_indices),root/"bootstrap_indices.csv")
    D=pd.DataFrame(drawrows);wr(D,root/"bootstrap_draws.csv")
    primary=S[S.model_id.isin(cfg["primary_models"])]
    wr(primary,root/"primary_summary.csv")
    wr(S[~S.model_id.isin(cfg["primary_models"])],root/"sensitivity_summary.csv")
    contrasts=[]
    for p in panels:
        for a_,b_,title in [("uni2_seed44_best","uni2_seed42_best","uni2_source_rule_minus_base"),("mstar_seed42_best","uni2_seed42_best","mstar_minus_uni2_base")]:
            A=S[(S.panel==p)&(S.model_id==a_)].iloc[0];B=S[(S.panel==p)&(S.model_id==b_)].iloc[0]
            for endpoint in ["disagreement","error_rate","prob_diff"]:
                dr=D[(D.panel==p)&(D.model_id==a_)][endpoint].to_numpy()-D[(D.panel==p)&(D.model_id==b_)][endpoint].to_numpy()
                lo,hi=ci(dr)
                point=(A["patient_equal_"+endpoint] if endpoint!="disagreement" else A.patient_equal_disagreement)-(B["patient_equal_"+endpoint] if endpoint!="disagreement" else B.patient_equal_disagreement)
                contrasts.append(dict(panel=p,contrast=title,endpoint=endpoint,difference=float(point),ci_low=lo,ci_high=hi,draw_count=len(dr)))
    C=pd.DataFrame(contrasts);wr(C,root/"paired_model_contrasts.csv")
    seed=S[S.model_id.str.startswith("uni2")].copy()
    seedagg=seed.groupby("panel").agg(disagreement_mean=("patient_equal_disagreement","mean"),disagreement_sd=("patient_equal_disagreement","std"),disagreement_min=("patient_equal_disagreement","min"),disagreement_max=("patient_equal_disagreement","max"),error_mean=("patient_equal_error_rate","mean"),error_sd=("patient_equal_error_rate","std"),error_min=("patient_equal_error_rate","min"),error_max=("patient_equal_error_rate","max")).reset_index()
    wr(seedagg,root/"uni2_five_seed_descriptives.csv")
    wide=diagnostic[diagnostic.model_id.str.startswith("uni2")].pivot(index=["panel","stem","case_id","identity_union_group"],columns="model_id",values="decision")
    across=wide.reset_index();across["n_positive_seed"]=wide.sum(axis=1).to_numpy();across["any_seed_disagreement"]=wide.nunique(axis=1).gt(1).to_numpy()
    wr(across,root/"uni2_cross_seed_slide_decisions.csv")
    exceptions=pp[~pp.primary_eligible];wr(exceptions,root/"excluded_pair_audit.csv")
    # independent arithmetic check, using original slide data rather than derived case/group tables
    indep=[]
    for (m,p),q in valid.groupby(["model_id","panel"]):
        temp=q.groupby(["identity_union_group","case_id"]).disagree.mean().groupby(level=0).mean()
        actual=float(temp.mean());expected=float(S[(S.model_id==m)&(S.panel==p)].patient_equal_disagreement.iloc[0])
        assert abs(actual-expected)<1e-12
        indep.append(dict(model_id=m,panel=p,recomputed=actual,summary=expected,match=True,groups=len(temp)))
    wr(pd.DataFrame(indep),root/"independent_QA.csv")
    for metric,label,filename in [("patient_equal_disagreement","患者组等权跨实验室预测不一致率","figure_primary_disagreement.png"),("patient_equal_error_rate","患者组等权逐病例错误比例","figure_patient_error.png")]:
        fig,ax=plt.subplots(figsize=(10,4.5))
        colors=plt.cm.tab10(np.linspace(0,1,6))
        for j,p in enumerate(panels):
            ss=S[S.panel==p].set_index("model_id").loc[model_ids]
            x=np.arange(6)+j*.37
            field= "disagreement" if metric.endswith("disagreement") else "error_rate"
            vals=ss[metric].to_numpy()
            ax.errorbar(x,vals,yerr=np.maximum(0,[vals-ss[field+"_ci_low"].to_numpy(),ss[field+"_ci_high"].to_numpy()-vals]),fmt="o",capsize=3,label="Panel "+p)
        ax.set_xticks(np.arange(6)+.185,model_ids,rotation=35,ha="right")
        ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.legend();fig.tight_layout();fig.savefig(root/filename,dpi=200);plt.close(fig)
    with open(out/"run_meta.json","x") as f:json.dump(dict(study="panelab-fullsource-frozen-statistics",run_tag=out.name,purpose="six frozen models Panel A/B statistical analysis",status="completed",scientific_status="candidate_pending_human_review",host=platform.node(),python=platform.python_version(),parent_run=str(src),bootstrap_seed=20261103,bootstrap_draws=5000,threshold=.5,output_path=str(out),completed_at=datetime.now().astimezone().isoformat(),selection_on_panel=False),f,indent=2)
    with open(out/"REPORT.md","x") as f:
        f.write("# Panel A/B 六模型冻结统计报告\n\n分析对象：同人工标签有效配对；case 内、保守身份组内、身份组间依次等权。所有模型固定阈值 0.5，未以 Panel 标签选权重。区间为同组共享抽样的5,000次患者组 bootstrap percentile pointwise 95% CI；不提供事后 p 值。\n\n")
        f.write("## 主结果（比率；括号为95% CI）\n\n")
        for p in panels:
            f.write("### Panel "+p+"\n\n")
            for _,v in primary[primary.panel==p].iterrows():
                f.write(f"- {v.model_id}: 不一致率 {v.patient_equal_disagreement:.4f} [{v.disagreement_ci_low:.4f},{v.disagreement_ci_high:.4f}]；错误比例 {v.patient_equal_error_rate:.4f} [{v.error_rate_ci_low:.4f},{v.error_rate_ci_high:.4f}]；原始翻转pair {int(v.flipped_pairs)}/{int(v.valid_pairs)}；全错case {int(v.all_wrong_cases)}。\n")
            f.write("\n")
        f.write("## 关键配对效应（百分点）\n\n")
        for _,v in C[C.contrast=="uni2_source_rule_minus_base"].iterrows():
            f.write(f"- {v.panel} / {v.endpoint}: {100*v.difference:+.2f} pp [ {100*v.ci_low:+.2f}, {100*v.ci_high:+.2f} ]\n")
        f.write("\n## 限制\n\n本研究有先前 Panel 接触；切片/染色/扫描和切面结构存在混杂。跨模型概率尺度不直接可比。低不一致率不代表准确；患者保守身份组不一定等于完全确认的临床患者。B组数少、零事件需要额外精确组事件区间。多seed仅描述性，不将seed当独立患者。结果仅候选，未经人工批准不得进入 studies。\n")
    outputs=[]
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name!="OUTPUT_MANIFEST.csv":
            outputs.append(dict(relative_path=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size))
    wr(pd.DataFrame(outputs),out/"OUTPUT_MANIFEST.csv")
    with open(out/"VALIDATION.json","x") as f:json.dump(dict(status="passed",rows_predictions=len(pred),rows_pairs=len(pairs),model_panel_cells=12,models=model_ids,eligible_pairs_A=895,eligible_pairs_B=275,bootstrap_indices=len(bootstrap_indices),bootstrap_draws=len(D),recomputation_passed=True,non_clobbering=True),f,indent=2)
    print(S[["model_id","panel","patient_equal_disagreement","disagreement_ci_low","disagreement_ci_high","patient_equal_error_rate","error_rate_ci_low","error_rate_ci_high"]].to_string(index=False))
    print(C[C.contrast=="uni2_source_rule_minus_base"].to_string(index=False))
if __name__=="__main__":main()
