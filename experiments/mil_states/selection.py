"""R2 source-only state-selection rule with *audit-only* disjoint Guard.

No fallback, no second-best rescue, no target/clinical-data inputs.
The selected candidate is *always* the Risk-calibration argmin, even on Guard failure.
Research heldout evaluation uses this locked decision; Guard labels do not update it.
New run directories are exclusive. No model forward pass or training is performed.

Metrics:
  worst_cell_nll (predefined primary, continuity with prior R2 source-rule prep)
  worst_cell_brier, balanced_nll, shrunk_worst_cell_nll (predefined ablations).
Ablations must not be chosen based on Guard/heldout performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CANDIDATES = ("best", "last", "swa_last5", "ema_epoch_0p9")
METHODS = ("worst_cell_nll", "worst_cell_brier",
           "balanced_nll", "shrunk_worst_cell_nll")
PRIMARY = "worst_cell_nll"
TIE_TOL = 1e-6
THRESHOLD = .5
SHRINK_PSEUDOCOUNT = 5.
COLS = ("raw_path", "patient_group_id", "identity_union_group",
        "center", "label")
REQUIRED_CENTERS = {"省立","新昌","迈新"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(1048576), b""):
            h.update(buf)
    return h.hexdigest()


def validate_scores(scores, manifest, label):
    if not set(COLS).issubset(scores.columns) or not set(COLS).issubset(manifest.columns):
        raise ValueError(label + ": missing identity/label columns")
    if "new_split" not in manifest or set(manifest.new_split) != {"calibration"}:
        raise ValueError(label + ": source must be original calibration role")
    if not set(manifest.center) <= REQUIRED_CENTERS:
        raise ValueError(label + ": unexpected source center")
    for frame in (scores, manifest):
        if frame.raw_path.duplicated().any() or frame.raw_path.isna().any():
            raise ValueError(label + ": duplicate/null WSI")
        if frame[list(COLS)].isna().any().any():
            raise ValueError(label + ": null identity")
    if set(scores.raw_path) != set(manifest.raw_path) or len(scores) != len(manifest):
        raise ValueError(label + ": prediction/manifest WSI mismatch")
    x=scores.set_index("raw_path").loc[manifest.raw_path]
    y=manifest.set_index("raw_path")
    for c in COLS[1:]:
        if x[c].astype(str).tolist()!=y[c].astype(str).tolist():
            raise ValueError(label + ": source identity mismatch "+c)
    if y.groupby("patient_group_id").center.nunique().max()!=1:
        raise ValueError(label + ": multi-center patient")
    if y.groupby("identity_union_group").patient_group_id.nunique().max()!=1:
        raise ValueError(label + ": identity union spans multiple patients")
    if not y.label.isin([0,1]).all():
        raise ValueError(label + ": nonbinary label")
    for state in CANDIDATES:
        name=state+"_p"
        if name not in x or not pd.api.types.is_numeric_dtype(x[name]):
            raise ValueError(label + ": missing numeric state "+name)
        if not np.isfinite(x[name].to_numpy(dtype=float)).all():
            raise ValueError(label + ": nonfinite probabilities")
        if not x[name].between(0,1).all():
            raise ValueError(label + ": probability out of range")
    return x.reset_index()


def patient_frame(slides, candidate):
    z=slides.groupby("patient_group_id", sort=True).agg(
        y=("label", "max"),center=("center","first"),
        identity_union_group=("identity_union_group","first"),
        p=(candidate+"_p", "max"))
    z["pred"]=z.p>=THRESHOLD
    p=np.clip(z.p.to_numpy(dtype=float),1e-7,1.-1e-7)
    y=z.y.to_numpy(dtype=float)
    z["nll"]=-y*np.log(p)-(1-y)*np.log1p(-p)
    z["brier"]=(z.p-z.y)**2
    return z


def candidate_table(risk):
    table=[]
    cells=[]
    for m in CANDIDATES:
        p=patient_frame(risk,m)
        classes=p.groupby("y")
        if 0 not in classes.groups or 1 not in classes.groups:
            raise ValueError("Risk-calibration has no positive or negative patients")
        class_nll=classes.nll.mean().to_dict()
        detail=p.groupby(["center","y"],sort=True).agg(
            n=("nll","size"),nll=("nll","mean"),brier=("brier","mean"))
        all_centers=sorted(set(risk.center))
        for center in all_centers:
            for label in (0,1):
                if (center,label) in detail.index:
                    row=detail.loc[(center,label)]
                    cells.append({"candidate":m,"center":center,"patient_class":label,
                                  "patients":int(row.n),"nll":float(row.nll),
                                  "brier":float(row.brier)})
                else:
                    cells.append({"candidate":m,"center":center,"patient_class":label,
                                  "patients":0,"nll":float("nan"),"brier":float("nan")})
        shrunken=[]
        for (center,cl), cell in detail.iterrows():
            shrunken.append((float(cell.n)*float(cell.nll) +
                             SHRINK_PSEUDOCOUNT*class_nll[int(cl)]) /
                            (float(cell.n)+SHRINK_PSEUDOCOUNT))
        table.append({
            "candidate":m,
            "worst_cell_nll":float(detail.nll.max()),
            "worst_cell_brier":float(detail.brier.max()),
            "balanced_nll":float(.5*class_nll[0]+.5*class_nll[1]),
            "shrunk_worst_cell_nll":float(max(shrunken)),
            "positive_patients":int((p.y==1).sum()),
            "negative_patients":int((p.y==0).sum()),
            "FN":int(((p.y==1)&~p.pred).sum()),
            "FP":int(((p.y==0)&p.pred).sum()),
            "source_centers":int(len(detail.index.get_level_values(0).unique())),
            "observed_center_class_cells":len(detail),
        })
    return pd.DataFrame(table),pd.DataFrame(cells)


def risk_select(table, method):
    if method not in METHODS:
        raise ValueError("Unregistered risk method")
    minv=table[method].min()
    matched=set(table.loc[table[method] <= minv+TIE_TOL, "candidate"])
    return next(c for c in CANDIDATES if c in matched)


def guard_audit(slides, selected):
    best=patient_frame(slides,"best")
    trial=patient_frame(slides,selected)
    assert best.index.equals(trial.index)
    total_pos=int((best.y==1).sum())
    total_neg=int((best.y==0).sum())
    if selected=="best":
        return {"status":"BASELINE_SELECTED","is_independent_certification":False,
                "candidate_modified":False,"positive_patients":total_pos,
                "negative_patients":total_neg}
    new_fn=int(((best.y==1)&best.pred&~trial.pred).sum())
    rescued_fn=int(((best.y==1)&~best.pred&trial.pred).sum())
    new_fp=int(((best.y==0)&~best.pred&trial.pred).sum())
    rescued_fp=int(((best.y==0)&best.pred&~trial.pred).sum())
    source_centers=sorted(set(best.center))
    cell_support=[]
    per_center=[]
    dropped=False
    incomplete=False
    for center in source_centers:
        b=best[best.center==center]
        t=trial[trial.center==center]
        for cls in (0,1):
            n=int((b.y==cls).sum())
            cell_support.append({"center":center,"patient_label":cls,"patients":n,
                                 "estimable":n>0})
            if n==0:
                incomplete=True
        neg=b.y==0
        if neg.any():
            bspec=float((~b.loc[neg,"pred"]).mean())
            tspec=float((~t.loc[neg,"pred"]).mean())
            delta=tspec-bspec
            if delta < -.05-1e-12:
                dropped=True
            per_center.append({"center":center,"negative_patients":int(neg.sum()),
                               "best_spec":bspec,"selected_spec":tspec,
                               "delta_spec":delta,"loss_beyond_5pp":delta<-.05-1e-12})
        else:
            per_center.append({"center":center,"negative_patients":0,
                               "best_spec":None,"selected_spec":None,
                               "delta_spec":None,"loss_beyond_5pp":None})
    if total_pos==0 or total_neg==0:
        status="NOT_ASSESSABLE"
    elif new_fn>0 or dropped:
        status="FAIL_OBSERVED"
    elif incomplete:
        status="PASS_OBSERVED_INCOMPLETE_SUPPORT"
    else:
        status="PASS_OBSERVED"
    return {
        "status":status,
        "candidate_modified":False,
        "guard_used_to_rank":False,
        "positive_patients":total_pos,"negative_patients":total_neg,
        "introduced_FN_against_best":new_fn,"rescued_FN_against_best":rescued_fn,
        "introduced_FP_against_best":new_fp,"rescued_FP_against_best":rescued_fp,
        "source_center_class_support":cell_support,
        "per_center_specificity":per_center,
        "is_independent_certification":False,
        "not_for_clinical_deployment":True,
        "note":"Guard is independent from Risk for this fold, not a new external center."}


def select_no_fallback(risk_scores, guard_scores, risk_manifest, guard_manifest,
                       method=PRIMARY):
    risk=validate_scores(risk_scores,risk_manifest,"risk")
    guard=validate_scores(guard_scores,guard_manifest,"guard")
    for col in ("patient_group_id","identity_union_group","raw_path"):
        if set(risk[col]) & set(guard[col]):
            raise ValueError("Risk/Guard patient identity leakage: "+col)
    table,cells=candidate_table(risk)
    chosen=risk_select(table,method)
    audit=guard_audit(guard,chosen)
    choice={
        "rule_version":"r2-no-fallback-v2-predeclared",
        "risk_method":method,
        "is_primary":method==PRIMARY,
        "risk_selected_state":chosen,
        "locked_selected_state":chosen,
        "no_fallback":True,
        "no_guard_based_reselection":True,
        "no_second_candidate_search":True,
        "guard_status":audit["status"],
        "guard":audit,
        "research_heldout_scoring_state":chosen,
        "research_heldout_scoring_includes_guard_failures":True,
        "clinical_use_or_final_model_freeze_authorized":False,
        "target_labels_used":False,
        "comparison_baseline":"best",
        "patient_probability_aggregation":"max_wsi_probability",
        "threshold":THRESHOLD,
        "tie_tolerance":TIE_TOL,
        "candidate_tie_order":CANDIDATES,
        "missing_center_class":"not_estimable_not_zero",
    }
    return choice, table, cells


def self_test():
    def synth(prefix,best,last):
        man=pd.DataFrame({
            "raw_path":[f"/{prefix}1",f"/{prefix}2"],
            "patient_group_id":[f"新昌|{prefix}1",f"新昌|{prefix}2"],
            "identity_union_group":[f"新昌|{prefix}1",f"新昌|{prefix}2"],
            "center":["新昌","新昌"], "label":[0,1],
            "new_split":["calibration","calibration"]})
        scores=man.drop(columns=["new_split"]).copy()
        scores["best_p"]=best
        scores["last_p"]=last
        scores["swa_last5_p"]=best
        scores["ema_epoch_0p9_p"]=best
        return scores,man
    risk,rm=synth("risk",[.4,.6],[.1,.9])
    guard,gm=synth("guard",[.1,.9],[.1,.1])
    s,table,_=select_no_fallback(risk,guard,rm,gm)
    assert s["locked_selected_state"]=="last"
    assert s["guard_status"]=="FAIL_OBSERVED"
    assert s["research_heldout_scoring_state"]=="last"
    assert s["guard"]["introduced_FN_against_best"]==1
    guard["last_p"]=[.1,.95]
    s,_,_=select_no_fallback(risk,guard,rm,gm)
    assert s["locked_selected_state"]=="last"
    assert s["guard_status"]=="PASS_OBSERVED"
    risk["best_p"]=[.01,.99]
    risk["swa_last5_p"]=[.4,.6]
    risk["ema_epoch_0p9_p"]=[.4,.6]
    s,_,_=select_no_fallback(risk,guard,rm,gm)
    assert s["locked_selected_state"]=="best" and s["guard_status"]=="BASELINE_SELECTED"
    bad=gm.copy()
    bad["patient_group_id"]=rm["patient_group_id"]
    try:
        select_no_fallback(risk,guard,rm,bad)
        raise AssertionError("identity mismatch accepted")
    except ValueError:
        pass
    for m in METHODS:
        sel,_,_=select_no_fallback(risk,guard,rm,gm,method=m)
        assert sel["locked_selected_state"] in CANDIDATES
        assert sel["no_fallback"]
    print(json.dumps({"self_test":"PASS","guard_failure_keeps_selected":"last",
                      "guard_pass_keeps_selected":"last",
                      "natural_best_selection":"best",
                      "identity_mismatch_rejected":True,
                      "methods_tested":list(METHODS),
                      "training_started":False,"external_inference":False}))


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    cli=argparse.ArgumentParser()
    cli.add_argument("--risk-scores",required=True,type=Path)
    cli.add_argument("--guard-scores",required=True,type=Path)
    cli.add_argument("--risk-manifest",required=True,type=Path)
    cli.add_argument("--guard-manifest",required=True,type=Path)
    cli.add_argument("--method",choices=METHODS,default=PRIMARY)
    cli.add_argument("--output",required=True,type=Path)
    args=cli.parse_args()
    provenance_root=Path("/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance").resolve()
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(provenance_root):
        raise ValueError("Scientific decisions must be written under governed NAS provenance")
    if args.output.exists():
        raise FileExistsError("No-clobber output already exists")
    files=(args.risk_scores,args.guard_scores,args.risk_manifest,args.guard_manifest)
    decision,rank_table,cell_table=select_no_fallback(
        pd.read_csv(args.risk_scores),pd.read_csv(args.guard_scores),
        pd.read_csv(args.risk_manifest),pd.read_csv(args.guard_manifest),
        method=args.method)
    inputs={}
    for name,file in zip(("risk_scores","guard_scores","risk_manifest","guard_manifest"),files):
        inputs[name]={"path":str(file),"sha256":sha256_file(file)}
    decision["input_provenance"]=inputs
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"SOURCE_ONLY_DECISION.json").open("x") as f:
        json.dump(decision,f,indent=2,ensure_ascii=False)
    with (args.output/"RISK_RANKING.csv").open("x",newline="") as f:
        rank_table.to_csv(f,index=False)
    with (args.output/"RISK_CELL_SUPPORT.csv").open("x",newline="") as f:
        cell_table.to_csv(f,index=False)
    print(json.dumps({"locked_selected":decision["locked_selected_state"],
                      "guard_status":decision["guard_status"],
                      "no_fallback":True,
                      "external_labels_used":False,"training_started":False}))


if __name__=="__main__":
    main()
