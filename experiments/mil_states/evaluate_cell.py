"""R2 offline source-only selection + audit-only Guard + internal heldout evaluation.

All Risk/Guard/Heldout features are loaded one WSI at a time from original NAS.
No test staging or dataset preloading. No external 301/Yunnan/Panel inference.
The Risk-selected model is locked before any heldout feature/label is read.
No cleanup or model freeze.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path("/NAS145/linboliao/Experiments/ProstateDiagnosis")
RUNS=ROOT/"provenance/generalization/r2-mil-retrain"
PREP = RUNS / "20261105-guard-prep-r1"
R1=ROOT/"PFM_LOCO"
ORDER=("best","last","swa_last5","ema_epoch_0p9")
DIMS={"mstar":1024,"uni2":1536,"virchow2":2560,"h-optimus-1":1536}
FOLDS=("SL","XC","CNB","RP","TURP")

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def jwrite(path,obj):
    with path.open("x") as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str)

def csvwrite(path,df):
    with path.open("x",newline="") as f:df.to_csv(f,index=False)

def feature_map(pfm,fold,include_heldout=False):
    frames=[]
    roles=("train","selection","calibration","test") if include_heldout else ("train","selection","calibration")
    for role in roles:
        d=pd.read_csv(R1/pfm/fold/"splits"/(role+".csv"))
        frames.append(d[["raw_path","slide_path"]])
    d=pd.concat(frames,ignore_index=True)
    if d.groupby("raw_path").slide_path.nunique().max()!=1:
        raise ValueError("Conflicting R1 feature-index lineage")
    z=d.drop_duplicates("raw_path").set_index("raw_path").slide_path
    if len(z)<100 or not z.str.startswith("/NAS145/").all():
        raise ValueError("Missing original source feature references")
    return z

def manifests(pfm,fold,include_heldout=False):
    mapping=feature_map(pfm,fold,include_heldout=include_heldout)
    roles={}
    requested=("risk_calibration","guard","heldout") if include_heldout else ("risk_calibration","guard")
    for role in requested:
        d=pd.read_csv(PREP/"splits"/fold/(role+".csv"))
        if d.raw_path.duplicated().any() or not set(d.raw_path)<=set(mapping.index):
            raise ValueError("Missing or duplicated raw WSI: "+role)
        d["slide_path_original"]=d.raw_path.map(mapping)
        if not d.slide_path_original.str.startswith("/NAS145/").all():
            raise ValueError("Non-NAS test feature")
        roles[role]=d
    for i,x in enumerate(roles):
        for y in list(roles)[i+1:]:
            if set(roles[x].patient_group_id)&set(roles[y].patient_group_id):
                raise ValueError("Patient overlap in evaluation")
            if set(roles[x].identity_union_group)&set(roles[y].identity_union_group):
                raise ValueError("Identity union overlap in evaluation")
    return roles

def build_model(cfg,state_path,dim):
    from utils.yaml_utils import read_yaml
    from utils.process_utils import get_act
    from modules.AB_MIL.ab_mil import AB_MIL
    a=read_yaml(str(cfg))
    if a.General.MODEL_NAME!="AB_MIL" or int(a.Model.in_dim)!=dim:
        raise ValueError("Wrong AB-MIL architecture/dimension")
    model=AB_MIL(L=a.Model.L,D=a.Model.D,num_classes=2,
                 dropout=a.Model.dropout,act=get_act(a.Model.act),in_dim=dim)
    state=torch.load(state_path,map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to("cuda:0").eval()

def infer(pfm,df,model):
    # Intentionally no Dataset/DataLoader with preload, no cache lookup:
    # only the original NAS filepath from R1, one WSI held at a time.
    preds=[]
    with torch.inference_mode():
        for row in df.itertuples(index=False):
            p=row.slide_path_original
            if not p.startswith("/NAS145/") or f"/pt_files/{pfm}/" not in p:
                raise ValueError("Evaluation must read frozen NAS PFM features")
            f=torch.load(p,map_location="cpu",weights_only=True)
            if isinstance(f,dict):
                f=f["feats"] if "feats" in f else f["features"]
            if f.dim()==3:f=f.squeeze(0)
            if f.dim()!=2 or f.shape[1]!=DIMS[pfm] or not torch.isfinite(f).all():
                raise ValueError("Feature shape/finite failure: "+p)
            logits=model(f.unsqueeze(0).float().to("cuda:0"))["logits"]
            prob=float(torch.softmax(logits.reshape(1,2),dim=1)[0,1].item())
            preds.append(prob)
    return np.array(preds,dtype=float)

def patient_report(x,prob_col):
    g=x.groupby("patient_group_id",sort=True).agg(
        truth=("label","max"),p=(prob_col,"max"))
    pred=g.p>=.5
    pos=g.truth==1;neg=~pos
    tp=int((pos&pred).sum());fn=int((pos&~pred).sum())
    tn=int((neg&~pred).sum());fp=int((neg&pred).sum())
    sens=tp/(tp+fn) if tp+fn else None
    spec=tn/(tn+fp) if tn+fp else None
    from sklearn.metrics import roc_auc_score
    auc=float(roc_auc_score(g.truth,g.p)) if g.truth.nunique()==2 else None
    return {"patients":len(g),"positive":int(pos.sum()),"negative":int(neg.sum()),
            "TP":tp,"TN":tn,"FP":fp,"FN":fn,"sensitivity":sens,"specificity":spec,
            "balanced_accuracy":(sens+spec)/2 if sens is not None and spec is not None else None,
            "auroc":auc}

def execute(args):
    who={"138":("server94","/data12/jing/anaconda3/envs/PrePATH/bin/python",
                "/NAS3/lbliao/Code-138/MIL_BASELINE"),
         "195":("ubuntu","/home/lbliao/anaconda3/envs/clam/bin/python",
                "/NAS2/Data1/lbliao/Code-195/MIL_BASELINE")}
    hostname,py,repo=who[args.host]
    if socket.gethostname()!=hostname or sys.executable!=py:
        raise RuntimeError("Host/Python mismatch")
    if args.pfm not in DIMS or args.fold not in FOLDS or args.seed not in (42,43):
        raise RuntimeError("Unexpected matrix cell")
    if not args.run_id.startswith("20261105-r2-mil-twohost-"):
        raise RuntimeError("Unregistered run")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly one GPU must be explicitly visible")
    root=RUNS/args.run_id
    train=root/"hosts"/args.host/args.pfm/"train"/args.fold/f"seed{args.seed}"
    if not (train/"TRAIN_COMPLETE.json").is_file():
        raise RuntimeError("Training not complete, do not evaluate")
    mapping=json.loads((train/"CANDIDATES.json").read_text())
    if set(mapping)!=set(ORDER):
        raise RuntimeError("Not all four states retained")
    for state,item in mapping.items():
        if not Path(item["path"]).is_file() or sha(item["path"])!=item["sha256"]:
            raise RuntimeError("Unverifiable checkpoint: "+state)
    # Do not read heldout feature index, manifest, or labels until the source
    # decision has been selected and durably locked on independent source roles.
    roles=manifests(args.pfm,args.fold,include_heldout=False)
    out=root/"hosts"/args.host/args.pfm/"eval"/args.fold/f"seed{args.seed}"
    if out.exists():raise FileExistsError("No-clobber eval attempt")
    sys.path.insert(0,repo)
    sys.path.insert(0,str(root/"scripts"))
    from . import selection as selector
    from utils.yaml_utils import read_yaml
    cfg=PREP/"config"/"training"/args.pfm/args.fold/f"seed{args.seed}.yaml"
    if not args.execute:
        print(json.dumps({"preflight":"EVAL_PASS_NO_MODEL_FORWARD",
             "pfm":args.pfm,"fold":args.fold,"seed":args.seed,
             "NAS_roles":{k:len(v) for k,v in roles.items()},
             "candidates_verified":list(mapping)}))
        return
    out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
    risk=roles["risk_calibration"].copy()
    guard=roles["guard"].copy()
    for state in ORDER:
        m=build_model(cfg,mapping[state]["path"],DIMS[args.pfm])
        risk[state+"_p"]=infer(args.pfm,risk,m)
        guard[state+"_p"]=infer(args.pfm,guard,m)
        del m
    csvwrite(out/"SOURCE_RISK_WSI.csv",risk)
    csvwrite(out/"SOURCE_GUARD_WSI.csv",guard)
    selected,rank,cell=selector.select_no_fallback(
        risk,guard,roles["risk_calibration"],roles["guard"],method=selector.PRIMARY)
    # Lock and persist the source-only decision *before* reading heldout labels.
    selected["candidate_checkpoint_sha256"]=mapping[selected["locked_selected_state"]]["sha256"]
    selected["source_risk_scores_sha256"]=sha(out/"SOURCE_RISK_WSI.csv")
    selected["source_guard_scores_sha256"]=sha(out/"SOURCE_GUARD_WSI.csv")
    jwrite(out/"LOCKED_SOURCE_DECISION.json",selected)
    csvwrite(out/"SOURCE_RISK_CANDIDATES.csv",rank)
    csvwrite(out/"SOURCE_RISK_CELLS.csv",cell)
    hold=manifests(args.pfm,args.fold,include_heldout=True)["heldout"].copy()
    chosen=selected["locked_selected_state"]
    for state in ("best",chosen):
        if state+"_p" in hold.columns:continue
        m=build_model(cfg,mapping[state]["path"],DIMS[args.pfm])
        hold[state+"_p"]=infer(args.pfm,hold,m)
        del m
    report={"host":args.host,"pfm":args.pfm,"fold":args.fold,"seed":args.seed,
        "source_choice":chosen,"guard_status":selected["guard_status"],
        "no_fallback":True,"heldout_WSI":len(hold),
        "selected_patient":patient_report(hold,chosen+"_p"),
        "best_patient":patient_report(hold,"best_p"),
        "external_inference":False,"final_freeze":False}
    csvwrite(out/"HELDOUT_WSI.csv",hold)
    jwrite(out/"HELDOUT_PATIENT_REPORT.json",report)
    jwrite(out/"EVAL_COMPLETE.json",{
        "status":"EVAL_COMPLETE_PENDING_INDEPENDENT_QA",
        "risk_state":chosen,"guard_status":selected["guard_status"],
        "NAS_only_test_reads":True,"no_stage_test_data":True,
        "source_selected_before_heldout":True,
        "heldout_WSI_sha256":sha(out/"HELDOUT_WSI.csv"),
        "patient_report_sha256":sha(out/"HELDOUT_PATIENT_REPORT.json"),
        "external_inference":False,"final_freeze":False})
    print("SOURCE_AND_HELDOUT_EVAL_COMPLETE",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",choices=("138","195"),required=True)
    ap.add_argument("--pfm",choices=tuple(DIMS),required=True)
    ap.add_argument("--fold",choices=FOLDS,required=True)
    ap.add_argument("--seed",type=int,choices=(42,43),required=True)
    ap.add_argument("--prep", type=Path, default=PREP, help="Frozen guard-preparation directory")
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    PREP = args.prep.resolve(strict=True)
    execute(args)
