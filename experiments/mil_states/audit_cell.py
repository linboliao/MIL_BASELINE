"""Independent file/lineage audit for one R2 PFM-fold-seed.
No training, test inference, cache writes, or deletion. Exclusive QA evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
import pandas as pd

ROOT=Path("/NAS145/linboliao/Experiments/ProstateDiagnosis")
RUNS=ROOT/"provenance/generalization/r2-mil-retrain"
PREP = RUNS / "20261105-guard-prep-r1"
STATES=("best","last","swa_last5","ema_epoch_0p9")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):
            h.update(b)
    return h.hexdigest()

def main():
    global PREP
    a=argparse.ArgumentParser()
    a.add_argument("--prep", type=Path, default=PREP, help="Frozen guard-preparation directory")
    a.add_argument("--run-id",required=True)
    a.add_argument("--host",choices=("138","195"),required=True)
    a.add_argument("--pfm",required=True)
    a.add_argument("--fold",required=True)
    a.add_argument("--seed",type=int,required=True)
    args=a.parse_args()
    PREP = args.prep.resolve(strict=True)
    expected={"138":("server94","/data12/jing/anaconda3/envs/PrePATH/bin/python"),
              "195":("ubuntu","/home/lbliao/anaconda3/envs/clam/bin/python")}
    assert (socket.gethostname(),sys.executable)==expected[args.host]
    run=RUNS/args.run_id
    train=run/"hosts"/args.host/args.pfm/"train"/args.fold/f"seed{args.seed}"
    ev=run/"hosts"/args.host/args.pfm/"eval"/args.fold/f"seed{args.seed}"
    if not (train/"TRAIN_COMPLETE.json").is_file() or not (ev/"EVAL_COMPLETE.json").is_file():
        raise RuntimeError("Incomplete training/evaluation")
    if (ev/"INDEPENDENT_QA.json").exists():
        raise FileExistsError("QA cannot overwrite")
    candidates=json.loads((train/"CANDIDATES.json").read_text())
    decision=json.loads((ev/"LOCKED_SOURCE_DECISION.json").read_text())
    summary=json.loads((ev/"EVAL_COMPLETE.json").read_text())
    report=json.loads((ev/"HELDOUT_PATIENT_REPORT.json").read_text())
    if (set(candidates)!=set(STATES) or not decision.get("no_fallback")
        or decision["risk_selected_state"]!=decision["locked_selected_state"]
        or decision["research_heldout_scoring_state"]!=decision["locked_selected_state"]):
        raise RuntimeError("Unexpected target leakage or candidate change")
    for state,item in candidates.items():
        if not Path(item["path"]).is_file() or sha(item["path"])!=item["sha256"]:
            raise RuntimeError("Candidate checksum failed: "+state)
    eps=sorted((train/"epochs").glob("epoch_*_manifest.json"))
    if len(eps)<2:raise RuntimeError("Not enough complete epochs")
    for p in eps:
        m=json.loads(p.read_text())
        ep=int(m["epoch"])
        raw=train/"epochs"/f"epoch_{ep:03d}_model.pth"
        bundle=train/"epochs"/f"epoch_{ep:03d}_training.pth"
        if sha(raw)!=m["model_sha256"] or sha(bundle)!=m["bundle_sha256"]:
            raise RuntimeError("Epoch checkpoint hash mismatch "+str(ep))
    scores={}
    for role,filename in (("risk_calibration","SOURCE_RISK_WSI.csv"),
                          ("guard","SOURCE_GUARD_WSI.csv"),
                          ("heldout","HELDOUT_WSI.csv")):
        z=pd.read_csv(ev/filename)
        parent=pd.read_csv(PREP/"splits"/args.fold/(role+".csv"))
        if len(z)!=len(parent) or set(z.raw_path)!=set(parent.raw_path):
            raise RuntimeError("WSI role coverage mismatch "+role)
        if not z.slide_path_original.str.startswith("/NAS145/").all():
            raise RuntimeError("Test path leaves NAS "+role)
        for state in (STATES if role!="heldout" else ("best",decision["locked_selected_state"])):
            col=state+"_p"
            if col not in z or not np.isfinite(z[col]).all() or not z[col].between(0,1).all():
                raise RuntimeError("Invalid probability "+role+"/"+col)
        scores[role]=len(z)
    if (sha(ev/"HELDOUT_WSI.csv")!=summary["heldout_WSI_sha256"]
        or sha(ev/"HELDOUT_PATIENT_REPORT.json")!=summary["patient_report_sha256"]):
        raise RuntimeError("Heldout audit hash mismatch")
    if report["source_choice"]!=decision["locked_selected_state"]:
        raise RuntimeError("Evaluator changed source decision")
    for key in ("selected_patient","best_patient"):
        d=report[key]
        if d["TP"]+d["FN"]!=d["positive"] or d["TN"]+d["FP"]!=d["negative"]:
            raise RuntimeError("Patient metric arithmetic inconsistent")
    if (ev/"LOCKED_SOURCE_DECISION.json").stat().st_mtime_ns>(ev/"HELDOUT_WSI.csv").stat().st_mtime_ns:
        raise RuntimeError("Source choice was not locked before heldout evaluation")
    qa={"status":"INDEPENDENT_QA_PASS","host":args.host,"pfm":args.pfm,
        "fold":args.fold,"seed":args.seed,"candidate_states":list(STATES),
        "selected_state":decision["locked_selected_state"],
        "guard_status":decision["guard_status"],
        "source_only_choice_no_fallback":True,
        "epoch_state_files_verified":len(eps),"WSI_by_role":scores,
        "real_training_complete":True,"external_inference":False,
        "run_id":args.run_id}
    with (ev/"INDEPENDENT_QA.json").open("x") as f:
        json.dump(qa,f,ensure_ascii=False,indent=2)
    print(json.dumps(qa,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
