"""R2 AB-MIL one seed: strict fold-specific CPU RAM preload of locally staged PFM.

This is NOT a generic framework wrapper; it faithfully reuses the MIL repository's
AB_MIL, train_loop, val_loop, optimizers/schedulers, anchor-window early stopping,
and macroF1+CE Best tie-breaking. It writes all real epoch model+resume states and
materializes Last, SWA-last5 and epoch-end EMA0.9 without target scores.
Only a complete, SHA-verified PFM stage may be used. No test labels/features.
Outputs are exclusive NAS run artifacts; nothing is deleted.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .training_inputs import RecordingSampler, StrictResident, mem_available_bytes

RUN_PARENT=Path("/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/r2-mil-retrain")
PREP = RUN_PARENT / "20261105-guard-prep-r1"
FOLDS=("SL","XC","CNB","RP","TURP")
DIMS={"uni2":1536,"virchow2":2560,"h-optimus-1":1536,"mstar":1024}
HEADROOM=30*(2**30)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def save_json_exclusive(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,default=str)

def save_torch_exclusive(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:
        torch.save(obj,f)
        f.flush();os.fsync(f.fileno())

def get_paths_and_params(host,pfm,fold,seed,run_id):
    hosts={"138":("server94","/data12/jing/anaconda3/envs/PrePATH/bin/python",
                   "/NAS3/lbliao/Code-138/MIL_BASELINE"),
           "195":("ubuntu","/home/lbliao/anaconda3/envs/clam/bin/python",
                   "/NAS2/Data1/lbliao/Code-195/MIL_BASELINE")}
    expected_host,python,repo=hosts[host]
    if socket.gethostname()!=expected_host or sys.executable!=python:
        raise RuntimeError("Host/Python mismatch")
    if pfm not in DIMS or fold not in FOLDS or seed not in (42,43):
        raise RuntimeError("Unknown matrix cell")
    if not run_id.startswith("20261105-r2-mil-twohost-"):
        raise RuntimeError("Invalid run identity")
    root=RUN_PARENT/run_id
    if not (root/"RUN_META.json").is_file():
        raise RuntimeError("Missing exclusive NAS run")
    stageroot=root/"hosts"/host/pfm/"stage"
    stage_complete=stageroot/"STAGE_COMPLETE.json"
    if not stage_complete.is_file(): raise RuntimeError("PFM not yet fully staged/verified")
    stage=json.loads(stage_complete.read_text())
    if stage["host"]!=host or stage["pfm"]!=pfm or stage["file_count"]!=1815:
        raise RuntimeError("Unexpected stage identity")
    if sha(stageroot/"stage_manifest.tsv")!=stage["sha256_manifest"]:
        raise RuntimeError("Stage manifest mutated")
    stage_records=pd.read_csv(stageroot/"stage_manifest.tsv",sep="\t").to_dict("records")
    if len(stage_records)!=1815:
        raise RuntimeError("Bad full PFM stage")
    csvpath=stageroot/f"{fold}_train_val_local.csv"
    x=pd.read_csv(csvpath)
    expected=pd.read_csv(PREP/"inputs"/pfm/fold/"train_val_only.csv")
    mapping={v["nas_path"]:v["local_path"] for v in stage_records}
    for group in ("train","val"):
        original=expected[f"{group}_slide_path"].dropna().tolist()
        local=x[f"{group}_slide_path"].dropna().tolist()
        if local != [mapping[z] for z in original]:
            raise RuntimeError("Corrupt or different fold mapping")
        if x[f"{group}_label"].dropna().astype(int).tolist()!=expected[f"{group}_label"].dropna().astype(int).tolist():
            raise RuntimeError("Split labels modified")
    if x.test_slide_path.notna().any():
        raise RuntimeError("Heldout entered training CSV")
    out=root/"hosts"/host/pfm/"train"/fold/f"seed{seed}"
    if out.exists(): raise FileExistsError("No-clobber training cell "+str(out))
    return Path(repo),root,csvpath,out,stage_records,stage


def run_one(args):
    import yaml
    from utils.yaml_utils import read_yaml
    from utils.model_utils import get_optimizer,get_scheduler,get_criterion
    from utils.process_utils import get_act
    from utils.general_utils import (set_global_seed,init_epoch_info_log,
                                     add_epoch_info_log,cal_is_stopping)
    from utils.loop_utils import train_loop,val_loop
    from utils.repro_utils import deterministic_requested,enable_full_determinism,seed_worker
    from modules.AB_MIL.ab_mil import AB_MIL

    repo,run_root,csvpath,out,records,stage=get_paths_and_params(
        args.host,args.pfm,args.fold,args.seed,args.run_id)
    cfgp=PREP/"config"/"training"/args.pfm/args.fold/f"seed{args.seed}.yaml"
    argsconf=read_yaml(str(cfgp))
    if (argsconf.General.MODEL_NAME!="AB_MIL" or argsconf.General.num_workers!=0
        or argsconf.General.num_epochs!=50
        or dict(argsconf.General.earlystop)!={"use":True,"patience":15,"metric":"macro_f1"}
        or argsconf.Dataset.balanced_sampler.use):
        raise RuntimeError("Config differs from frozen experiment")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Explicit GPU must be assigned")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count()!=1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one GPU")
    if torch.cuda.mem_get_info(0)[0] < 4*(2**30):
        raise RuntimeError("Insufficient free GPU memory")
    if args.fold in ("SL","XC") and args.pfm=="uni2" and args.seed==42:
        raise RuntimeError("This matrix cell is history reuse, not training")
    expected=pd.read_csv(csvpath)
    size=0
    for f in set(expected.train_slide_path.dropna())|set(expected.val_slide_path.dropna()):
        size+=os.stat(f).st_size
    ram=mem_available_bytes()
    if ram < size + HEADROOM:
        raise MemoryError(f"Strict preload preflight requires {size/2**30:.1f}GiB+30GiB, available={ram/2**30:.1f}GiB")
    if not args.execute:
        print(json.dumps({"preflight":"PASS_NOT_LAUNCHED","host":args.host,
                          "cell":[args.pfm,args.fold,args.seed],"staged_GiB":round(size/2**30,2),
                          "RAM_available_GiB":round(ram/2**30,2),"output":str(out)}))
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(exist_ok=False)
    (out/"epochs").mkdir(exist_ok=False)
    (out/"states").mkdir(exist_ok=False)
    # Source provenance: for scientifically relevant files record actual bytes
    files=[repo/x for x in ("train_mil.py","test_mil.py","process/AB_MIL/process_ab_mil.py",
         "utils/general_utils.py","utils/model_utils.py","utils/wsi_utils.py",
         "utils/loop_utils.py","utils/process_utils.py","utils/repro_utils.py",
         "utils/yaml_utils.py","modules/AB_MIL/ab_mil.py")]
    deps={str(x.relative_to(repo)):sha(x) for x in files}
    try: head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    except Exception: head="UNAVAILABLE"
    meta={"host":args.host,"hostname":socket.gethostname(),"pfm":args.pfm,
          "fold":args.fold,"seed":args.seed,"python":sys.executable,
          "torch":torch.__version__,"git_head":head,"deps":deps,
          "trainer_source":str(Path(__file__).resolve()),"trainer_sha256":sha(__file__),
          "config_source":str(cfgp),"config_sha256":sha(cfgp),
          "frozen_input_sha256":sha(PREP/"inputs"/args.pfm/args.fold/"train_val_only.csv"),
          "resume_from_bundle":str(args.resume_bundle) if args.resume_bundle else None,
          "training_csv_source":str(csvpath),"training_csv_sha256":sha(csvpath),
          "stage_manifest_sha256":stage["sha256_manifest"],
          "stage_uuid":stage["stage_uuid"],"cuda_visible":os.environ["CUDA_VISIBLE_DEVICES"],
          "strict_cpu_residency":True,"external_test":False,
          "source_guard_used_to_train":False,"started_utc":datetime.now(timezone.utc).isoformat()}
    save_json_exclusive(out/"LAUNCH.json",meta)
    tdf=pd.DataFrame(pd.read_csv(csvpath))
    tdf.to_csv(out/"training_fold_manifest.csv",index=False)
    # Upstream code reads same training CSV labels, but all tensor bytes come
    # solely from the new local stage. Original WSI stems stay intact.
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    gen=torch.Generator()
    gen.manual_seed(args.seed)
    if deterministic_requested():
        enable_full_determinism(args.seed)
        worker_init_fn=seed_worker
    else:
        set_global_seed(args.seed)
        worker_init_fn=None
    train_ds=StrictResident(csvpath,"train",records,DIMS[args.pfm])
    val_ds=StrictResident(csvpath,"val",records,DIMS[args.pfm])
    sampler=RecordingSampler(train_ds,generator=gen)
    tr_loader=DataLoader(train_ds,batch_size=1,num_workers=0,sampler=sampler,
                         generator=gen,worker_init_fn=worker_init_fn)
    va_loader=DataLoader(val_ds,batch_size=1,num_workers=0,shuffle=False)
    torch.cuda.set_device(0)
    device=torch.device("cuda:0")
    mil_model=AB_MIL(L=argsconf.Model.L,D=argsconf.Model.D,
                    num_classes=argsconf.General.num_classes,dropout=argsconf.Model.dropout,
                    act=get_act(argsconf.Model.act),in_dim=argsconf.Model.in_dim).to(device)
    optimizer,base_lr=get_optimizer(argsconf,mil_model)
    scheduler,warmup=get_scheduler(argsconf,optimizer,base_lr)
    criterion=get_criterion(argsconf.Model.criterion)
    warmup_epochs=int(argsconf.Model.scheduler.warmup)
    logs=init_epoch_info_log()
    best_epoch=1;best_metric=0.;best_loss=float("inf")
    ema_state=None;completed_epochs=[];global_step=0
    epoch_paths={}
    start_epoch=0
    if args.resume_bundle:
        prior=Path(args.resume_bundle)
        if not prior.is_absolute() or not str(prior.resolve()).startswith(str(RUN_PARENT)+"/"):
            raise RuntimeError("Resume must come from a governed NAS provenance run")
        b=torch.load(prior,map_location=device,weights_only=False)
        if (b.get("version")!="r2_resume_v2" or b.get("source_input_sha256")!=meta["frozen_input_sha256"]
            or b.get("dependency_sha256")!=deps or b.get("stopped")):
            raise RuntimeError("Resume bundle incompatible, already complete or source changed")
        start_epoch=int(b["epoch"])
        if start_epoch<=0 or start_epoch>=int(argsconf.General.num_epochs):
            raise RuntimeError("Resume epoch out of range")
        for k,path in b["history_epoch_paths"].items():
            ep=int(k)
            q=Path(path)
            if ep>start_epoch or not q.is_file():
                raise RuntimeError("Missing prior epoch snapshot for resume")
            epoch_paths[ep]=q
        if set(epoch_paths)!=set(range(1,start_epoch+1)):
            raise RuntimeError("Incomplete prior epoch lineage")
        mil_model.load_state_dict(b["model_state_dict"])
        optimizer.load_state_dict(b["optimizer_state_dict"])
        scheduler.load_state_dict(b["scheduler_state_dict"])
        warmup.load_state_dict(b["warmup_scheduler_state_dict"])
        best_epoch=int(b["best_epoch"])
        best_metric=float(b["best_val_metric"])
        best_loss=float(b["best_val_loss"])
        logs=b["epoch_info_log"]
        global_step=int(b["global_step"])
        completed_epochs=list(range(1,start_epoch+1))
        ema_state=b["ema_state_dict"]
        # Restore only after model/optimizer/schedulers and all resident data are built.
        random.setstate(b["python_rng_state"])
        np.random.set_state(b["numpy_rng_state"])
        torch.set_rng_state(b["torch_cpu_rng_state"])
        torch.cuda.set_rng_state_all(b["torch_cuda_rng_states"])
        gen.set_state(b["loader_generator_state"])
        print("RESUME_VERIFIED",start_epoch,"from",str(prior),flush=True)
    for epoch_idx in range(start_epoch,argsconf.General.num_epochs):
        sched=warmup if epoch_idx+1<=warmup_epochs else scheduler
        train_loss,duration=train_loop(device,mil_model,tr_loader,criterion,optimizer,sched)
        if len(sampler.last_order)!=len(train_ds):
            raise RuntimeError("Not all training examples consumed")
        val_loss,val_metrics,_=val_loop(device,2,mil_model,va_loader,criterion)
        add_epoch_info_log(logs,epoch_idx,train_loss,val_loss,None,val_metrics,None)
        current_metric=float(val_metrics["macro_f1"])
        better=current_metric>best_metric+1e-9
        tie=abs(current_metric-best_metric)<=1e-9
        if better or (tie and val_loss<best_loss-1e-9):
            best_epoch=epoch_idx+1
            best_metric=current_metric
            best_loss=float(val_loss)
        raw={k:v.detach().cpu().clone() for k,v in mil_model.state_dict().items()}
        if ema_state is None:
            ema_state={k:v.float().clone() if torch.is_floating_point(v) else v.clone()
                       for k,v in raw.items()}
        else:
            for k,v in raw.items():
                if torch.is_floating_point(v):
                    ema_state[k].mul_(.9).add_(v.float(),alpha=.1)
                else:ema_state[k]=v.clone()
        ep=epoch_idx+1
        global_step+=len(tr_loader)
        stop=bool(cal_is_stopping(argsconf,logs,"Train_Val"))
        raw_path=out/"epochs"/f"epoch_{ep:03d}_model.pth"
        bundle_path=out/"epochs"/f"epoch_{ep:03d}_training.pth"
        save_torch_exclusive(raw_path,raw)
        epoch_paths[ep]=raw_path
        full={"version":"r2_resume_v2","epoch":ep,"model_state_dict":raw,
             "history_epoch_paths":{str(k):str(v) for k,v in epoch_paths.items()},
             "output_dir":str(out),
             "source_input_sha256":meta["frozen_input_sha256"],
             "optimizer_state_dict":optimizer.state_dict(),
             "scheduler_state_dict":scheduler.state_dict() if scheduler else None,
             "warmup_scheduler_state_dict":warmup.state_dict() if warmup else None,
             "amp_scaler_state_dict":None,"amp_used":False,
             "python_rng_state":random.getstate(),
             "numpy_rng_state":np.random.get_state(),
             "torch_cpu_rng_state":torch.get_rng_state(),
             "torch_cuda_rng_states":torch.cuda.get_rng_state_all(),
             "loader_generator_state":gen.get_state(),"consumed_sampler_indices":list(sampler.last_order),
             "global_step":global_step,"best_epoch":best_epoch,
             "best_val_metric":best_metric,"best_val_loss":best_loss,
             "epoch_info_log":logs,"ema_state_dict":ema_state,
             "stopped":stop,"dataset_csv_sha256":meta["training_csv_sha256"],
             "dependency_sha256":deps}
        save_torch_exclusive(bundle_path,full)
        save_json_exclusive(out/"epochs"/f"epoch_{ep:03d}_manifest.json",
                {"epoch":ep,"model_sha256":sha(raw_path),"bundle_sha256":sha(bundle_path),
                 "selection_macro_f1":current_metric,"selection_ce":float(val_loss),
                 "is_best":ep==best_epoch,"train_loss":float(train_loss),
                 "time_s":round(duration,2),"stopped":stop})
        completed_epochs.append(ep)
        print("R2_EPOCH_RETAINED",args.host,args.pfm,args.fold,args.seed,ep,
              "macroF1",round(current_metric,5),"best_epoch",best_epoch,
              "stop",stop,flush=True)
        if stop:break
    last_ep=completed_epochs[-1]
    rawpaths=[epoch_paths[ep] for ep in completed_epochs[-5:]]
    last={k:v.detach().cpu().clone() for k,v in mil_model.state_dict().items()}
    raws=[torch.load(p,map_location="cpu",weights_only=True) for p in rawpaths]
    swa={}
    for k,v in last.items():
        if torch.is_floating_point(v):
            swa[k]=torch.stack([item[k].float() for item in raws],dim=0).mean(dim=0)
        else:swa[k]=v.clone()
    save_torch_exclusive(out/"states"/"swa_last5.pth",swa)
    save_torch_exclusive(out/"states"/"ema_epoch_0p9.pth",ema_state)
    pd.DataFrame(logs).to_csv(out/"selection_epochs.csv",index=False)
    mapping={"best":{"epoch":best_epoch,"path":str(epoch_paths[best_epoch])},
             "last":{"epoch":last_ep,"path":str(epoch_paths[last_ep])},
             "swa_last5":{"epochs":completed_epochs[-5:],"path":str(out/"states"/"swa_last5.pth")},
             "ema_epoch_0p9":{"epochs":completed_epochs,"beta":0.9,
                             "path":str(out/"states"/"ema_epoch_0p9.pth")}}
    for v in mapping.values():v["sha256"]=sha(v["path"])
    save_json_exclusive(out/"CANDIDATES.json",mapping)
    save_json_exclusive(out/"TRAIN_COMPLETE.json",
        {"status":"TRAIN_COMPLETE_PENDING_REPLAY_QA","host":args.host,"pfm":args.pfm,
         "fold":args.fold,"seed":args.seed,"epochs":len(completed_epochs),
         "best_epoch":best_epoch,"last_epoch":last_ep,
         "source_selection_completed":False,"heldout_completed":False,
         "stage_cache_must_remain":True,
         "finished_utc":datetime.now(timezone.utc).isoformat()})
    print("TRAIN_COMPLETE_PENDING_REPLAY_QA",args.host,args.pfm,args.fold,args.seed,flush=True)


def self_test():
    # Tiny deterministic model/sampler/scheduler/RNG epoch-boundary resume smoke.
    from torch.utils.data import TensorDataset
    def create(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        x=torch.arange(48,dtype=torch.float32).reshape(12,4)/41.
        y=(torch.arange(12)%2).long()
        d=TensorDataset(x,y)
        g=torch.Generator().manual_seed(seed)
        sampler=RecordingSampler(d,generator=g)
        loader=DataLoader(d,batch_size=3,sampler=sampler,generator=g,num_workers=0)
        m=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.Dropout(.2),torch.nn.Linear(6,2))
        opt=torch.optim.Adam(m.parameters(),lr=.01)
        sch=torch.optim.lr_scheduler.StepLR(opt,step_size=3,gamma=.9)
        return loader,g,sampler,m,opt,sch
    def epoch(loader,m,opt,sch):
        m.train()
        for x,y in loader:
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(x),y)
            loss.backward();opt.step()
        sch.step()
    loader,g,sampler,m,opt,sch=create(7)
    epoch(loader,m,opt,sch)
    saved={"model":copy.deepcopy(m.state_dict()),"opt":copy.deepcopy(opt.state_dict()),
           "scheduler":copy.deepcopy(sch.state_dict()),
           "py_rng":random.getstate(),"np_rng":np.random.get_state(),
           "torch_rng":torch.get_rng_state(),"gen":g.get_state()}
    epoch(loader,m,opt,sch)
    expected={k:v.detach().clone() for k,v in m.state_dict().items()}
    loader2,g2,sampler2,m2,opt2,sch2=create(7)
    m2.load_state_dict(saved["model"]);opt2.load_state_dict(saved["opt"])
    sch2.load_state_dict(saved["scheduler"])
    random.setstate(saved["py_rng"]);np.random.set_state(saved["np_rng"])
    torch.set_rng_state(saved["torch_rng"]);g2.set_state(saved["gen"])
    epoch(loader2,m2,opt2,sch2)
    assert all(torch.equal(v,m2.state_dict()[k]) for k,v in expected.items())
    print(json.dumps({"self_test":"PASS_CPU_EPOCH_BOUNDARY_RESTORE",
                      "torch":torch.__version__,"checkpoint_components":
                       ["model","optimizer","scheduler","rng","loader_generator"],
                       "training_started":False}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--host",choices=("138","195"))
    p.add_argument("--pfm",choices=tuple(DIMS))
    p.add_argument("--fold",choices=FOLDS)
    p.add_argument("--seed",type=int,choices=(42,43))
    p.add_argument("--prep", type=Path, default=PREP, help="Frozen guard-preparation directory")
    p.add_argument("--run-id")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--resume-bundle",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    PREP = args.prep.resolve(strict=True)
    if args.self_test:self_test()
    else:
        if not all((args.host,args.pfm,args.fold,args.seed,args.run_id)):
            raise RuntimeError("Missing formal run fields")
        hostrepo={"138":"/NAS3/lbliao/Code-138/MIL_BASELINE",
                  "195":"/NAS2/Data1/lbliao/Code-195/MIL_BASELINE"}[args.host]
        sys.path.insert(0,hostrepo)
        run_one(args)
