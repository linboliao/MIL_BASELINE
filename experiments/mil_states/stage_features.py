"""Stage exactly the TRAIN+SELECTION union of a PFM once per host, five folds.
No deletion capability. All attempts create brand-new exclusive local and NAS paths.
Use the project-prescribed interpreter for the selected server.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

import pandas as pd

NAS = Path("/NAS145/linboliao/Experiments/ProstateDiagnosis")
RUN_PARENT = NAS / "provenance/generalization/r2-mil-retrain"
PREP = RUN_PARENT / "20261105-guard-prep-r1"
CACHE_PARENT = Path("/data2/lbliao/r2_mil_stage")
PFMS = {"mstar", "uni2", "h-optimus-1", "virchow2"}
FOLDS = ("SL", "XC", "CNB", "RP", "TURP")
SOURCE_PREFIX = (NAS / "../../linboliao/Data/迈新生物_特征/ProstateDiagnosis").resolve()
SOURCES_LITERAL_PREFIX = "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis/"
LIMIT_FREE_BYTES = 30 * (2**30)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def exclusive_text(file, content):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("x", encoding="utf-8") as f:
        f.write(content)


def records(pfm):
    refs = set()
    counts = {}
    folded = {}
    for fold in FOLDS:
        q = PREP / "inputs" / pfm / fold / "train_val_only.csv"
        x = pd.read_csv(q)
        for group in ("train", "val"):
            pts = x[group + "_slide_path"].dropna().tolist()
            if len(pts) != len(set(pts)):
                raise RuntimeError(f"duplicate WSI in {pfm}/{fold}/{group}")
            folded[fold,group] = pts
            refs.update(pts)
            counts[f"{fold}_{group}"] = len(pts)
        if set(folded[fold,"train"]) & set(folded[fold,"val"]):
            raise RuntimeError("Train/Selection overlap")
        if x.test_slide_path.notna().any():
            raise RuntimeError("test paths not empty")
    if len(refs) != 1815:
        raise RuntimeError(f"Unexpected union size {len(refs)}")
    data = []
    for src in sorted(refs):
        if not src.startswith(SOURCES_LITERAL_PREFIX) or f"/pt_files/{pfm}/" not in src:
            raise RuntimeError("Path outside expected feature root "+src)
        if not src.endswith(".pt") or Path(src).is_symlink():
            raise RuntimeError("Unexpected/nonregular feature "+src)
        s = Path(src).stat()
        if not os.path.isfile(src) or s.st_size<=0:
            raise RuntimeError("Empty/nonregular source "+src)
        # Filename includes canonical source-path digest; no collisions across subroots.
        dest_name = hashlib.sha256(src.encode("utf-8")).hexdigest() + "_" + Path(src).name
        data.append({"nas_path":src, "name":dest_name, "source_bytes":s.st_size,
                     "source_mtime_ns":s.st_mtime_ns})
    if len({x["name"] for x in data}) != len(data):
        raise RuntimeError("Hash/path collision")
    return data, counts


def run(args):
    who = {"server94":("138","/data12/jing/anaconda3/envs/PrePATH/bin/python"),
           "ubuntu":("195","/home/lbliao/anaconda3/envs/clam/bin/python")}
    hostname = socket.gethostname()
    if hostname not in who:
        raise RuntimeError("Not an approved host: "+hostname)
    host, python = who[hostname]
    if sys.executable != python:
        raise RuntimeError("Wrong host Python "+sys.executable)
    if args.host != host or args.pfm not in PFMS:
        raise RuntimeError("Host/PFM mismatch")
    if not args.run_id.startswith("20261105-r2-mil-twohost-"):
        raise RuntimeError("Unexpected run id")
    run_root = RUN_PARENT / args.run_id
    if not run_root.is_dir() or not (run_root/"RUN_META.json").is_file():
        raise RuntimeError("New NAS run root must be precreated")
    out = run_root / "hosts" / host / args.pfm / "stage"
    if out.exists():
        raise FileExistsError("Stage output already exists; no retry in place "+str(out))
    rs, counts = records(args.pfm)
    size = sum(x["source_bytes"] for x in rs)
    free = shutil.disk_usage("/data2").free
    if free < size + LIMIT_FREE_BYTES:
        raise RuntimeError("Insufficient disk for whole-PFM stage (require 30 GiB reserve)")
    if args.dry_run:
        print(json.dumps({"stage":"dry_run_only","host":host,"pfm":args.pfm,
                          "files":len(rs),"bytes":size,"disk_free":free,
                          "roles":counts,"write":False},ensure_ascii=False))
        return
    # A stage is uniquely identified and NEVER shared between attempts/hosts/PFMs.
    stage_uuid = uuid.uuid4().hex
    stage_base = CACHE_PARENT / args.run_id / args.pfm / stage_uuid
    if stage_base.exists():
        raise FileExistsError(stage_base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    stage_base.mkdir(parents=True, exist_ok=False)
    target_dir = stage_base / "features" / args.pfm
    target_dir.mkdir(parents=True, exist_ok=False)
    owner={"run_id":args.run_id,"host":host,"hostname":hostname,"user":os.getlogin() if os.isatty(0) else os.environ.get("USER"),
           "pfm":args.pfm,"stage_uuid":stage_uuid,"stage_root":str(stage_base),
           "stage_features_dir":str(target_dir),
           "files_expected":len(rs),"bytes_expected":size,
           "creation_utc":datetime.now(timezone.utc).isoformat(),
           "status":"COPY_IN_PROGRESS","permission":"only_this_stage_managed_pt_files"}
    exclusive_text(stage_base/"owner.json",json.dumps(owner,ensure_ascii=False,indent=2)+"\n")
    exclusive_text(out/"owner.json",json.dumps(owner,ensure_ascii=False,indent=2)+"\n")
    for item in rs:
        item["local_path"]=str(target_dir / item["name"])
    planned=out/"planned_manifest.tsv"
    with planned.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),delimiter="\t")
        w.writeheader();w.writerows(rs)
    print("STAGE_CREATED",stage_base,"total",len(rs),"bytes",size,flush=True)
    complete=[]
    for i,r in enumerate(rs,1):
        src=Path(r["nas_path"])
        dst=Path(r["local_path"])
        s_before=src.stat()
        if (s_before.st_size,s_before.st_mtime_ns)!=(r["source_bytes"],r["source_mtime_ns"]):
            raise RuntimeError("Source changed before stage "+str(src))
        src_h=hashlib.sha256()
        with src.open("rb") as fsrc, dst.open("xb") as fdst:
            for chunk in iter(lambda:fsrc.read(4*1024*1024), b""):
                src_h.update(chunk); fdst.write(chunk)
            fdst.flush();os.fsync(fdst.fileno())
        s_after=src.stat();dst_s=dst.lstat()
        if ((s_after.st_size,s_after.st_mtime_ns)!=(r["source_bytes"],r["source_mtime_ns"])
           or dst_s.st_size!=r["source_bytes"] or dst_s.st_nlink!=1
           or not dst.is_file() or dst.is_symlink()):
            raise RuntimeError("Source changed or destination unexpected: "+str(dst))
        dest_h=hash_file(dst)
        if dest_h!=src_h.hexdigest():
            raise RuntimeError("Stage sha256 mismatch: "+str(dst))
        complete.append({**r,"sha256":dest_h,
                         "local_device":dst_s.st_dev,"local_inode":dst_s.st_ino,
                         "local_nlink":dst_s.st_nlink})
        if i%25==0 or i==len(rs):
            print("STAGE_PROGRESS",host,args.pfm,i,len(rs),flush=True)
        if shutil.disk_usage("/data2").free < LIMIT_FREE_BYTES:
            raise RuntimeError("Disk reserve exhausted during stage")
    with (out/"stage_manifest.tsv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(complete[0]),delimiter="\t")
        w.writeheader();w.writerows(complete)
    # Fold-specific training indexes use exactly the current fold's white list.
    mapping={r["nas_path"]:r["local_path"] for r in complete}
    for fold in FOLDS:
        x=pd.read_csv(PREP/"inputs"/args.pfm/fold/"train_val_only.csv")
        for col in ("train_slide_path","val_slide_path"):
            x[col]=x[col].map(lambda path:mapping[path] if pd.notna(path) else float("nan"))
            assert x[col].dropna().map(str).str.startswith(str(target_dir)+"/").all()
        if x.test_slide_path.notna().any():
            raise RuntimeError("Test data leakage")
        with (out/f"{fold}_train_val_local.csv").open("x",newline="") as f:
            x.to_csv(f,index=False)
    summary={"stage_status":"STAGED_HASH_VERIFIED_NOT_TRAINED",
             "host":host,"pfm":args.pfm,"run_id":args.run_id,"stage_uuid":stage_uuid,
             "file_count":len(complete),"bytes":size,"cache_root":str(stage_base),
             "sha256_manifest":hash_file(out/"stage_manifest.tsv"),
             "source_fold_counts":counts,"training_started":False,
             "finished_utc":datetime.now(timezone.utc).isoformat()}
    exclusive_text(out/"STAGE_COMPLETE.json",json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("STAGE_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",choices=("138","195"),required=True)
    ap.add_argument("--pfm",choices=tuple(sorted(PFMS)),required=True)
    ap.add_argument("--prep", type=Path, default=PREP, help="Frozen guard-preparation directory")
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--dry-run",action="store_true")
    opts=ap.parse_args()
    PREP = opts.prep.resolve(strict=True)
    run(opts)
