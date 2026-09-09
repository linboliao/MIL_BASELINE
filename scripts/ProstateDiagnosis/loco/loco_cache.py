"""Stage all feature .pt for one encoder from NAS to $LOCO_CACHE.
Covers dev_clean + internal_test_clean + external 301 + external ynzl
(~2520 slides) so both the internal-LOCO and 5-site-LOCO builders can use it.

Two modes:
  default        torch.load -> .half() -> torch.save  (fp16; ~1/2 disk + ~1/2
                 per-epoch local read, but .half()/pickle hold the GIL, so on a
                 CPU-thrashed host neither the NAS nor the NIC saturates)
  LOCO_RAW=1     shutil.copy2 the .pt verbatim (fp32; pure I/O, no CPU, no GIL
                 -> fast even at load 80; costs ~2x local disk)

Uses PROCESSES not threads so the fp16 path parallelizes deserialize+convert
across cores.  --workers defaults to a modest CPU count.
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch

from loco_common import NAS, CACHE, pooled_cohort

torch.set_num_threads(1)
RAW = bool(os.environ.get("LOCO_RAW"))


def one(job):
    src, dst = job
    if os.path.exists(dst):
        return 0
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        if RAW:
            shutil.copy2(src, tmp)
        else:
            torch.save(torch.load(src, map_location="cpu", weights_only=True).half(), tmp)
        os.rename(tmp, dst)
        return 1
    except Exception as e:
        print(f"FAIL {src}: {e}")
        return -1


def main(model, workers):
    df = pooled_cohort(include_external=True)
    jobs = [(f"{NAS}/{r.feat_dir}/feat_0_224/pt_files/{model}/{r.stem}.pt",
             f"{CACHE}/{r.feat_dir}/feat_0_224/pt_files/{model}/{r.stem}.pt")
            for r in df.itertuples()]
    print(f"{model}: {len(jobs)} files -> {CACHE}  ({workers} processes, "
          f"{'RAW fp32 copy' if RAW else 'fp16 convert'})")
    d = s = f = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4), 1):
            d += r == 1; s += r == 0; f += r == -1
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}  new={d} skip={s} fail={f}")
    print(f"done: new={d} skip={s} fail={f}")
    if f:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 8)))
    a = ap.parse_args()
    main(a.model, a.workers)
