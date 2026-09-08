"""Stage all feature .pt for one encoder from NAS to /data2 as fp16.
Covers dev_clean + internal_test_clean + external 301 + external ynzl
(~2520 slides) so both the internal-LOCO and 5-site-LOCO builders can use it."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from loco_common import NAS, CACHE, pooled_cohort


def one(job):
    src, dst = job
    if os.path.exists(dst):
        return 0
    try:
        t = torch.load(src, map_location="cpu", weights_only=True).half()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        torch.save(t, tmp)
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
    print(f"{model}: {len(jobs)} files -> {CACHE}  ({workers} workers)")
    d = s = f = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            d += r == 1; s += r == 0; f += r == -1
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)}  new={d} skip={s} fail={f}")
    print(f"done: new={d} skip={s} fail={f}")
    if f:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    main(a.model, a.workers)
