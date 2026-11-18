"""Fold-local resident training features and sampler."""
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, RandomSampler


HEADROOM = 30 * (2**30)


def mem_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])*1024
    raise RuntimeError("MemAvailable unavailable")

class StrictResident(Dataset):
    def __init__(self, csv_path, role, stage_manifest, dim):
        if role not in ("train","val"):
            raise ValueError("No test preload permitted")
        df=pd.read_csv(csv_path)
        self.paths=df[f"{role}_slide_path"].dropna().tolist()
        self.labels=df[f"{role}_label"].dropna().astype(int).tolist()
        if len(self.paths)!=len(self.labels) or not self.paths or len(set(self.paths))!=len(self.paths):
            raise RuntimeError("Bad dataset column count or duplicate paths")
        # Stage manifest must already be entirely verified, and the current fold's
        # only eligible paths must belong to the manifest's per-PFM whitelist.
        trusted={r["local_path"]:r for r in stage_manifest}
        if not set(self.paths)<=set(trusted):
            raise RuntimeError("Fold CSV references unstaged, invalid or test feature")
        if not all(y in (0,1) for y in self.labels):
            raise RuntimeError("Nonbinary label")
        self.data=[]
        self.loaded_bytes=0
        # NB: this intentionally fails closed instead of the framework's partial
        # preload+on-demand fallback. Load sequentially to avoid memory races.
        for idx,p in enumerate(self.paths):
            r=trusted[p];q=Path(p)
            st=q.lstat()
            if not q.is_file() or q.is_symlink() or st.st_nlink!=1:
                raise RuntimeError("Staged input file not owned regular copy")
            if st.st_ino != int(r["local_inode"]) or st.st_dev!=int(r["local_device"]):
                raise RuntimeError("Local stage file identity changed")
            if st.st_size!=int(r["source_bytes"]):
                raise RuntimeError("Stage file byte mismatch")
            if mem_available_bytes()<HEADROOM+512*1024*1024:
                raise MemoryError("Insufficient RAM headroom for strict resident preload")
            item=torch.load(p,map_location="cpu",weights_only=True)
            if isinstance(item,dict):
                item=item["feats"] if "feats" in item else item["features"]
            if not isinstance(item,torch.Tensor):
                raise TypeError("Expected tensor feature "+p)
            if item.dim()==3: item=item.squeeze(0)
            if item.dim()!=2 or item.shape[1]!=dim or not torch.isfinite(item).all():
                raise RuntimeError("Wrong feature dimension or nonfinite "+p)
            item=item.contiguous()
            self.loaded_bytes+=int(item.numel()*item.element_size())
            # Use the original frozen NAS filename rather than stage SHA prefix.
            self.data.append((item,torch.tensor(int(self.labels[idx])),
                              Path(r["nas_path"]).stem))
        if len(self.data)!=len(self.paths):
            raise RuntimeError("Partial preload prohibited")
        print("STRICT_RESIDENT_PASS",role,len(self.data),
              "tensor_GiB",round(self.loaded_bytes/2**30,3),
              "RAM_available_GiB",round(mem_available_bytes()/2**30,2),flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i]

class RecordingSampler(RandomSampler):
    def __iter__(self):
        self.last_order=[]
        for idx in super().__iter__():
            self.last_order.append(int(idx))
            yield idx
