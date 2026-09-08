"""Shared: build the pooled ProstateDiagnosis cohort with site + type labels.

Paths are repo-relative + env-driven so the pipeline runs unchanged on 138 / 195:
  repo root  <- Path(__file__).parents[3]  (scripts/ProstateDiagnosis/loco/..)
  $PROSTATE_FEAT_ROOT  shared NAS feature root  (default: the NAS145 迈新生物_特征 path)
  $LOCO_CACHE          per-server local-disk scratch for the fp16 feature cache
                       (195: /data2/lbliao/loco_cache, 138: /data14/lbliao/loco_cache)
"""
import os
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
DS = str(_REPO / "datasets" / "ProstateDiagnosis")
NAS = os.environ.get("PROSTATE_FEAT_ROOT",
                     "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")
CACHE = os.path.join(os.environ.get("LOCO_CACHE", "/tmp/loco_cache"), "ProstateDiagnosis")
POOL_DIR = {"dev": "MIL训练数据", "oldtest": "MIL测试数据", "ext_sl": "MIL外部测试"}

# mode -> (held-out column, list of held-out values [each becomes a fold],
#          include external cohorts in the pool?)
MODE_CFG = {
    "internal": ("center", ["省立", "新昌"], False),           # 迈新 always in train
    "fivesite": ("center", ["省立", "新昌", "301", "云南肿瘤"], True),
    "type":     ("type",   ["CNB", "RP", "TURP"], False),
}


def pooled_cohort(include_external):
    frames = []
    for fn in ("dev_clean.csv", "internal_test_clean.csv"):
        df = pd.read_csv(f"{DS}/{fn}", dtype={"patient_id": str}).copy()
        df["feat_dir"] = df["pool"].map(POOL_DIR)
        frames.append(df[["filename", "patient_id", "label", "center", "type", "feat_dir"]])
    if include_external:
        for fn, site in (("external_test_301.csv", "301"),
                         ("external_test_ynzl.csv", "云南肿瘤")):
            df = pd.read_csv(f"{DS}/{fn}", dtype={"patient_id": str})
            df = df[df["pool"].astype(str).str.startswith("ext")].copy()
            df["center"] = site
            df["feat_dir"] = "MIL外部测试"
            frames.append(df[["filename", "patient_id", "label", "center", "type", "feat_dir"]])
    df = pd.concat(frames, ignore_index=True)
    df["stem"] = df["filename"].map(lambda x: os.path.splitext(str(x))[0])
    df = df.drop_duplicates("stem").reset_index(drop=True)
    return df


def feat_path(row, model, root=CACHE):
    return f"{root}/{row['feat_dir']}/feat_0_224/pt_files/{model}/{row['stem']}.pt"
