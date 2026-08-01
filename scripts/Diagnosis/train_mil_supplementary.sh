#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/MEAN_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/MAX_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/DTFD_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/RRT_MIL.yaml

