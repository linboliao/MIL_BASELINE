#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/CONCH.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/h-optimus-1.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/mstar.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/omiclip.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/UNI.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/UNI2.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/PFM/virchow2.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/Mag/10x.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/Mag/5x.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/Stains/Macenko.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/Stains/Reinhard.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/Stains/Vahadane.yaml
