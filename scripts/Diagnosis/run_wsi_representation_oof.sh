#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/CONCH.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/h-optimus-1.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/mstar.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/omiclip.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/UNI.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/UNI2.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/PFM/virchow2.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Mag/20x.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Mag/10x.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Mag/5x.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Stains/Macenko.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Stains/Reinhard.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/oof_mil.py --yaml-path configs/Diagnosis/Stains/Vahadane.yaml --device cuda:0
