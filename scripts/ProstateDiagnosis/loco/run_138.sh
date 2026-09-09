#!/usr/bin/env bash
# 8-PFM robustness matrix — SERVER 138 — env preset for the PrePATH conda + GPU 0-3.
#
#   ssh 138 ; tmux new -s loco_matrix
#   bash scripts/ProstateDiagnosis/loco/run_138.sh               # default: gigapath gpfm mstar
#   bash scripts/ProstateDiagnosis/loco/run_138.sh gigapath gpfm # or name the encoder(s)
#
# All 8x RTX 3090 free; folds use GPU LOCO_GPU_BASE..+3 (default 0-3).
# 138's driver (455 / CUDA 11.1) does NOT support expandable_segments — leave
# LOCO_ALLOC_CONF unset (this script does).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export LOCO_CACHE=${LOCO_CACHE:-/data14/lbliao/loco_cache}
export LOCO_PYTHON=${LOCO_PYTHON:-/data12/jing/anaconda3/envs/PrePATH/bin/python}
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/jing/mil_runs/loco}
# 138: PrePATH env needs no LD_LIBRARY_PATH override, and its driver (455 / CUDA 11.1)
# does NOT support PYTORCH_CUDA_ALLOC_CONF=expandable_segments — leave LOCO_ALLOC_CONF unset.
# export LOCO_MODES="internal type fivesite"   # default

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(gigapath gpfm mstar)
mkdir -p "$LOCO_LOGD"
echo "models: ${MODELS[*]}"
bash "$HERE/loco_run.sh" "${MODELS[@]}" \
  2>&1 | tee "$LOCO_LOGD/RUN_matrix_138_$(date +%Y%m%d_%H%M).log"
