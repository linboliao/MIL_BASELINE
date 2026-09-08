#!/usr/bin/env bash
# 8-PFM robustness matrix — SERVER 138 — encoders: gigapath, gpfm, mstar
#
#   ssh 138
#   tmux new -s loco_matrix
#   bash scripts/ProstateDiagnosis/loco/run_138.sh        # from the repo root
#       (or: bash /NAS3/lbliao/Code-138/MIL_BASELINE/scripts/ProstateDiagnosis/loco/run_138.sh)
#
# ~5-6 h sequential. All 8x RTX 3090 are free; folds use GPU 0-3.
# Faster (concurrent) alternative in two tmux panes:
#   LOCO_CACHE=/data14/lbliao/loco_cache_a LOCO_GPU_BASE=0 bash .../loco_run.sh gigapath
#   LOCO_CACHE=/data14/lbliao/loco_cache_b LOCO_GPU_BASE=4 bash .../loco_run.sh gpfm mstar
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export LOCO_CACHE=${LOCO_CACHE:-/data14/lbliao/loco_cache}
export LOCO_PYTHON=${LOCO_PYTHON:-/data12/jing/anaconda3/envs/PrePATH/bin/python}
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/jing/mil_runs/loco}
# 138's PrePATH env needs no LD_LIBRARY_PATH override.
# export LOCO_MODES="internal type fivesite"   # default

mkdir -p "$LOCO_LOGD"
bash "$HERE/loco_run.sh" gigapath gpfm mstar \
  2>&1 | tee "$LOCO_LOGD/RUN_matrix_138_$(date +%Y%m%d_%H%M).log"
