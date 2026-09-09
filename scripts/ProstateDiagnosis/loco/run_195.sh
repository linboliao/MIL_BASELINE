#!/usr/bin/env bash
# 8-PFM robustness matrix — SERVER 195 — env preset for the clam conda + GPU 0-3.
#
#   ssh 195 ; tmux new -s loco_matrix
#   bash scripts/ProstateDiagnosis/loco/run_195.sh            # default: conch uni2
#   bash scripts/ProstateDiagnosis/loco/run_195.sh mstar      # or name the encoder(s)
#
# Check `nvidia-smi` first: folds use GPU LOCO_GPU_BASE..+3 (default 0-3; 4 & 7 are
# another user's). Different GPUs:  LOCO_GPU_BASE=<n> bash .../run_195.sh <model>...
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export LOCO_CACHE=${LOCO_CACHE:-/data2/lbliao/loco_cache}
export LOCO_PYTHON=${LOCO_PYTHON:-/home/lbliao/anaconda3/envs/clam/bin/python}
export LOCO_LD_LIBRARY_PATH=${LOCO_LD_LIBRARY_PATH:-/home/lbliao/anaconda3/envs/clam/lib}
export LOCO_ALLOC_CONF=${LOCO_ALLOC_CONF:-expandable_segments:True}   # 195 driver supports it
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/lbliao/mil_runs/loco}
# export LOCO_MODES="internal type fivesite"   # default; drop fivesite to go faster

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(conch uni2)
mkdir -p "$LOCO_LOGD"
echo "models: ${MODELS[*]}"
bash "$HERE/loco_run.sh" "${MODELS[@]}" \
  2>&1 | tee "$LOCO_LOGD/RUN_matrix_195_$(date +%Y%m%d_%H%M).log"
