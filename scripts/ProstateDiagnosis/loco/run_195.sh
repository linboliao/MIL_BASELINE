#!/usr/bin/env bash
# 8-PFM robustness matrix — SERVER 195 — encoders: conch, uni2
#
#   ssh 195
#   tmux new -s loco_matrix
#   bash scripts/ProstateDiagnosis/loco/run_195.sh        # from the repo root
#       (or: bash /NAS2/Data1/lbliao/Code-195/MIL_BASELINE/scripts/ProstateDiagnosis/loco/run_195.sh)
#
# ~3-4 h. Check `nvidia-smi` first: needs GPU 0-3 free (4 & 7 are another user's).
# To use different GPUs: LOCO_GPU_BASE=<n> bash .../run_195.sh   (needs n..n+3 all free)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export LOCO_CACHE=${LOCO_CACHE:-/data2/lbliao/loco_cache}
export LOCO_PYTHON=${LOCO_PYTHON:-/home/lbliao/anaconda3/envs/clam/bin/python}
export LOCO_LD_LIBRARY_PATH=${LOCO_LD_LIBRARY_PATH:-/home/lbliao/anaconda3/envs/clam/lib}
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/lbliao/mil_runs/loco}
# export LOCO_MODES="internal type fivesite"   # default; drop fivesite to go faster

mkdir -p "$LOCO_LOGD"
bash "$HERE/loco_run.sh" conch uni2 \
  2>&1 | tee "$LOCO_LOGD/RUN_matrix_195_$(date +%Y%m%d_%H%M).log"
