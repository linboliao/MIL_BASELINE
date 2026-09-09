#!/usr/bin/env bash
# PSIR rigorous re-check — SERVER 195 (clam conda).
#
#   ssh 195 ; tmux new -s psir_recheck
#   bash scripts/ProstateDiagnosis/psir/run_recheck_195.sh h-optimus-1
#
# Per encoder ~2-3 h: stage -> bare (5 CV) -> psir (5 proj x 5 CV) -> shuf (5x5) -> eval.
# 5 CV folds spread over PSIR_NGPU GPUs from PSIR_GPU_BASE (default 4 from 0 — 195 has
# GPU 0-3 free; 4 & 7 are another user's). fp16 cache on /data2.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export PSIR_CACHE=${PSIR_CACHE:-/data2/lbliao/psir_cache}
export PSIR_PYTHON=${PSIR_PYTHON:-/home/lbliao/anaconda3/envs/clam/bin/python}
export PSIR_LD_LIBRARY_PATH=${PSIR_LD_LIBRARY_PATH:-/home/lbliao/anaconda3/envs/clam/lib}
export PSIR_ALLOC_CONF=${PSIR_ALLOC_CONF:-expandable_segments:True}   # 195 driver supports it
export PSIR_GPU_BASE=${PSIR_GPU_BASE:-0}
export PSIR_NGPU=${PSIR_NGPU:-4}       # GPU 0-3
export PSIR_LOGD=${PSIR_LOGD:-/home/lbliao/mil_runs/psir_recheck}

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(h-optimus-1)
mkdir -p "$PSIR_LOGD"
echo "models: ${MODELS[*]}  (GPU ${PSIR_GPU_BASE}..$((PSIR_GPU_BASE+PSIR_NGPU-1)))"
bash "$HERE/psir_recheck.sh" "${MODELS[@]}" \
  2>&1 | tee "$PSIR_LOGD/RUN_recheck_195_$(date +%Y%m%d_%H%M).log"
