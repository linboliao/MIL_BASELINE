#!/usr/bin/env bash
# PSIR rigorous re-check — SERVER 138 (PrePATH conda).
#
#   ssh 138 ; tmux new -s psir_recheck
#   bash scripts/ProstateDiagnosis/psir/run_recheck_138.sh virchow2
#   PSIR_GPU_BASE=4 PSIR_NGPU=4 bash .../run_recheck_138.sh virchow2   # coexist w/ a 4-GPU job on 0-3
#
# Per encoder ~2.5-4 h: stage -> bare (5 CV) -> psir (5 proj x 5 CV) -> shuf (5x5) -> eval.
# 5 CV folds spread over PSIR_NGPU GPUs from PSIR_GPU_BASE (default 5 from 0). fp16 cache.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export PSIR_CACHE=${PSIR_CACHE:-/data14/lbliao/psir_cache}
export PSIR_PYTHON=${PSIR_PYTHON:-/data12/jing/anaconda3/envs/PrePATH/bin/python}
export PSIR_GPU_BASE=${PSIR_GPU_BASE:-0}
export PSIR_NGPU=${PSIR_NGPU:-5}
export PSIR_LOGD=${PSIR_LOGD:-/home/jing/mil_runs/psir_recheck}
# 138 driver (455/CUDA 11.1): do NOT set PSIR_ALLOC_CONF.

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(virchow2)
mkdir -p "$PSIR_LOGD"
echo "models: ${MODELS[*]}  (GPU ${PSIR_GPU_BASE}..$((PSIR_GPU_BASE+PSIR_NGPU-1)))"
bash "$HERE/psir_recheck.sh" "${MODELS[@]}" \
  2>&1 | tee "$PSIR_LOGD/RUN_recheck_138_$(date +%Y%m%d_%H%M).log"
