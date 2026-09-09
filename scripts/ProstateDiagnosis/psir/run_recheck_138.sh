#!/usr/bin/env bash
# PSIR rigorous re-check — SERVER 138 (the box with CPU + GPU headroom; 195 is at load ~80).
#
#   ssh 138 ; tmux new -s psir_recheck
#   bash scripts/ProstateDiagnosis/psir/run_recheck_138.sh              # default: virchow2 h-optimus-1
#   bash scripts/ProstateDiagnosis/psir/run_recheck_138.sh gpfm        # or name the encoder(s)
#
# Per encoder ~2.5 h: stage -> bare (5 CV) -> psir (5 proj x 5 CV) -> shuf control (5 x 5) -> eval.
# 5 CV folds train in parallel on GPU PSIR_GPU_BASE..+4 (default 0-4).
# Feature cache is fp16 + KEPT at $PSIR_CACHE.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

export PSIR_CACHE=${PSIR_CACHE:-/data14/lbliao/psir_cache}
export PSIR_PYTHON=${PSIR_PYTHON:-/data12/jing/anaconda3/envs/PrePATH/bin/python}
export PSIR_GPU_BASE=${PSIR_GPU_BASE:-0}
export PSIR_LOGD=${PSIR_LOGD:-/home/jing/mil_runs/psir_recheck}
# 138 driver (455/CUDA 11.1): do NOT set PSIR_ALLOC_CONF (no expandable_segments).
# export PSIR_VARIANTS="bare psir shuf"   # default

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(virchow2 h-optimus-1)
mkdir -p "$PSIR_LOGD"
echo "models: ${MODELS[*]}"
bash "$HERE/psir_recheck.sh" "${MODELS[@]}" \
  2>&1 | tee "$PSIR_LOGD/RUN_recheck_$(date +%Y%m%d_%H%M).log"
