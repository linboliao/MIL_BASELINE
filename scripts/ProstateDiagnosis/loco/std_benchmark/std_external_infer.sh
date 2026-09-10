#!/usr/bin/env bash
# ═══ PHASE-2 EXTERNAL INFERENCE (301 / ynzl) — PREPARED, NOT AUTO-RUN ═══
# After phase-1 checkpoints exist, score every internal-mode fold model on the
# pristine 301 + ynzl cohorts with the SAME std_predict schema as the main
# benchmark. No retraining.
#
#   REPO=... LOCO_PYTHON=... LOCO_CACHE=/data14/lbliao/stdbench_cache \
#   STD_TAG=std-20260911 LOCO_GPU_BASE=0 \
#     bash scripts/ProstateDiagnosis/loco/std_benchmark/std_external_infer.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
LOCO=$(cd "$HERE/.." && pwd)
PY=${LOCO_PYTHON:-python}
TAG=${STD_TAG:-std-$(date +%Y%m%d)}
MODELS=${LOCO_MODELS:-"conch uni uni2 virchow2 h-optimus-1 mstar gigapath gpfm"}
G=${LOCO_GPU_BASE:-0}
: "${LOCO_CACHE:?set LOCO_CACHE}"
export LOCO_RAW=1
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export LOCO_CACHE PYTHONPATH="$LOCO:${PYTHONPATH:-}"
[ -n "${LOCO_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$LOCO_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
cd "$REPO"
declare -A DIM=( [conch]=512 [uni]=1024 [uni2]=1536 [virchow2]=2560
                 [h-optimus-1]=1536 [gigapath]=1536 [gpfm]=1024 [mstar]=1024 )

for M in $MODELS; do
  SEEDDIR="result/ProstateDiagnosis/DataAnalysis/AB_MIL_${M}_loco_internal/AB_MIL/seed_42_${TAG}"
  [ -d "$SEEDDIR" ] || { echo "skip $M — no $SEEDDIR"; continue; }
  # external features must be staged for THIS model
  rm -rf "$LOCO_CACHE"; mkdir -p "$LOCO_CACHE"
  $PY "$LOCO/loco_cache.py" --model "$M" || { echo "$M CACHE FAIL"; continue; }
  for k in 1 2; do
    FD="$SEEDDIR/fold_$k"; [ -d "$FD" ] || continue
    ho=$([ $k = 1 ] && echo 省立 || echo 新昌)
    CUDA_VISIBLE_DEVICES=$G $PY "$HERE/std_predict.py" --repo "$REPO" \
      --fold_dir "$FD" --in_dim "${DIM[$M]}" --model "$M" --mode internal \
      --held_out "$ho" --splits --external 301 云南肿瘤
  done
  rm -rf "$LOCO_CACHE"
done
echo "external inference done — re-run std_collect.py to fold 301/ynzl into the summary"
