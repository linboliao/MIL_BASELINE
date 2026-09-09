#!/usr/bin/env bash
# PSIR rigorous re-check for one or more PFM encoders.
#
#   PSIR_CACHE=/data14/lbliao/psir_cache \
#   PSIR_PYTHON=/data12/jing/anaconda3/envs/PrePATH/bin/python \
#   scripts/ProstateDiagnosis/psir/psir_recheck.sh  virchow2
#
# Per encoder, on the identical 5-fold-3-center split (seed 42) + identical
# external eval, runs 3 matched conditions (PSIR_VARIANTS, default "bare psir shuf"):
#   bare : raw features (native dim)
#   psir : Panel-A SupCon projection, 5 folds  (the method under test)
#   shuf : same recipe, case labels shuffled   (negative control)
# then eval on 301 + ynzl (AUC, spec@0.5, spec@sens95-internal-calibrated).
# Feature cache is fp16, local, and KEPT (reuse a LOCO_KEEP_CACHE=1 loco cache
# by pointing PSIR_CACHE at it — same layout).
#
# env:
#   PSIR_CACHE          (required) local fp16 feature dir (kept)
#   PSIR_PYTHON         env python                 (default: python)
#   PSIR_GPU_BASE       first GPU id               (default: 0)
#   PSIR_NGPU          spread 5 CV folds over N GPUs, BASE..BASE+N-1 (default 5;
#                      set 4 to run alongside a 4-GPU LOCO job — AB_MIL is tiny)
#   PSIR_VARIANTS       subset of "bare psir shuf" (default: all)
#   PSIR_LD_LIBRARY_PATH   prepended if set (195 clam)
#   PSIR_ALLOC_CONF     PYTORCH_CUDA_ALLOC_CONF    (195: expandable_segments:True; 138: unset)
#   PSIR_LOGD           log dir                    (default: $HOME/mil_runs/psir_recheck)
#   PSIR_STAGE_WORKERS  staging threads            (default: 24)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${PSIR_PYTHON:-python}
RC=scripts/ProstateDiagnosis/psir/psir_recheck.py
GPU_BASE=${PSIR_GPU_BASE:-0}
NGPU=${PSIR_NGPU:-5}          # spread the 5 CV folds over this many GPUs (BASE..BASE+NGPU-1)
VARIANTS=${PSIR_VARIANTS:-"bare psir shuf"}
LOGD=${PSIR_LOGD:-$HOME/mil_runs/psir_recheck}
SW=${PSIR_STAGE_WORKERS:-}   # empty -> psir_recheck.py picks min(16, ncpu) processes
: "${PSIR_CACHE:?set PSIR_CACHE to a local fp16 feature dir}"
[ -n "${PSIR_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$PSIR_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
[ -n "${PSIR_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$PSIR_ALLOC_CONF"
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export PSIR_CACHE
mkdir -p "$LOGD"; cd "$REPO"

train_cv () {   # $1 = result-dir name (AB_MIL_..._recheck_...)
  local NAME=$1 pids=() cv
  $PY "$RC" configs --model "$MODEL" --variant "$VARIANT" ${FOLD:+--fold $FOLD} || return 1
  echo ">>> [$NAME] train 5 CV over GPU $GPU_BASE..$((GPU_BASE+NGPU-1))  $(date '+%H:%M:%S')"
  for cv in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$((GPU_BASE + (cv - 1) % NGPU)) nohup $PY train_mil.py \
      --yaml_path "configs/ProstateDiagnosis/DataAnalysis/$NAME/fold_${cv}.yaml" \
      > "$LOGD/${NAME}_cv${cv}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
}

for MODEL in "$@"; do
  echo "############################################################"
  echo "### PSIR re-check: $MODEL   $(date '+%F %H:%M:%S')"
  echo "############################################################"
  $PY "$RC" stage --model "$MODEL" ${SW:+--workers "$SW"} || { echo "$MODEL STAGE FAIL"; exit 1; }

  for VARIANT in $VARIANTS; do
    if [ "$VARIANT" = bare ]; then
      FOLD=""
      $PY "$RC" folds --model "$MODEL" --variant bare || { echo "$MODEL bare folds FAIL"; continue; }
      train_cv "AB_MIL_${MODEL}_recheck_bare"
    else
      for FOLD in 1 2 3 4 5; do
        echo ">>> [$MODEL/$VARIANT] projection fold $FOLD  $(date '+%H:%M:%S')"
        CUDA_VISIBLE_DEVICES=$GPU_BASE $PY "$RC" proj --model "$MODEL" --fold "$FOLD" \
          $([ "$VARIANT" = shuf ] && echo --shuffle) > "$LOGD/${MODEL}_${VARIANT}_proj${FOLD}.log" 2>&1 \
          || { echo "$MODEL/$VARIANT proj$FOLD FAIL"; continue; }
        CUDA_VISIBLE_DEVICES=$GPU_BASE $PY "$RC" apply --model "$MODEL" --fold "$FOLD" --variant "$VARIANT" \
          >> "$LOGD/${MODEL}_${VARIANT}_proj${FOLD}.log" 2>&1 || { echo "$MODEL/$VARIANT apply$FOLD FAIL"; continue; }
        $PY "$RC" folds --model "$MODEL" --variant "$VARIANT" --fold "$FOLD" || continue
        train_cv "AB_MIL_${MODEL}_recheck_${VARIANT}_fold${FOLD}"
      done
      FOLD=""
    fi
    $PY "$RC" eval --model "$MODEL" --variant "$VARIANT" | tee "$LOGD/${MODEL}_${VARIANT}_eval.txt"
  done
  echo ">>> $MODEL re-check DONE (cache kept at $PSIR_CACHE)  $(date '+%F %H:%M:%S')"
done

echo "############################################################"
echo "### PSIR re-check DONE for: $*   $(date '+%F %H:%M:%S')"
echo "### collate:  python $RC ... (see psir_recheck_collect.py)"
echo "############################################################"
