#!/usr/bin/env bash
# Leave-one-X-out AB_MIL robustness benchmark for one or more PFM encoders.
#
#   usage:  LOCO_CACHE=/data14/lbliao/loco_cache \
#           LOCO_PYTHON=/data12/jing/anaconda3/envs/PrePATH/bin/python \
#           scripts/ProstateDiagnosis/loco/loco_run.sh  gigapath gpfm mstar
#
# Per encoder (sequential): wipe local cache -> stage this encoder's ~2520
# cohort features to $LOCO_CACHE as fp16 -> run the requested modes:
#   internal  : leave-one-center-out (省立, 新昌)  [+ pristine external eval on 301/ynzl]
#   type      : leave-one-type-out (CNB, RP, TURP)
#   fivesite  : leave-one-site-out (省立, 新昌, 301, 云南肿瘤)
# folds within a mode train in parallel on GPU $LOCO_GPU_BASE .. +n-1.
# cache is wiped again before the next encoder and at the end.
#
# env:
#   LOCO_CACHE          (required) per-server local scratch dir, wiped between encoders
#   LOCO_PYTHON         python with the MIL env         (default: python)
#   LOCO_GPU_BASE       first GPU id to use             (default: 0)   folds use BASE..BASE+nf-1
#   LOCO_MODES          space-separated subset of modes (default: "internal type fivesite")
#   LOCO_LOGD           per-fold log dir                (default: $HOME/mil_runs/loco)
#   LOCO_LD_LIBRARY_PATH  prepended to LD_LIBRARY_PATH if set (195 conda needs its lib/)
#   PROSTATE_FEAT_ROOT  shared NAS feature root         (default: NAS145 迈新生物_特征 path)
#   LOCO_STAGE_WORKERS  cache staging threads           (default: 24)
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
P=scripts/ProstateDiagnosis/loco
PY=${LOCO_PYTHON:-python}
GPU_BASE=${LOCO_GPU_BASE:-0}
MODES=${LOCO_MODES:-"internal type fivesite"}
LOGD=${LOCO_LOGD:-$HOME/mil_runs/loco}
STAGE_WORKERS=${LOCO_STAGE_WORKERS:-}   # empty -> loco_cache.py picks min(16, ncpu) processes
KEEP_CACHE=${LOCO_KEEP_CACHE:-}   # set to 1 to NOT wipe the fp16 feature cache (so PSIR etc can reuse it)
: "${LOCO_CACHE:?set LOCO_CACHE to a per-server local-disk scratch dir}"

[ -n "${LOCO_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$LOCO_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export LOCO_CACHE
export PYTHONPATH="$REPO/$P:${PYTHONPATH:-}"
# expandable_segments needs driver >= 470 (CUDA 11.4+). 138's driver is 455 / CUDA 11.1
# and crashes on it ("nvmlDeviceGetNvLinkRemoteDeviceType ... INTERNAL ASSERT FAILED").
# Opt in via $LOCO_ALLOC_CONF (195 sets it); leave unset elsewhere. The model is tiny
# (AB_MIL L=512) so the allocator hint is cosmetic here anyway.
[ -n "${LOCO_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$LOCO_ALLOC_CONF"
mkdir -p "$LOGD"
cd "$REPO"

declare -A DIM=(
  [conch]=512 [uni]=1024 [uni2]=1536 [virchow2]=2560
  [h-optimus-1]=1536 [gigapath]=1536 [gpfm]=1024 [mstar]=1024
)
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && { echo "usage: LOCO_CACHE=... $0 <model> [model...]  (known: ${!DIM[*]})"; exit 2; }
for M in "${MODELS[@]}"; do
  [ -z "${DIM[$M]:-}" ] && { echo "unknown model: $M   (known: ${!DIM[*]})"; exit 2; }
done

# One yaml per (model,mode). The N folds run in PARALLEL (GPU GPU_BASE..+N-1),
# each `train_mil.py --only_fold k --run_ts $TS --no_merge` so they land in ONE
# shared result/.../AB_MIL_<model>_loco_<mode>/AB_MIL/seed_42_<TS>/fold_<k>/ ;
# a final --merge_only pass writes merge_<N>_fold_metrics.json.
declare -A NF=( [internal]=2 [type]=3 [fivesite]=4 )
run_mode () {   # $1=model  $2=mode
  local M=$1 MODE=$2 nf=${NF[$2]} f
  $PY $P/loco_build_folds.py --model "$M" --mode "$MODE" || { echo "$M/$MODE BUILD FAIL"; return 1; }
  $PY $P/loco_gen_configs.py --model "$M" --mode "$MODE" --in_dim "${DIM[$M]}" || return 1
  local TS Y pids=()
  TS=$(TZ=Asia/Shanghai date +%Y-%m-%d-%H-%M)
  Y="configs/ProstateDiagnosis/DataAnalysis/AB_MIL_${M}_loco_${MODE}.yaml"
  echo ">>> [$M/$MODE] $nf folds ‖ on GPU $GPU_BASE..$((GPU_BASE+nf-1))  seed dir $TS  $(date '+%F %H:%M:%S')"
  for f in $(seq 1 "$nf"); do
    CUDA_VISIBLE_DEVICES=$((GPU_BASE + f - 1)) nohup $PY train_mil.py --yaml_path "$Y" \
      --only_fold "$f" --run_ts "$TS" --no_merge > "$LOGD/${M}_${MODE}_fold${f}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  $PY train_mil.py --yaml_path "$Y" --run_ts "$TS" --merge_only >/dev/null 2>&1 || true
  $PY $P/loco_collect.py --model "$M" --mode "$MODE" | tee "$LOGD/${M}_${MODE}_summary.txt"
}

for M in "${MODELS[@]}"; do
  echo "############################################################"
  echo "### $M  (in_dim ${DIM[$M]})  $(date '+%F %H:%M:%S')"
  echo "############################################################"
  [ -z "$KEEP_CACHE" ] && rm -rf "$LOCO_CACHE"
  mkdir -p "$LOCO_CACHE"
  echo ">>> staging $M (fp16)  $(date '+%H:%M:%S')"
  $PY $P/loco_cache.py --model "$M" ${STAGE_WORKERS:+--workers "$STAGE_WORKERS"} || { echo "$M CACHE FAIL"; exit 1; }

  for MODE in $MODES; do
    run_mode "$M" "$MODE" || { echo "$M/$MODE FAIL"; continue; }
    if [ "$MODE" = internal ]; then
      echo ">>> [$M/internal] pristine external eval on 301 + ynzl"
      $PY $P/loco_eval_external.py --model "$M" --in_dim "${DIM[$M]}" \
        | tee "$LOGD/${M}_internal_external.txt"
    fi
  done
  if [ -z "$KEEP_CACHE" ]; then rm -rf "$LOCO_CACHE"; echo ">>> $M DONE, cache wiped  $(date '+%F %H:%M:%S')";
  else echo ">>> $M DONE, cache KEPT at $LOCO_CACHE  $(date '+%F %H:%M:%S')"; fi
  echo
done

echo "############################################################"
echo "### LOCO DONE for: ${MODELS[*]}   $(date '+%F %H:%M:%S')"
echo "### summaries -> $LOGD/{model}_{internal,type,fivesite}_summary.txt"
echo "############################################################"
