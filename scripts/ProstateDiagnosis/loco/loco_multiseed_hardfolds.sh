#!/usr/bin/env bash
# ── PREPARED, NOT AUTO-RUN ──────────────────────────────────────────────────
# Multi-seed replication of ONLY the folds that decide the robustness ranking,
# for the 4 closest top models, with full determinism enabled.
#
#   models : gigapath  h-optimus-1  mstar  gpfm
#   folds  : internal/留省立 (fold 1)   +   type/留RP (fold 2)      <- only these
#   seeds  : 42 43 44 45 46             (same seeds for every model)
#   env    : MIL_DETERMINISM=1  (see utils/repro_utils.py)
#
# Rationale: the 0.052 bACC gap on gigapath/留省立 was traced to model-selection
# on a saturated val metric + cross-env numerical drift, NOT to a split/feature/
# code difference (all byte-identical). A fair comparison therefore needs the
# SAME seed set on EVERY model on ONE machine with determinism on — never a
# multi-seed gigapath vs single-seed others.
#
#   ssh <server> ; tmux new -s loco_multiseed
#   REPO=<MIL_BASELINE>  LOCO_CACHE=<scratch>  LOCO_PYTHON=<py> \
#     bash scripts/ProstateDiagnosis/loco/loco_multiseed_hardfolds.sh
#
# Do NOT run this until the reproducibility patch (repro_utils + the 3-line
# hooks) is merged and pulled on the chosen server.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
P="$HERE"
PY=${LOCO_PYTHON:-python}
GPU_BASE=${LOCO_GPU_BASE:-0}
SEEDS=${LOCO_SEEDS:-"42 43 44 45 46"}
MODELS=${LOCO_MODELS:-"gigapath h-optimus-1 mstar"}
LOGD=${LOCO_LOGD:-$HOME/mil_runs/loco_multiseed}
: "${LOCO_CACHE:?set LOCO_CACHE to a per-server local-disk scratch dir}"
export MIL_DETERMINISM=${MIL_DETERMINISM:-1}
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export LOCO_CACHE PYTHONPATH="$P:${PYTHONPATH:-}"
[ -n "${LOCO_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$LOCO_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
mkdir -p "$LOGD"; cd "$REPO"

declare -A DIM=( [gigapath]=1536 [h-optimus-1]=1536 [mstar]=1024 [gpfm]=1024 [virchow2]=2560 [uni]=1024 [uni2]=1536 [conch]=512 )
# mode -> the single held-out fold index that matters
declare -A HARD_FOLD=( [internal]=1 [type]=2 )   # internal fold1=留省立 ; type fold2=留RP

echo "repo $REPO @ $(git rev-parse --short HEAD)   MIL_DETERMINISM=$MIL_DETERMINISM"
echo "models: $MODELS | seeds: $SEEDS | folds: internal#1(留省立) type#2(留RP)"

for M in $MODELS; do
  echo "########## $M (in_dim ${DIM[$M]})  $(date '+%F %H:%M:%S') ##########"
  rm -rf "$LOCO_CACHE"; mkdir -p "$LOCO_CACHE"
  $PY "$P/loco_cache.py" --model "$M" || { echo "$M CACHE FAIL"; continue; }

  for MODE in internal type; do
    F=${HARD_FOLD[$MODE]}
    $PY "$P/loco_build_folds.py" --model "$M" --mode "$MODE" || { echo "$M/$MODE BUILD FAIL"; continue; }
    $PY "$P/loco_gen_configs.py" --model "$M" --mode "$MODE" --in_dim "${DIM[$M]}" || continue
    Y="configs/ProstateDiagnosis/DataAnalysis/AB_MIL_${M}_loco_${MODE}.yaml"
    for SEED in $SEEDS; do
      TS="ms${SEED}_$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
      echo ">>> $M/$MODE fold $F seed $SEED -> seed_${SEED}_${TS}"
      CUDA_VISIBLE_DEVICES=$GPU_BASE $PY train_mil.py --yaml_path "$Y" \
        --only_fold "$F" --run_ts "$TS" --no_merge \
        --options General.seed="$SEED" \
        > "$LOGD/${M}_${MODE}_f${F}_s${SEED}.log" 2>&1 \
        || echo "   !! $M/$MODE/$SEED FAILED (see log)"
    done
  done
  rm -rf "$LOCO_CACHE"
done

echo "########## DONE $(date '+%F %H:%M:%S') ##########"
echo "aggregate:  $PY $P/loco_multiseed_collect.py --root result/ProstateDiagnosis/DataAnalysis --out $LOGD/multiseed_summary"
