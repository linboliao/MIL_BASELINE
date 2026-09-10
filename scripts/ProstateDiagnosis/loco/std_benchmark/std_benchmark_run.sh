#!/usr/bin/env bash
# ═══ PHASE-1 STANDARDIZED 8-PFM LOCO BENCHMARK — PREPARED, NOT AUTO-RUN ═══
#
# One machine, one commit, one env, fp32 raw features, MIL_DETERMINISM=1,
# frozen patient splits. 8 PFM × 5 held-out folds = 40 trainings.
#
#   folds:  internal#1 留省立 · internal#2 留新昌
#           type#1 留CNB · type#2 留RP · type#3 留TURP
#   protocol: Plain AB_MIL, seed 42, StratifiedGroupKFold(7) 1/7 patient val
#   per PFM: stage fp32 -> 5 folds in parallel on GPU $GPU_BASE..+4 -> wipe
#
#   ssh 138 ; tmux new -s std_bench
#   REPO=/NAS3/lbliao/Code-138/MIL_BASELINE \
#   LOCO_PYTHON=/data12/jing/anaconda3/envs/PrePATH/bin/python \
#   LOCO_CACHE=/data14/lbliao/stdbench_cache  LOCO_GPU_BASE=0 \
#     bash scripts/ProstateDiagnosis/loco/std_benchmark/std_benchmark_run.sh
#
# Requires: determinism + checkpoint-selection patch merged & pulled
#           (utils/repro_utils.py present; model_select has val_loss tie-break).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
LOCO=$(cd "$HERE/.." && pwd)                        # scripts/ProstateDiagnosis/loco
PY=${LOCO_PYTHON:-python}
GPU_BASE=${LOCO_GPU_BASE:-0}
MODELS=${LOCO_MODELS:-"conch uni uni2 virchow2 h-optimus-1 mstar gigapath gpfm"}
TAG=${STD_TAG:-std-$(date +%Y%m%d)}
LOGD=${LOCO_LOGD:-$HOME/mil_runs/std_bench}
NOCLOBBER=${STD_NOCLOBBER:-1}
: "${LOCO_CACHE:?set LOCO_CACHE to a local-disk scratch dir (>=350G free for fp32)}"

export LOCO_RAW=1                                   # fp32 verbatim — no fp16 rounding
export MIL_DETERMINISM=${MIL_DETERMINISM:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export LOCO_CACHE PYTHONPATH="$LOCO:${PYTHONPATH:-}"
[ -n "${LOCO_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$LOCO_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
mkdir -p "$LOGD"; cd "$REPO"

declare -A DIM=( [conch]=512 [uni]=1024 [uni2]=1536 [virchow2]=2560
                 [h-optimus-1]=1536 [gigapath]=1536 [gpfm]=1024 [mstar]=1024 )
# mode -> "fold:held_out" list (the phase-1 core five)
declare -A MODEFOLDS=( [internal]="1:省立 2:新昌" [type]="1:CNB 2:RP 3:TURP" )

echo "repo $REPO @ $(git rev-parse --short HEAD)"
echo "TAG=$TAG  MIL_DETERMINISM=$MIL_DETERMINISM  fp32(raw)  GPU $GPU_BASE..$((GPU_BASE+4))"
echo "models: $MODELS"
$PY -c "import utils.repro_utils" 2>/dev/null || { echo "!! utils/repro_utils.py missing — apply the patch first"; exit 3; }

for M in $MODELS; do
  echo "########## $M (in_dim ${DIM[$M]})  $(date '+%F %H:%M:%S') ##########"
  rm -rf "$LOCO_CACHE"; mkdir -p "$LOCO_CACHE"
  $PY "$LOCO/loco_cache.py" --model "$M" || { echo "$M CACHE FAIL"; continue; }

  for MODE in internal type; do
    $PY "$LOCO/loco_build_folds.py" --model "$M" --mode "$MODE" || { echo "$M/$MODE BUILD FAIL"; continue; }
    $PY "$LOCO/loco_gen_configs.py" --model "$M" --mode "$MODE" --in_dim "${DIM[$M]}" || continue
    Y="configs/ProstateDiagnosis/DataAnalysis/AB_MIL_${M}_loco_${MODE}.yaml"
    SEEDDIR="result/ProstateDiagnosis/DataAnalysis/AB_MIL_${M}_loco_${MODE}/AB_MIL/seed_42_${TAG}"
    pids=(); i=0
    for spec in ${MODEFOLDS[$MODE]}; do
      f=${spec%%:*}; ho=${spec##*:}
      if [ "$NOCLOBBER" = 1 ] && compgen -G "$SEEDDIR/fold_${f}/Best_Log_*.csv" >/dev/null; then
        echo "   skip $M/$MODE fold $f ($ho) -- already done in $SEEDDIR/fold_$f"; continue
      fi
      G=$((GPU_BASE + i)); i=$((i+1))
      echo ">>> $M/$MODE fold $f=$ho  GPU $G  -> $SEEDDIR/fold_$f"
      CUDA_VISIBLE_DEVICES=$G nohup $PY train_mil.py --yaml_path "$Y" \
        --only_fold "$f" --run_ts "$TAG" --no_merge \
        > "$LOGD/${M}_${MODE}_f${f}.log" 2>&1 &
      pids+=($!)
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
    $PY train_mil.py --yaml_path "$Y" --run_ts "$TAG" --merge_only >/dev/null 2>&1 || true

    # standardized per-fold prediction dump (slide + patient level)
    for spec in ${MODEFOLDS[$MODE]}; do
      f=${spec%%:*}; ho=${spec##*:}
      FD="$SEEDDIR/fold_$f"
      [ -d "$FD" ] || continue
      SP="test val"
      $PY "$HERE/std_predict.py" --repo "$REPO" --fold_dir "$FD" \
        --in_dim "${DIM[$M]}" --model "$M" --mode "$MODE" --held_out "$ho" \
        --splits $SP >> "$LOGD/${M}_${MODE}_preds.log" 2>&1 || echo "   pred FAIL $M/$MODE/$f"
    done
  done
  rm -rf "$LOCO_CACHE"
  echo ">>> $M done  $(date '+%F %H:%M:%S')"
done

echo "########## PHASE-1 BENCHMARK DONE  $(date '+%F %H:%M:%S') ##########"
echo "collect:  $PY $HERE/std_collect.py --repo $REPO --tag $TAG \\"
echo "            --models $MODELS --out $LOGD/summary"
