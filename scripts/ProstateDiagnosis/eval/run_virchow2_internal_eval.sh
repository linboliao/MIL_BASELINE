#!/usr/bin/env bash
# Internal held-out test eval (the fixed 332-slide set every fold was validated
# against; = internal_test.csv, properly held out of all fold train/val).
# Produces per-slide predictions so we can stratify by specimen type / center
# and compare against the 301 external failure pattern.
#
# Prereq: internal_test_virchow2_fp16.csv built from the fold test column.
# Usage: run_virchow2_internal_eval.sh [RUN_DIR]
set -uo pipefail
REPO=/NAS2/Data1/lbliao/Code-195/MIL_BASELINE
PY=/home/lbliao/anaconda3/envs/maixin/bin/python
MAXPAR=${MAXPAR:-5}
CSV=$REPO/datasets/ProstateDiagnosis/DataAnalysis/internal_test/internal_test_virchow2_fp16.csv
RES_ROOT=$REPO/result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local
RUN_DIR=${1:-$(ls -dt "$RES_ROOT"/run_* | head -1)}
cd "$REPO"
OUT=$RUN_DIR/internal_test
mkdir -p "$OUT"
echo "run dir: $RUN_DIR"
[ -f "$CSV" ] || { echo "missing $CSV"; exit 1; }

running=0
for k in 1 2 3 4 5; do
  gpu=$((k - 1))
  ckpt=$(ls -t "$RUN_DIR"/fold_${k}/Best_EPOCH_*.pth 2>/dev/null | head -1)
  [ -z "$ckpt" ] && { echo "fold $k: no ckpt, skip"; continue; }
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u test_mil.py \
    --yaml_path "$RUN_DIR/fold_${k}/fold_${k}.yaml" \
    --test_dataset_csv "$CSV" \
    --model_weight_path "$ckpt" \
    --test_log_dir "$OUT/fold_${k}" \
    --device cuda:0 > "$OUT/fold${k}.log" 2>&1 &
  echo "launched fold $k -> GPU $gpu (ckpt $(basename "$ckpt"))"
  running=$((running + 1))
  [ $running -ge $MAXPAR ] && { wait -n; running=$((running - 1)); }
done
wait
echo "done -> $OUT"
