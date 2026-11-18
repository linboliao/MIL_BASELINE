#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 195 环境；在 138 上改为对应的 Python 和 TMPDIR。
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP=$TMPDIR TEMP=$TMPDIR

# conch：DATA 指向已准备好的划分；OUT 和 CACHE 每次使用新目录。
SOURCE=/NAS145/linboliao/Experiments/ProstateDiagnosis/PFM_LOCO/conch
DATA=/path/to/prepared/conch
OUT=/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/pfm-loco-prostate-reproduction/seed42-conch-r1
CACHE=/data2/lbliao/pfm_loco_cache/seed42-conch-r1

# staging
"$PY" -B -m experiments.stage --cache "$CACHE" \
  --dataset "$DATA/SL/train_val.csv" --dataset "$DATA/XC/train_val.csv" \
  --dataset "$DATA/CNB/train_val.csv" --dataset "$DATA/TURP/train_val.csv" --dataset "$DATA/RP/train_val.csv"
mkdir -p "$OUT"
cp -n "$CACHE/manifest.json" "$OUT/staging_manifest.json"

# train：每行一个留出任务，按需增加模型或划分。
"$PY" -B -m experiments.run_fold --config "$SOURCE/SL/configs/model.yaml" --dataset "$CACHE/datasets/0.csv" --test-dataset "$DATA/SL/test.csv" --output "$OUT/SL" --gpu 0 --tie-break val_loss
"$PY" -B -m experiments.run_fold --config "$SOURCE/XC/configs/model.yaml" --dataset "$CACHE/datasets/1.csv" --test-dataset "$DATA/XC/test.csv" --output "$OUT/XC" --gpu 0 --tie-break val_loss
"$PY" -B -m experiments.run_fold --config "$SOURCE/CNB/configs/model.yaml" --dataset "$CACHE/datasets/2.csv" --test-dataset "$DATA/CNB/test.csv" --output "$OUT/CNB" --gpu 0 --tie-break val_loss
"$PY" -B -m experiments.run_fold --config "$SOURCE/TURP/configs/model.yaml" --dataset "$CACHE/datasets/3.csv" --test-dataset "$DATA/TURP/test.csv" --output "$OUT/TURP" --gpu 0 --tie-break val_loss
"$PY" -B -m experiments.run_fold --config "$SOURCE/RP/configs/model.yaml" --dataset "$CACHE/datasets/4.csv" --test-dataset "$DATA/RP/test.csv" --output "$OUT/RP" --gpu 0 --tie-break val_loss

# clean staging：前面的训练失败时不会执行到这里。
"$PY" -B -m experiments.clean_stage --manifest "$CACHE/manifest.json" --execute --receipt "$OUT/cleanup.jsonl"
