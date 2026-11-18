#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 195. With no arguments this checks the two arm inputs only.
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP=$TMPDIR TEMP=$TMPDIR
ROOT=/NAS145/linboliao/Experiments/ProstateDiagnosis
CONFIG="$ROOT/rp_support/configs/training/mstar"

if [[ $# -eq 0 ]]; then
  "$PY" -B -m experiments.rp_support.train_cell --config "$CONFIG/fold_1_A.json" --plan
  "$PY" -B -m experiments.rp_support.train_cell --config "$CONFIG/fold_1_D.json" --plan
  exit 0
fi
[[ $# -eq 2 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $2 =~ ^[0-9]+$ ]]
OUT="$ROOT/provenance/generalization/rp-support-retrain/$1"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"

# mSTAR, fold 1. A/D is the primary comparison.
"$PY" -B -m experiments.rp_support.train_cell --config "$CONFIG/fold_1_A.json" --output "$OUT/A" --gpu "$2"
"$PY" -B -m experiments.rp_support.train_cell --config "$CONFIG/fold_1_D.json" --output "$OUT/D" --gpu "$2"
# "$PY" -B -m experiments.rp_support.train_cell --config "$CONFIG/fold_1_B.json" --output "$OUT/B" --gpu "$2"
