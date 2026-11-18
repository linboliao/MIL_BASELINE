#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
ROOT=/NAS145/linboliao/Experiments/ProstateDiagnosis
CONFIG="$ROOT/case_reliability/configs/reliability.json"
[[ $# -eq 0 ]] && exec "$PY" -B -m experiments.case_reliability.run --config "$CONFIG"
[[ $# -eq 1 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
"$PY" -B -m experiments.case_reliability.run --config "$CONFIG" \
  --output "$ROOT/provenance/generalization/case-reliability/$1"
