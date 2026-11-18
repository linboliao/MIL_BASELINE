#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
ROOT=/NAS145/linboliao/Experiments/ProstateDiagnosis
CONFIG="$ROOT/threshold_transport/configs/transport.json"

[[ $# -eq 0 ]] && exec "$PY" -B -m experiments.threshold_transport.analyze --config "$CONFIG"
[[ $# -eq 1 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
"$PY" -B -m experiments.threshold_transport.analyze --config "$CONFIG" \
  --output "$ROOT/provenance/generalization/threshold-transport/$1"
