#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR" MPLCONFIGDIR="$TMPDIR"
ROOT=/NAS145/linboliao/Experiments/ProstateDiagnosis
PANEL="$ROOT/panel_ab"

[[ $# -eq 0 ]] && exec "$PY" -B -m experiments.panel_ab.check --config "$PANEL/configs/panel.json"
[[ $# -eq 1 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
"$PY" -B -m experiments.panel_ab.analyze --config "$PANEL/configs/panel.json" \
  --predictions "$PANEL/datasets/reference_predictions.csv" \
  --out "$ROOT/provenance/reproducibility/panel-ab-analysis/$1"
