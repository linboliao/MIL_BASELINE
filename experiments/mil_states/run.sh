#!/usr/bin/env bash
# One native R2 cell on 195. Supply a prepared, exclusive R2 run ID.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP=$TMPDIR TEMP=$TMPDIR
PREP=/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/r2-mil-retrain/20261105-guard-prep-r1
RUN_ID=${1:?Pass the prepared R2 run ID}

"$PY" -B -m experiments.mil_states.stage_features --prep "$PREP" --host 195 --pfm mstar --run-id "$RUN_ID"
"$PY" -B -m experiments.mil_states.train_cell --prep "$PREP" --host 195 --pfm mstar --fold SL --seed 42 --run-id "$RUN_ID" --execute
"$PY" -B -m experiments.mil_states.audit_cell --prep "$PREP" --host 195 --pfm mstar --fold SL --seed 42 --run-id "$RUN_ID"
# After reviewing the source-only selection and evaluation protocol:
# "$PY" -B -m experiments.mil_states.evaluate_cell --prep "$PREP" --host 195 --pfm mstar --fold SL --seed 42 --run-id "$RUN_ID" --execute
