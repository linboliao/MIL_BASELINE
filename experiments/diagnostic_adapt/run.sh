#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PROJECT=/NAS145/linboliao/Projects/ProstateDiagnosis
EXPERIMENTS=/NAS145/linboliao/Experiments/ProstateDiagnosis
INPUT="$EXPERIMENTS/diagnostic_adapt"
OUTPUT="$EXPERIMENTS/provenance/generalization/diagnostic-adaptation-mve"
PREPATH=/data2/lbliao/Projects/PrePATH
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR="$PROJECT/.agent/nodes/195/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"

stage2=("$PY" -B -m experiments.diagnostic_adapt.run_stage2
        --config "$INPUT/configs/frozen_config.json" --datasets "$INPUT/datasets" --prepath "$PREPATH")

[[ $# -eq 0 ]] && exec "${stage2[@]}" plan
[[ $# -eq 2 && $1 =~ ^[0-9]{8}-[a-zA-Z0-9_-]+$ && $2 =~ ^[0-9]+$ ]]
export CUDA_VISIBLE_DEVICES="$2"
flock -n "$PROJECT/.agent/nodes/195/status/diagnostic_adaptation_mve_gpu.lock" \
  "${stage2[@]}" run --output "$OUTPUT/$1"
