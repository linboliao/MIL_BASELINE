#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -u scripts/Diagnosis/spe/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml \
  --devices cuda:1,cuda:3,cuda:4,cuda:7 \
  "$@"
