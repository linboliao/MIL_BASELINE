#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -u scripts/Diagnosis/summarize_oof.py --input-root result/Diagnosis/Mag/OOF --output-dir result/Diagnosis/Mag/Statistics --bootstrap-iterations 2000 --seed 2024 --threshold 0.5
