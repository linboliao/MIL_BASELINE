#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 195
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP=$TMPDIR TEMP=$TMPDIR
FULL_SOURCE=/NAS145/linboliao/Experiments/ProstateDiagnosis/FULL_SOURCE

# H1：固定源域划分，按 Selection macro-F1 选模。
"$PY" -B train_mil.py --yaml_path "$FULL_SOURCE/configs/h-optimus-1.yaml"

# 需要重跑对照时启用相应行。
# "$PY" -B train_mil.py --yaml_path "$FULL_SOURCE/configs/uni2.yaml"
# "$PY" -B train_mil.py --yaml_path "$FULL_SOURCE/configs/mstar.yaml"
