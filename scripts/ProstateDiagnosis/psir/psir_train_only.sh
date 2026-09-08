#!/usr/bin/env bash
# Train the AB_MIL classifier for all 5 CV folds of PSIR fold K, on GPU (K-1).
# Steps 2-4 (projection/apply/build-folds) already done; fold CSVs now point
# at the /data2 local cache.
set -uo pipefail
K=$1
GPU=$((K-1))
# Repo root = three levels up from scripts/ProstateDiagnosis/psir/. No hardcoded checkout path.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# Set $PSIR_PYTHON to the env python (138: /data12/jing/anaconda3/envs/PrePATH/bin/python,
# 195: /home/lbliao/anaconda3/envs/clam/bin/python). $PSIR_LD_LIBRARY_PATH optional.
PY=${PSIR_PYTHON:-python}
[ -n "${PSIR_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$PSIR_LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$GPU
cd "$REPO"
echo "=== PSIR fold $K classifier, GPU $GPU  $(date '+%F %H:%M:%S') ==="
for cv in 1 2 3 4 5; do
  echo "--- fold$K cv$cv  $(date '+%F %H:%M:%S') ---"
  $PY train_mil.py --yaml_path "configs/ProstateDiagnosis/DataAnalysis/AB_MIL_uni_psir_fold${K}_5fold_3center/fold_${cv}.yaml" \
    || echo "fold$K cv$cv TRAIN FAIL"
done
echo "=== PSIR fold $K DONE  $(date '+%F %H:%M:%S') ==="
