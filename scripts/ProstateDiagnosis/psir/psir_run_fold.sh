#!/usr/bin/env bash
# Run the full PSIR (UNI) pipeline for one PSIR fold K, pinned to GPU (K-1):
#   step2 train projection -> step3 apply -> step4 build folds -> gen configs
#   -> train classifier for all 5 CV folds.
set -uo pipefail
K=$1
GPU=$((K-1))
# Repo root = three levels up from scripts/ProstateDiagnosis/psir/. No hardcoded checkout path.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# Set $PSIR_PYTHON to the env python (138: /data12/jing/anaconda3/envs/PrePATH/bin/python,
# 195: /home/lbliao/anaconda3/envs/clam/bin/python). $PSIR_LD_LIBRARY_PATH optional.
PY=${PSIR_PYTHON:-python}
P=scripts/ProstateDiagnosis/psir
[ -n "${PSIR_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$PSIR_LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

echo "=== PSIR fold $K on GPU $GPU  $(date '+%F %H:%M:%S') ==="
$PY $P/psir_train_projection.py --fold $K   || { echo "fold$K STEP2 FAIL"; exit 1; }
$PY $P/psir_apply_projection.py --fold $K   || { echo "fold$K STEP3 FAIL"; exit 1; }
$PY $P/psir_build_folds_psir.py --fold $K   || { echo "fold$K STEP4 FAIL"; exit 1; }
$PY $P/psir_gen_configs.py --fold $K --gpu 0 || { echo "fold$K GENCFG FAIL"; exit 1; }

for cv in 1 2 3 4 5; do
  echo "--- fold$K classifier cv$cv  $(date '+%F %H:%M:%S') ---"
  $PY train_mil.py --yaml_path "configs/ProstateDiagnosis/DataAnalysis/AB_MIL_uni_psir_fold${K}_5fold_3center/fold_${cv}.yaml" \
    || echo "fold$K cv$cv TRAIN FAIL"
done
echo "=== PSIR fold $K DONE  $(date '+%F %H:%M:%S') ==="
