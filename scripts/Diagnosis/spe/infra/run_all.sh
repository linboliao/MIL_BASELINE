#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly REPO_ROOT
readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly MIL_CONFIG_ROOT="configs/Diagnosis/MIL_Mix"
readonly INTERNAL_DATASET_ROOT="datasets/Diagnosis/MIL_Mix"
readonly EXTERNAL_DATASET_CSV="datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv"
readonly INTERNAL_SPE_ROOT="result/Diagnosis/SPE/Internal"
readonly EXTERNAL_SPE_ROOT="result/Diagnosis/SPE/External"
readonly BOOTSTRAP_ITERATIONS="${SPE_BOOTSTRAP_ITERATIONS:-2000}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

readonly -a MODEL_NAMES=(
  AB_MIL CLAM_SB_MIL CLAM_MB_MIL TRANS_MIL WIKG_MIL MAMBA2D_MIL AEM_MIL
  MICO_MIL MSM_MIL TDA_MIL GDF_MIL RRT_MIL DS_MIL DTFD_MIL MEAN_MIL MAX_MIL
)

readonly -a TRAINING_JOBS=(
  "1:AB_MIL" "1:CLAM_SB_MIL" "1:CLAM_MB_MIL" "1:TRANS_MIL"
  "3:WIKG_MIL" "4:MAMBA2D_MIL" "4:AEM_MIL" "3:MICO_MIL"
  "3:MSM_MIL" "4:TDA_MIL" "7:GDF_MIL" "4:RRT_MIL"
  "1:DS_MIL" "3:DTFD_MIL" "0:MEAN_MIL" "0:MAX_MIL"
)

train_models() {
  local job gpu model
  for job in "${TRAINING_JOBS[@]}"; do
    IFS=: read -r gpu model <<< "$job"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u train_mil.py \
      --yaml_path "$MIL_CONFIG_ROOT/$model.yaml"
  done
}

test_models() {
  local target_name="$1"
  shift

  local model
  for model in "${MODEL_NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u \
      scripts/Diagnosis/spe/infra/test_best_checkpoints.py \
      "$@" \
      --target-name "$target_name" \
      --device cuda:0 \
      --configs "$MIL_CONFIG_ROOT/$model.yaml"
  done
}

summarize_internal() {
  local experiment_name="$1"
  "$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/summarize_performance.py \
    --predictions "$INTERNAL_SPE_ROOT/$experiment_name/spe_predictions.csv" \
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS" \
    --skip-center
}

run_internal_refit() {
  local variant="$1"
  local experiment_name="$2"
  "$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/run.py \
    --variant "$variant" \
    --refit-from "$INTERNAL_SPE_ROOT/bacc"
  summarize_internal "$experiment_name"
}

summarize_external() {
  local experiment_name="$1"
  "$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/summarize_performance.py \
    --predictions "$EXTERNAL_SPE_ROOT/$experiment_name/spe_predictions.csv" \
    --center-metadata "$EXTERNAL_DATASET_CSV" \
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS"
}

run_external_refit() {
  local variant="$1"
  local experiment_name="$2"
  "$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/run.py \
    --variant "$variant" \
    --refit-from "$EXTERNAL_SPE_ROOT/bacc_external" \
    --test-dataset-csv "$EXTERNAL_DATASET_CSV" \
    --output-name "$experiment_name"
  summarize_external "$experiment_name"
}

# 1. Train every configured MIL model (five folds per command).
#train_models

# 2. Evaluate every model on the locked internal cohort.
#test_models internal --test-dataset-root "$INTERNAL_DATASET_ROOT"

# 3. Evaluate every model on the locked external cohort.
# Stop on patient overlap; --allow-overlap is reserved for diagnostics.
#"$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/check_cohort_overlap.py \
#  --external "$EXTERNAL_DATASET_CSV" \
#  --internal-root "$INTERNAL_DATASET_ROOT"
#test_models external --test-dataset-csv "$EXTERNAL_DATASET_CSV"

# 4. Build and evaluate internal SPE variants.
#"$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/run.py \
#  --variant hierarchical_bacc \
#  --preflight
#CUDA_VISIBLE_DEVICES=1,2,3,4,6 "$PYTHON_BIN" -u \
#  scripts/Diagnosis/spe/infra/run.py \
#  --variant hierarchical_bacc \
#  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
#summarize_internal bacc
#summarize_internal cp_awa_single_anchor

#run_internal_refit diversity_veto diversity_veto
#run_internal_refit sensitivity_constrained sensitivity_constrained
#run_internal_refit constrained_linear_stacking constrained_linear_stacking
#run_internal_refit ra_spe ra_spe
#
## Best-state uses a separate pool with exactly one Best_EPOCH per fold.
#"$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/run.py \
#  --variant best_state_anchor \
#  --preflight
#CUDA_VISIBLE_DEVICES=5,6,7 "$PYTHON_BIN" -u \
#  scripts/Diagnosis/spe/infra/run.py \
#  --variant best_state_anchor \
#  --devices cuda:0,cuda:1,cuda:2
#summarize_internal best_state_anchor
#
## 5. Build and evaluate external SPE variants.
#"$PYTHON_BIN" -u scripts/Diagnosis/spe/infra/run.py \
#  --variant hierarchical_bacc \
#  --test-dataset-csv "$EXTERNAL_DATASET_CSV" \
#  --output-name bacc_external \
#  --preflight
#CUDA_VISIBLE_DEVICES=1,2,3,4,6 "$PYTHON_BIN" -u \
#  scripts/Diagnosis/spe/infra/run.py \
#  --variant hierarchical_bacc \
#  --test-dataset-csv "$EXTERNAL_DATASET_CSV" \
#  --output-name bacc_external \
#  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4
##summarize_external bacc_external
summarize_external cp_awa_single_anchor

#run_external_refit diversity_veto diversity_veto_external
#run_external_refit sensitivity_constrained sensitivity_constrained_external
#run_external_refit constrained_linear_stacking constrained_linear_stacking_external
#run_external_refit ra_spe ra_spe_external
#
#CUDA_VISIBLE_DEVICES=5,6,7 "$PYTHON_BIN" -u \
#  scripts/Diagnosis/spe/infra/run.py \
#  --variant best_state_anchor \
#  --test-dataset-csv "$EXTERNAL_DATASET_CSV" \
#  --output-name best_state_anchor_external \
#  --devices cuda:0,cuda:1,cuda:2
#summarize_external best_state_anchor_external
