#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Comma-separated stages: train,test,spe. All stages run by default.
STAGES="${STAGES:-train,test,spe}"
# Comma-separated locked evaluation cohorts: internal,external.
TARGETS="${TARGETS:-internal,external}"
TRAIN_GPU="${TRAIN_GPU:-0}"
TEST_GPU="${TEST_GPU:-0}"
SPE_VISIBLE_GPUS="${SPE_VISIBLE_GPUS:-0}"
SPE_DEVICES="${SPE_DEVICES:-cuda:0}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-2000}"
EXTERNAL_TEST_CSV="${EXTERNAL_TEST_CSV:-datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv}"
INTERNAL_TEST_ROOT="${INTERNAL_TEST_ROOT:-datasets/Diagnosis/MIL}"
ALLOW_COHORT_OVERLAP="${ALLOW_COHORT_OVERLAP:-0}"

TRAIN_CONFIGS=(
  configs/Diagnosis/MIL/AB_MIL.yaml
  configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
  configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
  configs/Diagnosis/MIL/TRANS_MIL.yaml
  configs/Diagnosis/MIL/WIKG_MIL.yaml
  configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
  configs/Diagnosis/MIL/AEM_MIL.yaml
  configs/Diagnosis/MIL/MICO_MIL.yaml
  configs/Diagnosis/MIL/MSM_MIL.yaml
  configs/Diagnosis/MIL/TDA_MIL.yaml
  configs/Diagnosis/MIL/GDF_MIL.yaml
  configs/Diagnosis/MIL/RRT_MIL.yaml
  configs/Diagnosis/MIL/DS_MIL.yaml
  configs/Diagnosis/MIL/DTFD_MIL.yaml
  configs/Diagnosis/MIL/MEAN_MIL.yaml
  configs/Diagnosis/MIL/MAX_MIL.yaml
)

SPE_CONFIGS=(
  configs/Diagnosis/SPE/hierarchical_spe.yaml
  configs/Diagnosis/SPE/diversity_veto.yaml
  configs/Diagnosis/SPE/sensitivity_constrained.yaml
  configs/Diagnosis/SPE/constrained_linear_stacking.yaml
  configs/Diagnosis/SPE/ra_spe.yaml
)
SPE_NAMES=(
  v3_bacc
  v4_diversity_veto
  v5_sensitivity_constrained
  v6_constrained_linear_stacking
  v7_ra_spe
)

enabled() {
  local list=",$1,"
  local item="$2"
  [[ "$list" == *",$item,"* ]]
}

preflight_external_cohort() {
  local args=(
    --external "$EXTERNAL_TEST_CSV"
    --internal-root "$INTERNAL_TEST_ROOT"
  )
  if [[ "$ALLOW_COHORT_OVERLAP" == "1" ]]; then
    args+=(--allow-overlap)
  fi
  python -u scripts/Diagnosis/spe/check_cohort_overlap.py "${args[@]}"
}

run_training() {
  echo "[stage=train] Training ${#TRAIN_CONFIGS[@]} MIL configurations on physical GPU ${TRAIN_GPU}"
  for config in "${TRAIN_CONFIGS[@]}"; do
    echo "[train] $config"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u train_mil.py --yaml_path "$config"
  done
}

run_model_tests() {
  if enabled "$TARGETS" internal; then
    echo "[stage=test] Testing all best checkpoints on the locked internal cohort"
    CUDA_VISIBLE_DEVICES="$TEST_GPU" python -u scripts/Diagnosis/spe/test_best_checkpoints.py \
      --test-dataset-root "$INTERNAL_TEST_ROOT" \
      --target-name internal \
      --device cuda:0
  fi
  if enabled "$TARGETS" external; then
    echo "[stage=test] Testing all best checkpoints on the locked external cohort"
    CUDA_VISIBLE_DEVICES="$TEST_GPU" python -u scripts/Diagnosis/spe/test_best_checkpoints.py \
      --test-dataset-csv "$EXTERNAL_TEST_CSV" \
      --target-name external \
      --device cuda:0
  fi
}

summarize_spe() {
  local output_name="$1"
  local target="$2"
  local args=(
    --predictions "result/Diagnosis/SPE/${output_name}/spe_predictions.csv"
    --output-dir "result/Diagnosis/SPE/${output_name}/performance"
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS"
  )
  if [[ "$target" == "external" ]]; then
    args+=(--center-metadata "$EXTERNAL_TEST_CSV")
  else
    args+=(--skip-center)
  fi
  python -u scripts/Diagnosis/spe/summarize_performance.py "${args[@]}"
}

run_spe_target() {
  local target="$1"
  local suffix=""
  local base_output="result/Diagnosis/SPE/v3_bacc"
  local base_args=()
  if [[ "$target" == "external" ]]; then
    suffix="_external"
    base_output="result/Diagnosis/SPE/v3_bacc_external"
    base_args=(--test-dataset-csv "$EXTERNAL_TEST_CSV" --output-name v3_bacc_external)
  fi

  echo "[stage=spe] Preflight for ${target} SPE"
  CUDA_VISIBLE_DEVICES="$SPE_VISIBLE_GPUS" python -u scripts/Diagnosis/spe/run.py \
    --spe-config "${SPE_CONFIGS[0]}" \
    --preflight \
    "${base_args[@]}"

  echo "[stage=spe] Full member inference and v3 aggregation for ${target}"
  CUDA_VISIBLE_DEVICES="$SPE_VISIBLE_GPUS" python -u scripts/Diagnosis/spe/run.py \
    --spe-config "${SPE_CONFIGS[0]}" \
    --devices "$SPE_DEVICES" \
    "${base_args[@]}"

  summarize_spe "v3_bacc${suffix}" "$target"

  for index in 1 2 3 4; do
    local output_name="${SPE_NAMES[$index]}${suffix}"
    local variant_args=(
      --spe-config "${SPE_CONFIGS[$index]}"
      --refit-from "$base_output"
    )
    if [[ "$target" == "external" ]]; then
      variant_args+=(--test-dataset-csv "$EXTERNAL_TEST_CSV" --output-name "$output_name")
    fi
    echo "[stage=spe] ${output_name} from cached ${base_output} member probabilities"
    python -u scripts/Diagnosis/spe/run.py "${variant_args[@]}"
    summarize_spe "$output_name" "$target"
  done
}

if enabled "$TARGETS" external && { enabled "$STAGES" test || enabled "$STAGES" spe; }; then
  preflight_external_cohort
fi

if enabled "$STAGES" train; then
  run_training
fi
if enabled "$STAGES" test; then
  run_model_tests
fi
if enabled "$STAGES" spe; then
  if enabled "$TARGETS" internal; then
    run_spe_target internal
  fi
  if enabled "$TARGETS" external; then
    run_spe_target external
  fi
fi

echo "All requested Diagnosis MIL/SPE stages completed."
