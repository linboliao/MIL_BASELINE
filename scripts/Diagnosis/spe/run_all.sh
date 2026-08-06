#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/lbliao/anaconda3/envs/clam/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# =============================================================================
# 1. Train every configured MIL model (five folds per command)
# =============================================================================

CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/AB_MIL.yaml
CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/TRANS_MIL.yaml
CUDA_VISIBLE_DEVICES=3 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/WIKG_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/AEM_MIL.yaml
CUDA_VISIBLE_DEVICES=3 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MICO_MIL.yaml
CUDA_VISIBLE_DEVICES=3 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MSM_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/TDA_MIL.yaml
CUDA_VISIBLE_DEVICES=7 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/GDF_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/RRT_MIL.yaml
CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=3 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/DTFD_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MEAN_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MAX_MIL.yaml

# =============================================================================
# 2. Test every model on the locked internal cohort
#    Each command discovers the latest complete run, tests five best
#    checkpoints through test_mil.py, and averages their probabilities.
# =============================================================================

CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/AB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/TRANS_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/WIKG_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/AEM_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/MICO_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/MSM_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/TDA_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/GDF_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/RRT_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/DTFD_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/MEAN_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-root datasets/Diagnosis/MIL --target-name internal --device cuda:0 --configs configs/Diagnosis/MIL/MAX_MIL.yaml

# =============================================================================
# 3. Test every model on the locked external cohort
# =============================================================================

# This command intentionally stops if an external patient overlaps the internal
# cohort. Resolve cohort ownership first; --allow-overlap is diagnostic only.
python -u scripts/Diagnosis/spe/check_cohort_overlap.py --external datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --internal-root datasets/Diagnosis/MIL

CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/AB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/TRANS_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/WIKG_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/AEM_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/MICO_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/MSM_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/TDA_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/GDF_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/RRT_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/DTFD_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/MEAN_MIL.yaml
CUDA_VISIBLE_DEVICES=0 python -u scripts/Diagnosis/spe/test_best_checkpoints.py --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --target-name external --device cuda:0 --configs configs/Diagnosis/MIL/MAX_MIL.yaml

# =============================================================================
# 4. Internal SPE v3-v7
# =============================================================================

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --preflight
CUDA_VISIBLE_DEVICES=1,3,4,7 python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --devices cuda:0,cuda:1,cuda:2,cuda:3
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v3_bacc/spe_predictions.csv --output-dir result/Diagnosis/SPE/v3_bacc/performance --bootstrap-iterations 2000 --skip-center

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/diversity_veto.yaml --refit-from result/Diagnosis/SPE/v3_bacc
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v4_diversity_veto/spe_predictions.csv --output-dir result/Diagnosis/SPE/v4_diversity_veto/performance --bootstrap-iterations 2000 --skip-center

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/sensitivity_constrained.yaml --refit-from result/Diagnosis/SPE/v3_bacc
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v5_sensitivity_constrained/spe_predictions.csv --output-dir result/Diagnosis/SPE/v5_sensitivity_constrained/performance --bootstrap-iterations 2000 --skip-center

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/constrained_linear_stacking.yaml --refit-from result/Diagnosis/SPE/v3_bacc
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v6_constrained_linear_stacking/spe_predictions.csv --output-dir result/Diagnosis/SPE/v6_constrained_linear_stacking/performance --bootstrap-iterations 2000 --skip-center

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/ra_spe.yaml --refit-from result/Diagnosis/SPE/v3_bacc
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v7_ra_spe/spe_predictions.csv --output-dir result/Diagnosis/SPE/v7_ra_spe/performance --bootstrap-iterations 2000 --skip-center

# =============================================================================
# 5. External SPE v3-v7
# =============================================================================

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v3_bacc_external --preflight
CUDA_VISIBLE_DEVICES=1,3,4,7 python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v3_bacc_external --devices cuda:0,cuda:1,cuda:2,cuda:3
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v3_bacc_external/spe_predictions.csv --center-metadata datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-dir result/Diagnosis/SPE/v3_bacc_external/performance --bootstrap-iterations 2000

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/diversity_veto.yaml --refit-from result/Diagnosis/SPE/v3_bacc_external --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v4_diversity_veto_external
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v4_diversity_veto_external/spe_predictions.csv --center-metadata datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-dir result/Diagnosis/SPE/v4_diversity_veto_external/performance --bootstrap-iterations 2000

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/sensitivity_constrained.yaml --refit-from result/Diagnosis/SPE/v3_bacc_external --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v5_sensitivity_constrained_external
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v5_sensitivity_constrained_external/spe_predictions.csv --center-metadata datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-dir result/Diagnosis/SPE/v5_sensitivity_constrained_external/performance --bootstrap-iterations 2000

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/constrained_linear_stacking.yaml --refit-from result/Diagnosis/SPE/v3_bacc_external --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v6_constrained_linear_stacking_external
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v6_constrained_linear_stacking_external/spe_predictions.csv --center-metadata datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-dir result/Diagnosis/SPE/v6_constrained_linear_stacking_external/performance --bootstrap-iterations 2000

python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/ra_spe.yaml --refit-from result/Diagnosis/SPE/v3_bacc_external --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-name v7_ra_spe_external
python -u scripts/Diagnosis/spe/summarize_performance.py --predictions result/Diagnosis/SPE/v7_ra_spe_external/spe_predictions.csv --center-metadata datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv --output-dir result/Diagnosis/SPE/v7_ra_spe_external/performance --bootstrap-iterations 2000
