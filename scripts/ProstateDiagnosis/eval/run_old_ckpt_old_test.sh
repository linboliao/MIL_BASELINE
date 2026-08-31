#!/bin/bash
# Internal test (old, includes 省立) evaluation using the OLD (pre-SL-merge) checkpoints.
set -e
cd /NAS3/lbliao/Code-138/MIL_BASELINE
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python
TEST_CSV=datasets/ProstateDiagnosis/DataAnalysis/internal_test/old_internal_test_plain.csv
CKPT_DIR=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_no_extsl/seed_42_2026-08-28-18-47
RESULT_ROOT=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_no_extsl/internal_test

for fold in 1 2 3 4 5; do
  gpu=$((fold - 1))
  yaml=${CKPT_DIR}/fold_${fold}/fold_${fold}.yaml
  ckpt=$(ls -t ${CKPT_DIR}/fold_${fold}/Best_EPOCH_*.pth 2>/dev/null | head -1)
  echo "fold ${fold}: using OLD checkpoint ${ckpt}"
  log_dir=${RESULT_ROOT}/fold_${fold}
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} -u test_mil.py \
    --yaml_path "${yaml}" \
    --test_dataset_csv "${TEST_CSV}" \
    --model_weight_path "${ckpt}" \
    --test_log_dir "${log_dir}" \
    --device cuda:0 \
    > "logs/Prostate_Diagnosis/AB_MIL_uni2_5fold_sl_dev_only/old_internal_persample_fold${fold}.log" 2>&1
  echo "fold ${fold}: done -> ${log_dir}"
done
echo "ALL_OLD_INTERNAL_EVAL_DONE"
