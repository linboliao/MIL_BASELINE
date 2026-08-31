#!/bin/bash
# Internal test (new, no-省立) evaluation using the NEW (SL-merged) checkpoints.
set -e
cd /NAS3/lbliao/Code-138/MIL_BASELINE
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python
TEST_CSV=datasets/ProstateDiagnosis/DataAnalysis/internal_test/internal_test_plain.csv
CKPT_DIR=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_sl_dev_only/seed_42_2026-08-28-21-29
RESULT_ROOT=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_sl_dev_only/internal_test

for fold in 1 2 3 4 5; do
  gpu=$((fold - 1))
  yaml=${CKPT_DIR}/fold_${fold}/fold_${fold}.yaml
  ckpt=$(ls -t ${CKPT_DIR}/fold_${fold}/Best_EPOCH_*.pth 2>/dev/null | head -1)
  echo "fold ${fold}: using checkpoint ${ckpt}"
  log_dir=${RESULT_ROOT}/fold_${fold}
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} -u test_mil.py \
    --yaml_path "${yaml}" \
    --test_dataset_csv "${TEST_CSV}" \
    --model_weight_path "${ckpt}" \
    --test_log_dir "${log_dir}" \
    --device cuda:0 \
    > "logs/Prostate_Diagnosis/AB_MIL_uni2_5fold_sl_dev_only/internal_persample_fold${fold}.log" 2>&1
  echo "fold ${fold}: done -> ${log_dir}"
done
echo "ALL_INTERNAL_EVAL_DONE"
