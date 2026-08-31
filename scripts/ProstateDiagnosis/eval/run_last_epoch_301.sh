#!/bin/bash
set -e
cd /NAS3/lbliao/Code-138/MIL_BASELINE
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python
EXT_DIR=datasets/ProstateDiagnosis/DataAnalysis/external_test
VERSION_DIR=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center
RESULT_ROOT=${VERSION_DIR}/external_test_last_epoch

for fold in 1 2 3 4 5; do
  gpu=$((fold - 1))
  latest_dir=$(ls -td ${VERSION_DIR}/fold_${fold}/ProstateDiagnosis_uni2_3center_fold${fold}/AB_MIL/seed_42_*/fold_1/ 2>/dev/null | head -1)
  yaml=${latest_dir}fold_${fold}.yaml
  ckpt=$(ls -t ${latest_dir}Last_EPOCH_*.pth 2>/dev/null | head -1)
  echo "fold ${fold}: using LAST checkpoint ${ckpt}"
  log_dir=${RESULT_ROOT}/301/fold_${fold}
  CUDA_VISIBLE_DEVICES=${gpu} ${PY} -u test_mil.py \
    --yaml_path "${yaml}" \
    --test_dataset_csv "${EXT_DIR}/external_test_301.csv" \
    --model_weight_path "${ckpt}" \
    --test_log_dir "${log_dir}" \
    --device cuda:0 \
    > "logs/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center/last_epoch_301_fold${fold}.log" 2>&1
  echo "fold ${fold}: done -> ${log_dir}"
done
echo "ALL_LAST_EPOCH_301_DONE"
