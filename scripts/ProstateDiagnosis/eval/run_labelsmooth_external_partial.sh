#!/bin/bash
# External test evaluation (301 + YNZL) for label-smoothing folds 1-4 (fold5 still training).
set -e
cd /NAS3/lbliao/Code-138/MIL_BASELINE
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python
EXT_DIR=datasets/ProstateDiagnosis/DataAnalysis/external_test
VERSION_DIR=result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center_labelsmooth
RESULT_ROOT=${VERSION_DIR}/external_test

for fold in 1 2 3 4; do
  gpu=$((fold - 1))
  latest_dir=$(ls -td ${VERSION_DIR}/fold_${fold}/ProstateDiagnosis_uni2_3center_labelsmooth_fold${fold}/AB_MIL/seed_42_*/fold_1/ 2>/dev/null | head -1)
  yaml=${latest_dir}fold_${fold}.yaml
  ckpt=$(ls -t ${latest_dir}Best_EPOCH_*.pth 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then
    echo "fold ${fold}: NO CHECKPOINT FOUND under ${latest_dir}, skipping"
    continue
  fi
  echo "fold ${fold}: using checkpoint ${ckpt}"
  for ext in 301 ynzl; do
    test_csv=${EXT_DIR}/external_test_${ext}.csv
    log_dir=${RESULT_ROOT}/${ext}/fold_${fold}
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} -u test_mil.py \
      --yaml_path "${yaml}" \
      --test_dataset_csv "${test_csv}" \
      --model_weight_path "${ckpt}" \
      --test_log_dir "${log_dir}" \
      --device cuda:0 \
      > "logs/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center_labelsmooth/external_${ext}_fold${fold}.log" 2>&1 &
  done
done
wait
echo "ALL_LABELSMOOTH_EXTERNAL_PARTIAL_DONE"
