#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# 195 环境。GPU 使用前请确认可用。
PY=/home/lbliao/anaconda3/envs/clam/bin/python
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/195/tmp
export TMP=$TMPDIR TEMP=$TMPDIR
CONTRAST=/NAS145/linboliao/Experiments/ProstateDiagnosis/Contrast

# PFM：每行训练一个编码器的五折。覆盖 YAML 中搬迁前的路径。
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/conch.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/conch"
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/uni.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/uni"
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/uni2.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/uni2"
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/virchow2.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/virchow2"
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/h-optimus-1.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/h-optimus-1"
"$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/PFM/mstar.yaml" --options "Dataset.dataset_root_dir=$CONTRAST/datasets/PFM/mstar"

# MIL：按需启用并替换模型 YAML。各模型共用 datasets/MIL 的五折。
# "$PY" -B train_mil.py --yaml_path "$CONTRAST/configs/MIL/AB_MIL.yaml"
