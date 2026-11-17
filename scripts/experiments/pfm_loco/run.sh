#!/usr/bin/env bash
# Example recipe: stage one PFM, train each supplied split, then clean owned copies.
set -euo pipefail
trap 'printf "Failed at line %s; outputs and staging preserved.\n" "$LINENO" >&2' ERR
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"
case "$(hostname)" in
  server94) PY=/data12/jing/anaconda3/envs/PrePATH/bin/python; NODE=138 ;;
  ubuntu) PY=/home/lbliao/anaconda3/envs/clam/bin/python; NODE=195
    export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:${LD_LIBRARY_PATH:-} ;;
  *) echo 'Use server 138 or 195' >&2; exit 2 ;;
esac
export TMPDIR=/NAS145/linboliao/Projects/ProstateDiagnosis/.agent/nodes/$NODE/tmp
export TMP=$TMPDIR TEMP=$TMPDIR
: "${RUN_ID:?Set a unique RUN_ID}"
: "${GPU:?Set the inspected physical GPU index}"
[[ "$RUN_ID" != */* && "$RUN_ID" != . && "$RUN_ID" != .. ]] || exit 2
[[ $# -gt 0 ]] || { echo 'usage: RUN_ID=... GPU=... bash run.sh conch uni ...' >&2; exit 2; }
SOURCE=${SOURCE:-/NAS145/linboliao/Experiments/ProstateDiagnosis/PFM_LOCO}
OUTPUT_PARENT=${OUTPUT_PARENT:-/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/pfm-loco-prostate-reproduction}
CACHE_PARENT=${CACHE_PARENT:-/data2/lbliao/pfm_loco_cache}
OUTPUT_PARENT=$(readlink -m "$OUTPUT_PARENT")
case "$OUTPUT_PARENT" in
  /NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/*) ;;
  *) echo 'OUTPUT_PARENT must be below project provenance' >&2; exit 2 ;;
esac
RUN=$OUTPUT_PARENT/$RUN_ID
mkdir -p "$OUTPUT_PARENT"
mkdir "$RUN"                         # Refuse to reuse any attempt.
read -r -a DOMAINS <<< "${LOCO_DOMAINS:-SL XC CNB TURP RP}"
[[ ${#DOMAINS[@]} -gt 0 ]] || exit 2
for model in "$@"; do
  [[ "$model" != */* && "$model" != . && "$model" != .. ]] || exit 2
  mkdir "$RUN/$model"
  stage_args=()
  # Optional adaptation only: the training entry point also accepts existing wide CSVs.
  for domain in "${DOMAINS[@]}"; do
    [[ "$domain" != */* && "$domain" != . && "$domain" != .. ]] || exit 2
    split=$SOURCE/$model/$domain/splits
    "$PY" -B -m scripts.experiments.pfm_loco.prepare_split \
      --train "$split/train.csv" --selection "$split/selection.csv" \
      --test "$split/test.csv" --calibration "$split/calibration.csv" \
      --output "$RUN/$model/prepared/$domain"
    stage_args+=(--dataset "$RUN/$model/prepared/$domain/train_val.csv")
  done
  cache=$CACHE_PARENT/$RUN_ID/$model
  # 1. Stage all feature inputs for this model once.
  "$PY" -B -m scripts.experiments.pfm_loco.stage "${stage_args[@]}" --cache "$cache"
  # Preserve the original-to-cache mapping with the attempt before cleanup.
  cp -n "$cache/manifest.json" "$RUN/$model/staging_manifest.json"
  # 2..i. Independent one-fold calls. Add models/domains here without changing Python.
  index=0
  for domain in "${DOMAINS[@]}"; do
    "$PY" -B -m scripts.experiments.run_fold \
      --config "$SOURCE/$model/$domain/configs/model.yaml" \
      --dataset "$cache/datasets/$index.csv" \
      --test-dataset "$RUN/$model/prepared/$domain/test.csv" \
      --output "$RUN/$model/$domain" --gpu "$GPU" --tie-break val_loss
    "$PY" -B -m scripts.experiments.pfm_loco.audit --output "$RUN/$model/$domain"
    index=$((index+1))
  done
  # i+1. Clean ONLY copied features; manifest, adapted CSVs and receipt remain.
  "$PY" -B -m scripts.experiments.pfm_loco.clean_stage \
    --manifest "$cache/manifest.json" --execute --receipt "$RUN/$model/cleanup.jsonl"
done
