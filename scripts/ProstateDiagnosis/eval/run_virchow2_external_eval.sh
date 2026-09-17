#!/usr/bin/env bash
# External-cohort eval (301 + ynzl) for the virchow2 5-fold run.
# Each fold's Best_EPOCH checkpoint is scored on both cohorts. fold k -> GPU k-1.
# test_mil.py loads the test set with preload=True (whole cohort into RAM, ~26-31 GB
# fp16 each, under the 40 GB cap), so MAXPAR is kept low to leave RAM for the
# still-resident /dev/shm training cache.
#
# Prereqs: stage_virchow2_external_fp16.py + build_virchow2_external_csvs_195.py
#
# Usage: run_virchow2_external_eval.sh [RUN_DIR]
#   RUN_DIR defaults to the newest run_* under the fp16local result tree.
set -uo pipefail

REPO=/NAS2/Data1/lbliao/Code-195/MIL_BASELINE
PY=/home/lbliao/anaconda3/envs/maixin/bin/python
MAXPAR=${MAXPAR:-3}
EXT_DIR=$REPO/datasets/ProstateDiagnosis/DataAnalysis/external_test
RES_ROOT=$REPO/result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local

RUN_DIR=${1:-$(ls -dt "$RES_ROOT"/run_* | head -1)}
cd "$REPO"
echo "run dir : $RUN_DIR   (max $MAXPAR concurrent)"
OUT=$RUN_DIR/external_test
mkdir -p "$OUT"

for c in 301 ynzl; do
  [ -f "$EXT_DIR/external_test_${c}_virchow2_fp16.csv" ] || {
    echo "ERROR: missing $EXT_DIR/external_test_${c}_virchow2_fp16.csv - run build_virchow2_external_csvs_195.py"; exit 1; }
done

run_one() {
  local k=$1 c=$2
  local gpu=$((k - 1))
  local yaml=$RUN_DIR/fold_${k}/fold_${k}.yaml
  local ckpt
  ckpt=$(ls -t "$RUN_DIR"/fold_${k}/Best_EPOCH_*.pth 2>/dev/null | head -1)
  local ld=$OUT/${c}/fold_${k}
  if [ -z "$ckpt" ]; then echo "fold $k: no checkpoint, skip"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u test_mil.py \
    --yaml_path "$yaml" \
    --test_dataset_csv "$EXT_DIR/external_test_${c}_virchow2_fp16.csv" \
    --model_weight_path "$ckpt" \
    --test_log_dir "$ld" \
    --device cuda:0 \
    > "$OUT/${c}_fold${k}.log" 2>&1 &
  echo "launched fold $k / $c -> GPU $gpu  pid $!  (ckpt $(basename "$ckpt"))"
}

running=0
for c in 301 ynzl; do
  for k in 1 2 3 4 5; do
    run_one "$k" "$c"
    running=$((running + 1))
    if [ $running -ge $MAXPAR ]; then wait -n; running=$((running - 1)); fi
  done
done
wait

echo "----"
"$PY" - "$OUT" <<'PYEOF'
import glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             f1_score, recall_score, precision_score, cohen_kappa_score,
                             confusion_matrix)

out = sys.argv[1]


def scores(y, p1):
    yhat = (np.asarray(p1) >= 0.5).astype(int)
    y = np.asarray(y).astype(int)
    return {
        'acc': accuracy_score(y, yhat),
        'bacc': balanced_accuracy_score(y, yhat),
        'auc': roc_auc_score(y, p1) if len(set(y)) > 1 else float('nan'),
        'macro_f1': f1_score(y, yhat, average='macro'),
        'sensitivity': recall_score(y, yhat, pos_label=1, zero_division=0),
        'specificity': recall_score(y, yhat, pos_label=0, zero_division=0),
        'precision': precision_score(y, yhat, pos_label=1, zero_division=0),
        'quadratic_kappa': cohen_kappa_score(y, yhat, weights='quadratic'),
        'cm': confusion_matrix(y, yhat).tolist(),
    }


summary = {}
for c in ('301', 'ynzl'):
    per_fold = []
    dfs = []
    for k in range(1, 6):
        f = os.path.join(out, c, f'fold_{k}', 'Infer_Result.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        dfs.append(d.set_index('slide_id')[['label', 'prob_1']].rename(columns={'prob_1': f'p{k}'}))
        per_fold.append((k, scores(d['label'], d['prob_1'])))
    if not per_fold:
        print(f'{c}: no results'); continue

    merged = pd.concat(dfs, axis=1)
    label = merged['label'].iloc[:, 0]
    ens_p = merged[[f'p{k}' for k, _ in per_fold]].mean(axis=1)
    ens = scores(label, ens_p)

    print(f'\n=== {c}   n={len(label)}   ({len(per_fold)} folds) ===')
    keys = ['acc', 'bacc', 'auc', 'macro_f1', 'sensitivity', 'specificity', 'quadratic_kappa']
    for key in keys:
        vals = [m[key] for _, m in per_fold]
        print(f'  {key:16s} fold mean {np.nanmean(vals):.4f} +- {np.nanstd(vals):.4f}   '
              f'| ensemble {ens[key]:.4f}')
    print(f'  ensemble confusion matrix [ [TN,FP],[FN,TP] ]: {ens["cm"]}')

    summary[c] = {
        'n': int(len(label)),
        'per_fold': {k: m for k, m in per_fold},
        'ensemble': ens,
    }
    merged.assign(ensemble_prob_1=ens_p).to_csv(os.path.join(out, c, 'ensemble_probs.csv'))

with open(os.path.join(out, 'external_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print(f'\nwrote {os.path.join(out, "external_summary.json")}')
PYEOF
echo "results under: $OUT"
