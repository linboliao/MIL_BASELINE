#!/usr/bin/env bash
# ═══ STEP B — same-machine reproducibility validation — PREPARED, NOT AUTO-RUN ═══
# Trains GigaPath internal/留省立 (seed 42) TWICE on the same GPU with the
# determinism patch on; asserts the two Best_Log CSVs are identical row-by-row
# and Best_EPOCH matches. Gate for the full benchmark.
#
#   REPO=/NAS3/lbliao/Code-138/MIL_BASELINE \
#   LOCO_PYTHON=/data12/jing/anaconda3/envs/PrePATH/bin/python \
#   LOCO_CACHE=/data14/lbliao/stdbench_cache  LOCO_GPU_BASE=0 \
#     bash scripts/ProstateDiagnosis/loco/std_benchmark/std_repro_check.sh
set -euo pipefail
export PYTHONHASHSEED=42
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
LOCO=$(cd "$HERE/.." && pwd)
PY=${LOCO_PYTHON:-python}
G=${LOCO_GPU_BASE:-0}
: "${LOCO_CACHE:?set LOCO_CACHE}"
export LOCO_RAW=1 MIL_DETERMINISM=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PROSTATE_FEAT_ROOT="${PROSTATE_FEAT_ROOT:-/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis}"
export LOCO_CACHE PYTHONPATH="$LOCO:${PYTHONPATH:-}"
[ -n "${LOCO_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH=$LOCO_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
cd "$REPO"
$PY -c "import utils.repro_utils" || { echo "patch not applied"; exit 3; }

# Require a fresh task-owned cache; never erase an existing directory.
[ ! -e "$LOCO_CACHE" ] || { echo "STOP: cache already exists: $LOCO_CACHE"; exit 3; }
mkdir -p "$LOCO_CACHE"
for R in A B; do
  D="result/ProstateDiagnosis/DataAnalysis/AB_MIL_gigapath_loco_internal/AB_MIL/seed_42_reprochk_$R"
  [ ! -e "$D" ] || { echo "STOP: existing repro output $D"; exit 3; }
done
$PY "$LOCO/loco_cache.py" --model gigapath || exit 1
$PY "$LOCO/loco_build_folds.py" --model gigapath --mode internal || exit 1
$PY "$LOCO/loco_gen_configs.py" --model gigapath --mode internal --in_dim 1536 || exit 1
Y="configs/ProstateDiagnosis/DataAnalysis/AB_MIL_gigapath_loco_internal.yaml"

for R in A B; do
  echo ">>> repro run $R  $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$G $PY train_mil.py --yaml_path "$Y" \
    --only_fold 1 --run_ts "reprochk_$R" --no_merge 2>&1 | tail -2
done

BA="result/ProstateDiagnosis/DataAnalysis/AB_MIL_gigapath_loco_internal/AB_MIL/seed_42_reprochk_A/fold_1"
BB="result/ProstateDiagnosis/DataAnalysis/AB_MIL_gigapath_loco_internal/AB_MIL/seed_42_reprochk_B/fold_1"
shopt -s nullglob
LA=("$BA"/Log_*.csv); LB=("$BB"/Log_*.csv)
CA=("$BA"/Best_EPOCH_*.pth); CB=("$BB"/Best_EPOCH_*.pth)
[ ${#LA[@]} -eq 1 ] && [ ${#LB[@]} -eq 1 ] &&
[ ${#CA[@]} -eq 1 ] && [ ${#CB[@]} -eq 1 ] ||
{ echo "FAIL — missing or ambiguous logs/checkpoints"; exit 1; }
[ -s "${LA[0]}" ] && [ -s "${LB[0]}" ] ||
{ echo "FAIL — empty logs"; exit 1; }
[ "${CA[0]##*/}" = "${CB[0]##*/}" ] ||
{ echo "FAIL — best epochs differ"; exit 1; }
if diff <(cat "$BA"/Log_*.csv) <(cat "$BB"/Log_*.csv) >/dev/null; then
  echo "PASS — full epoch Log identical between run A and run B"
  ls "$BA"/Best_EPOCH_*.pth "$BB"/Best_EPOCH_*.pth
else
  echo "FAIL — runs diverge. Per-epoch diff:"
  diff <(cat "$BA"/Log_*.csv) <(cat "$BB"/Log_*.csv) | head -40
  echo ">>> determinism patch does not yet cover some op; do NOT start the benchmark"
  exit 1
fi
