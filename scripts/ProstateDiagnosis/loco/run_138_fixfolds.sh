#!/usr/bin/env bash
# Re-run the 3 folds that DataLoader-Bus-errored on 138 during the 8-PFM matrix.
# New (edc0f51) config layout: each mode -> one flat AB_MIL/seed_42_<ts>/fold_<k>/.
# Re-runs the full affected MODES (a few extra fold trainings, cheap vs re-staging):
#
#   gigapath : internal   (省立 folded, 新昌 re-trained)
#   mstar    : internal + type   (省立 + CNB folded, 新昌/RP/TURP re-trained)
#
# The OLD migrated seed_42_2026-09-09-*/ dirs become stale (loco_collect.py picks the
# newest seed dir) — delete them later if you want.
#
#   ssh 138 ; tmux new -s loco_fixfolds
#   bash scripts/ProstateDiagnosis/loco/run_138_fixfolds.sh
#
# ~4-5 h (dominated by re-staging gigapath ~1536d + mstar ~1024d from the NAS).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)   (needs >= edc0f51 for the flat layout)"

export LOCO_CACHE=${LOCO_CACHE:-/data14/lbliao/loco_cache}
export LOCO_PYTHON=${LOCO_PYTHON:-/data12/jing/anaconda3/envs/PrePATH/bin/python}
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/jing/mil_runs/loco}
mkdir -p "$LOCO_LOGD"

LOCO_MODES=internal        bash "$HERE/loco_run.sh" gigapath \
  2>&1 | tee "$LOCO_LOGD/RUN_fixfolds_gigapath_$(date +%Y%m%d_%H%M).log"

LOCO_MODES="internal type" bash "$HERE/loco_run.sh" mstar \
  2>&1 | tee "$LOCO_LOGD/RUN_fixfolds_mstar_$(date +%Y%m%d_%H%M).log"

echo "############################################################"
echo "### fixfolds DONE  $(date '+%F %H:%M:%S')"
echo "### summaries: $LOCO_LOGD/{gigapath_internal,mstar_internal,mstar_type}_summary.txt"
echo "###            + gigapath_internal_external.txt"
echo "### then locally: bash consolidated_results/loco_matrix_run/collect_matrix.sh"
echo "############################################################"
