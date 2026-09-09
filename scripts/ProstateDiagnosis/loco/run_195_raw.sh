#!/usr/bin/env bash
# 8-PFM robustness matrix — SERVER 195, RAW staging (no fp16 convert).
#
# The fp16 path (torch.load -> .half() -> torch.save) is GIL-bound and 195 is at
# load ~80, so staging crawls. This variant just copies the .pt verbatim (fp32,
# pure I/O) -> fast even under CPU contention, at the cost of ~2x local disk.
#
#   ssh 195 ; tmux new -s loco_mstar
#   bash scripts/ProstateDiagnosis/loco/run_195_raw.sh mstar
#
# Cache dir is chosen by free space:
#   /data2/lbliao/loco_cache      if it has >= $LOCO_NEED_GB free   (fp32)
#   /dev/shm/loco_cache_$USER     else, if shm has >= $LOCO_NEED_GB (fp32, RAM!)
#   /dev/shm + fp16               else (shm can't hold fp32; warns)
# A /dev/shm cache is wiped on exit (trap) — it's RAM.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
cd "$REPO"
echo "repo $REPO @ $(git rev-parse --short HEAD)"

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(mstar)
NEED_GB=${LOCO_NEED_GB:-160}                 # mstar clean cohort ~150 GB fp32
freg () { df -PBG "$1" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4+0}'; }

D2=/data2/lbliao/loco_cache
SHM=/dev/shm/loco_cache_${USER:-lbliao}
d2f=$(freg /data2/lbliao 2>/dev/null || freg /data2); shf=$(freg /dev/shm)
if [ "${d2f:-0}" -ge "$NEED_GB" ]; then
  export LOCO_CACHE=$D2 LOCO_RAW=1
  echo "cache -> $D2  (${d2f}G free)   mode: RAW fp32"
elif [ "${shf:-0}" -ge "$NEED_GB" ]; then
  export LOCO_CACHE=$SHM LOCO_RAW=1; SHM_USED=1
  echo "cache -> $SHM  (${shf}G free)   mode: RAW fp32 (RAM — wiped on exit)"
else
  export LOCO_CACHE=$SHM; SHM_USED=1
  echo "!! /data2 ${d2f}G and /dev/shm ${shf}G both < ${NEED_GB}G for fp32."
  echo "!! falling back to /dev/shm + fp16 (~1/2 size).  cache -> $SHM"
fi
[ -n "${SHM_USED:-}" ] && trap 'echo "wiping $LOCO_CACHE"; rm -rf "$LOCO_CACHE"' EXIT

export LOCO_PYTHON=${LOCO_PYTHON:-/home/lbliao/anaconda3/envs/clam/bin/python}
export LOCO_LD_LIBRARY_PATH=${LOCO_LD_LIBRARY_PATH:-/home/lbliao/anaconda3/envs/clam/lib}
export LOCO_ALLOC_CONF=${LOCO_ALLOC_CONF:-expandable_segments:True}
export LOCO_GPU_BASE=${LOCO_GPU_BASE:-0}
export LOCO_LOGD=${LOCO_LOGD:-/home/lbliao/mil_runs/loco}
mkdir -p "$LOCO_LOGD"
echo "models: ${MODELS[*]}"
bash "$HERE/loco_run.sh" "${MODELS[@]}" \
  2>&1 | tee "$LOCO_LOGD/RUN_matrix_195raw_$(date +%Y%m%d_%H%M).log"
