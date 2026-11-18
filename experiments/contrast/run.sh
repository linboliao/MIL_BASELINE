#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(hostname) == server94 && $(id -un) == jing ]]
[[ $# -eq 2 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $2 =~ ^[0-9]+$ ]]
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python
PROJECT=/NAS145/linboliao/Projects/ProstateDiagnosis
export TMPDIR="$PROJECT/.agent/nodes/138/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" CUDA_VISIBLE_DEVICES="$2"
BASE=/NAS145/linboliao/Experiments/ProstateDiagnosis/Contrast
CACHE="/data5/jing/contrast_pfm_cache/$1"
OUT="$BASE/results/PFM/$1"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_head.txt"
git status --short > "$OUT/git_status.txt"
sha256sum experiments/contrast/run.sh > "$OUT/launcher.sha256"
printf 'host=%s\npython=%s\ngpu=%s\ncache=%s\n' "$(hostname)" "$PY" "$2" "$CACHE" > "$OUT/runtime.txt"
for model in conch uni uni2 virchow2 h-optimus-1 mstar; do
    echo "$(date -Is) Staging $model"
    args=()
    for split in "$BASE/datasets/PFM/$model/"*.csv; do args+=(--dataset "$split"); done
    [[ ${#args[@]} -eq 10 ]]
    "$PY" -B -m scripts.experiments.pfm_loco.stage --cache "$CACHE/$model" "${args[@]}"
    mkdir "$OUT/$model"
    cp -n "$CACHE/$model/manifest.json" "$OUT/$model/staging_manifest.json"
    cp -n "$BASE/configs/PFM/$model.yaml" "$OUT/$model/source_config.yaml"
    used=$(nvidia-smi -i "$2" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ $used =~ ^[0-9]+$ && $used -lt 256 ]]
    echo "$(date -Is) Training $model"
    "$PY" -B -u train_mil.py --yaml_path "$OUT/$model/source_config.yaml" --options \
        "Dataset.dataset_root_dir=$CACHE/$model/datasets" "Logs.log_root_dir=$OUT/$model" \
        > "$OUT/$model/train.log" 2>&1
    "$PY" -B - "$OUT/$model" "$CACHE/$model" <<'PYCODE'
import csv, json, math, sys
from pathlib import Path
from scripts.experiments.pfm_loco import clean_stage
out, cache = map(Path, sys.argv[1:])
logs = sorted(out.rglob('Best_Log_*.csv'))
assert len(logs) == 5 and {p.parent.name for p in logs} == {f'fold_{i}' for i in range(1,6)}
for p in logs:
    with p.open() as f:
        row, = csv.DictReader(f)
    assert math.isfinite(float(row['val_macro_auc']))
    fold = p.parent
    assert (fold/f'Best_EPOCH_{int(row["epoch"])}.pth').stat().st_size > 0
    assert any(x.stat().st_size > 0 for x in fold.glob('Last_EPOCH_*.pth'))
    assert fold.parent == logs[0].parent.parent
assert list(logs[0].parent.parent.glob('merge*'))
root = Path('/data5/jing/contrast_pfm_cache')
assert root in cache.parents and cache.resolve() == cache
manifest = cache/'manifest.json'
for item in json.loads(manifest.read_text())['files']:
    src = Path(item['source'])
    assert src.is_file() and src.stat().st_size == item['bytes']
clean_stage.LOCAL_CACHE_ROOTS = (root,)
clean_stage.clean(manifest, out/'cleanup.jsonl')
PYCODE
    echo "$(date -Is) Completed $model"
done
printf '%s\n' "$(date -Is)" > "$OUT/COMPLETE"
