# PFM LOCO shell recipes and independent execution utilities

Scientific configs and frozen splits stay on NAS at their original locations.
There is no Study registry, fixed PFM matrix or directory-discovery policy in the
training wrapper. Experiment-specific scripts live below `scripts/experiments/`:

```text
scripts/experiments/
  run_fold.py                   # generic: explicit config + one wide dataset CSV
  pfm_loco/
    prepare_split.py            # optional: explicit role CSVs -> wide input CSVs
    stage.py                    # explicit dataset CSVs -> feature copies + staged CSVs
    clean_stage.py              # explicit manifest -> owned copy cleanup
    audit.py                    # binary LOCO held-out coverage check
    run.sh                      # editable PFM LOCO shell recipe
    examples/nas_paths.json     # documentation only, not a runtime profile
```

## One fold, with no staging dependency

From the repository, use the mapped Python for the current server:

```bash
PY=/data12/jing/anaconda3/envs/PrePATH/bin/python   # 138
# 195: /home/lbliao/anaconda3/envs/clam/bin/python, with its lib/ in LD_LIBRARY_PATH
"$PY" -B -m scripts.experiments.run_fold \
  --config /absolute/config.yaml \
  --dataset /absolute/fold_train_val.csv \
  --output /NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/my-study/my-run/fold1 \
  --gpu 1
```

The dataset is a single existing MIL_BASELINE wide-format CSV, not a directory.
The wrapper does not know PFM names, domain names, model architecture, feature
staging or the original NAS directory structure. The existing model factory and
training implementation determine supported models. It requires nonempty train
and val roles because it selects a best validation checkpoint. It snapshots the
supplied config and dataset, overrides only runtime paths/device and checkpoint
retention, then invokes the existing train entry point. Scientific parameters
including checkpoint selection remain from the supplied config. Optional
`--tie-break val_loss` explicitly enables that policy; it is not forced globally.

`--test-dataset /absolute/test.csv` adds a separate post-training test with the
selected checkpoint. In that case training CSV test columns MUST be empty.
Without this flag, a supplied training CSV may retain its native epoch test role.
`--validate-only` reads input paths/labels without creating outputs or training.
The wrapper validates paths and label pairs, not tensor dimensions or scientific
feature provenance; those must be checked before launch. Execution requires a
clean committed checkout and a unique output under project provenance. Long
runs belong in a dedicated tmux session after inspecting actual resources.

## Optional split preparation

`pfm_loco.prepare_split` accepts explicit `--train`, `--selection`, `--test`,
optional `--calibration` CSV paths and a new `--output` directory. It checks
patient/slide separation and preserves source snapshots. It maps selection to
val, leaves epoch-test columns empty, and emits independent `train_val.csv` and
`test.csv`. Existing compatible wide CSVs need no preparation. No source file is
modified. Calibration is checked and preserved, not used for training or test.

## Independent staging and cleanup

```bash
"$PY" -B -m scripts.experiments.pfm_loco.stage \
  --dataset /absolute/fold1.csv --dataset /absolute/fold2.csv \
  --cache /data2/lbliao/pfm_loco_cache/my-run/conch
# Outputs: cache/datasets/0.csv, cache/datasets/1.csv, cache/manifest.json
# Each staged CSV is now an ordinary --dataset input for run_fold.
```

Staging deduplicates feature paths across folds, verifies byte-identical PT/H5
copies, and creates new CSVs pointing at them. It does not load tensors, convert
precision, edit configs, or start training. Original sample identities are added
to staged CSVs. Cache creation is exclusive; failed/partial caches remain intact.
Use a local node-specific staging path, never original NAS feature storage.
There must be payload capacity plus at least 1 GiB free. This is a minimum copy
check, not a training RAM or disk allocation guarantee.

```bash
# Read-only inspection is the default.
"$PY" -B -m scripts.experiments.pfm_loco.clean_stage --manifest /absolute/cache/manifest.json
# Explicit cleanup, only after all consumers finished successfully:
"$PY" -B -m scripts.experiments.pfm_loco.clean_stage \
  --manifest /absolute/cache/manifest.json --execute --receipt /absolute/attempt/cleanup.jsonl
```

Cleanup is limited to manifest-listed unchanged feature copies on the same host,
under local /data2/lbliao, /data14/lbliao or /dev/shm staging/cache paths. Symlink,
path, size and hash checks run before deletion. No recursive directory deletion
is used. Manifest, generated CSVs, receipts and directories remain for provenance.
Do not reuse the cache while cleanup is running. A partially completed cleanup
requires inspection, not blind rerun. No cleanup is performed during these tests.

## Shell composition

The editable `pfm_loco/run.sh` is a recipe, not a Python scheduler:

```text
for each model:
    prepare inputs if necessary
    stage inputs once
    train fold 1 (+ independent test and audit)
    train fold 2 (+ independent test and audit)
    ...
    clean staging
```

```bash
RUN_ID=seed42-conch-repro-r1 GPU=1 LOCO_DOMAINS='SL XC' \
  bash scripts/experiments/pfm_loco/run.sh conch
```

Model arguments and LOCO_DOMAINS control this example's matrix. SOURCE,
OUTPUT_PARENT and CACHE_PARENT only affect the shell recipe. You can replace its
loops with explicit lines for completely different config/dataset pairs. The
Python utilities remain unchanged. With `set -euo pipefail`, failure stops before
cleanup, preserving cache and outputs. Nothing is cleaned by an EXIT trap.
Separate servers need distinct run/cache IDs and non-overlapping outputs.

The old `python -m orchestration` interface is replaced by these explicit tools.
The script preserves the standardized LOCO design when given its original inputs;
this does not certify bitwise equivalence to the old training implementation.
