# Diagnosis workflows

Diagnosis scripts are grouped by experiment instead of by file type. Run the
commands below from any directory on Linux after activating the Python
environment; each shell wrapper resolves the repository root itself.

```text
scripts/Diagnosis/
├── spe/                         # Stability-Prioritized Ensemble
│   ├── run.py                   # ensemble inference entry point
│   ├── run.sh                   # configured inference wrapper
│   ├── train_mil.sh             # core architecture command list
│   ├── train_supplementary_mil.sh
│   ├── summarize_performance.py
│   └── analyze_tumor_content.py
└── wsi_representation/          # PFM, magnification and stain comparisons
    ├── train.sh
    ├── generate_oof.py
    ├── run_oof.sh
    ├── summarize_oof.py
    ├── summarize_oof.sh
    └── plots/
        ├── common.py
        ├── plot_foundation_models.py
        ├── plot_magnification.py
        └── plot_stain_normalization.py
```

## Stability-Prioritized Ensemble

Train the configured MIL architectures, validate the ensemble inputs, and run
inference:

```bash
bash scripts/Diagnosis/spe/train_mil.sh
bash scripts/Diagnosis/spe/train_supplementary_mil.sh
python -u scripts/Diagnosis/spe/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml \
  --preflight
bash scripts/Diagnosis/spe/run.sh
```

Post-processing entry points:

```bash
python -u scripts/Diagnosis/spe/summarize_performance.py --help
python -u scripts/Diagnosis/spe/analyze_tumor_content.py --help
```

See `configs/Diagnosis/SPE/README.md` for ensemble details.

### Unified training and locked evaluation

The following wrapper trains all 16 configured MIL models, evaluates every
model's five best checkpoints on the internal and external cohorts, runs all
five configured SPE variants, and writes performance summaries:

```bash
bash scripts/Diagnosis/spe/run_all.sh
```

`run_all.sh` is intentionally a plain, sequential command list, matching the
style of `train_mil.sh`: it contains no arrays, loops, functions, stage logic,
or runtime argument routing. Comment out sections or individual commands when
only part of the workflow is required, and edit each command's visible GPU ID
directly when adapting it to another server.

Best-checkpoint model results are stored below
`result/Diagnosis/ModelTest/{internal,external}/`. SPE versions are grouped as
`result/Diagnosis/SPE/{Internal,External}/<experiment>/`; external experiment
names carry the `_external` suffix. The performance summarizer automatically reads
the locked decision threshold from each SPE manifest, which is required for
the sensitivity-constrained version.

The explicit external/internal overlap-audit command appears immediately
before the external model-test section and stops the remaining commands if an
overlap is present. Resolve cohort ownership before final paper evaluation;
adding `--allow-overlap` is suitable only for a knowingly non-paper diagnostic
run.

## WSI representation comparison

### 1. Train

```bash
bash scripts/Diagnosis/wsi_representation/train.sh
```

The script lists 12 unique training commands. `Mag/20x` is the same
representation as `PFM/h-optimus-1`, so it reuses that five-fold training run.

### 2. Generate OOF predictions

```bash
bash scripts/Diagnosis/wsi_representation/run_oof.sh
```

For each enabled config, `generate_oof.py` finds the latest complete five-fold
run, loads each fold's `Best_EPOCH_*.pth`, predicts only that fold's validation
slides, and verifies that every development slide is predicted exactly once.
Existing fold predictions are reused unless `--overwrite` is added to a
command in the wrapper.

### 3. Summarize and plot

```bash
bash scripts/Diagnosis/wsi_representation/summarize_oof.sh
python -u scripts/Diagnosis/wsi_representation/plots/plot_foundation_models.py
python -u scripts/Diagnosis/wsi_representation/plots/plot_magnification.py
python -u scripts/Diagnosis/wsi_representation/plots/plot_stain_normalization.py
```

Generated checkpoints, predictions, summaries, manifests, and figures belong
under `result/Diagnosis/`; source code stays under `scripts/Diagnosis/`.
