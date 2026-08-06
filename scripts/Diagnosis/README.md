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
