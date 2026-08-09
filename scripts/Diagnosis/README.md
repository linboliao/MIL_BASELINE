# Diagnosis workflows

Diagnosis scripts are grouped by experiment. Run commands from the repository
root after activating the Python environment.

## Stability-Prioritized Ensemble

The SPE workflow has one Python entrypoint:

```bash
python -u scripts/Diagnosis/spe/run.py --list-variants
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc --preflight
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Custom configurations remain supported:

```bash
python -u scripts/Diagnosis/spe/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml
```

The complete sequential workflow trains all configured MIL models, evaluates
their best checkpoints on the locked internal and external cohorts, builds SPE
variants, and writes performance summaries:

```bash
bash scripts/Diagnosis/spe/run_all.sh
```

`run_all.sh` centralizes shared paths, model lists, GPU assignments, and repeated
internal/external commands in small functions. Adjust its constants and job
arrays when moving the workflow to another server.

Post-processing and audit commands:

```bash
python -u scripts/Diagnosis/spe/check_cohort_overlap.py --help
python -u scripts/Diagnosis/spe/test_best_checkpoints.py --help
python -u scripts/Diagnosis/spe/summarize_performance.py --help
python -u scripts/Diagnosis/spe/summarize_member_performance.py --help
python -u scripts/Diagnosis/spe/analyze_tumor_content.py --help
```

Model-test results are stored below
`result/Diagnosis/ModelTest/{internal,external}/`. SPE results are grouped under
`result/Diagnosis/SPE/{Internal,External}/<experiment>/`. The overlap audit
stops locked external evaluation when it detects internal patients;
`--allow-overlap` is intended only for diagnostics.

See `configs/Diagnosis/SPE/README.md` for variant definitions and methodology.

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
Existing fold predictions are reused unless `--overwrite` is enabled.

### 3. Summarize and plot

```bash
bash scripts/Diagnosis/wsi_representation/summarize_oof.sh
python -u scripts/Diagnosis/wsi_representation/plots/plot_foundation_models.py
python -u scripts/Diagnosis/wsi_representation/plots/plot_magnification.py
python -u scripts/Diagnosis/wsi_representation/plots/plot_stain_normalization.py
```

Generated checkpoints, predictions, summaries, manifests, and figures belong
under `result/Diagnosis/`; source code stays under `scripts/Diagnosis/`.
