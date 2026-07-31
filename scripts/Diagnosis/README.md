# Diagnosis WSI representation workflow

Run the following commands from the repository root on Linux after activating
the Python environment.

## 1. Train

```bash
bash scripts/Diagnosis/train_wsi_representation.sh
```

The script explicitly lists 12 unique training commands. `Mag/20x` is the same
representation as `PFM/h-optimus-1`, so it reuses that five-fold training run.
Training artifacts are saved under:

```text
result/Diagnosis/Mag/<DATASET_NAME>/AB_MIL/<seed_time>/fold_<1-5>/
```

## 2. Generate OOF predictions

```bash
bash scripts/Diagnosis/run_wsi_representation_oof.sh
```

The script explicitly lists all 13 evaluation configs. For each config,
`oof_mil.py` finds the latest complete five-fold run, loads each fold's
`Best_EPOCH_*.pth`, predicts only that fold's validation slides, and verifies
that every development slide is predicted exactly once.

```text
result/Diagnosis/Mag/OOF/
  <comparison>/<variant>/<training_run>/
    fold_<1-5>/predictions.csv
    oof_predictions.csv
    manifest.json
```

Existing fold predictions are reused. Add `--overwrite` to an individual
command in the Bash file when that experiment must be recomputed.

## 3. Summarize performance

```bash
bash scripts/Diagnosis/summarize_wsi_representation_oof.sh
```

Outputs under `result/Diagnosis/Mag/Statistics/`:

- `oof_performance_summary.csv`: pooled OOF metrics, patient-bootstrap 95% CIs,
  and within-comparison ranks.
- `oof_fold_metrics.csv`: metrics for each validation fold.
- `oof_specimen_metrics.csv`: CNB, TURP, and RP subgroup metrics.
- `oof_pairwise_patient_bootstrap.csv`: paired differences and 95% CIs.
- `oof_sources.csv`: prediction files and training runs used in statistics.
- `statistics_manifest.json`: threshold and bootstrap settings.

The statistics workflow reuses `Mag/10x` as the `Stains/Unnormalized`
baseline. Bootstrap resampling is clustered by patient rather than by slide.

## Source and result directories

Python and Bash workflow code belongs in `scripts/Diagnosis/`. Generated
checkpoints, OOF predictions, CSV summaries, and manifests belong in
`result/Diagnosis/Mag/`. Keeping source out of `result/` makes the result
directory safe to archive, replace, or regenerate.
