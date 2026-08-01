# Hierarchical SPE

This directory configures the manuscript's Stability-Prioritized Ensemble:

1. Within each architecture/fold trajectory, average up to five approximately
   evenly spaced checkpoints from the final validation-stable interval.
2. Average the five fold trajectories within each MIL architecture.
3. Fit non-negative architecture weights on patient-equal OOF development
   predictions, then apply the locked weights to the independent test cohort.

The stable interval is produced during training by `utils/spe_model_utils.py`:
at least five consecutive saved checkpoints with adjacent validation macro-F1
changes no greater than 0.003. If no interval qualifies, the final five saved
checkpoints are used. Therefore all 11 SPE architectures must be trained with:

```yaml
General:
  checkpoint:
    save_mode: every_epoch
    spe:
      metric: macro_f1
      stability_threshold: 0.003
      min_consecutive: 5
      max_checkpoints: 5
```

Run on Linux after all five folds of all 11 architectures have completed:

```bash
python -u scripts/Diagnosis/run_spe.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --preflight
bash scripts/Diagnosis/run_spe.sh
```

For the final paper run, replace every `run_dir: null` with the exact immutable
training-run directory. Automatic latest-run discovery is convenient during
development but is not appropriate for a locked analysis.

Outputs are written to `result/Diagnosis/SPE/hierarchical_spe_v1/`:

- `oof_architecture_predictions.csv`: development OOF architecture matrix.
- `architecture_weights.csv`: fitted non-negative weights and diagnostics.
- `residual_similarity.csv`: patient-weighted architecture error correlation.
- `architecture_test_predictions.csv`: fold-averaged member probabilities.
- `spe_predictions.csv`: final probability, label and state/fold/architecture
  disagreement for each independent-test WSI.
- `manifest.json`: config hash, exact run directories, checkpoints and fitting
  diagnostics needed to reconstruct the locked ensemble.

The manuscript supplied with this project says that exact weighting details
belong in Supplementary Methods but does not contain that equation. The current
transparent implementation minimizes patient-equal OOF binary cross-entropy
plus a configurable residual-correlation penalty. If the final Supplementary
Methods specifies a different objective, update `ensemble/spe.py`, increment the
experiment name, and rerun development-only weight locking before any test use.
