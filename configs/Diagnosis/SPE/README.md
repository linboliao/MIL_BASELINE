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
checkpoints are used. Therefore every configured SPE candidate architecture
must be trained with:

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

Run on Linux after all five folds of all configured architectures have completed:

```bash
python -u scripts/Diagnosis/spe/run.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml --preflight
bash scripts/Diagnosis/spe/run.sh
```

For architecture-level multi-GPU inference, assign one worker process to each
visible GPU either from the command line:

```bash
python -u scripts/Diagnosis/spe/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

The shell wrapper also forwards arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/Diagnosis/spe/run.sh \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

When checkpoint and fold predictions already exist under the same checkpoint
policy, selection and weighting can be rebuilt without model inference:

```bash
python -u scripts/Diagnosis/spe/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml \
  --refit-from result/Diagnosis/SPE/v3_bacc
```

Do not refit v3 from v1/v2: those directories contain probabilities from the
old checkpoint policy and do not contain TDA_MIL. Cached refitting still uses
only OOF labels; independent-test labels remain excluded.

or set `experiment.devices: [cuda:0, cuda:1, cuda:2, cuda:3]` in YAML. Each
GPU evaluates its MIL shard sequentially, while shards run concurrently. Do not
repeat the same GPU ID. `num_workers` applies to every GPU process, so increase
it conservatively to avoid excessive CPU workers and disk contention. Existing
checkpoint prediction CSVs remain reusable unless `--overwrite` is supplied.

For the final paper run, replace every `run_dir: null` with the exact immutable
training-run directory. Automatic latest-run discovery is convenient during
development but is not appropriate for a locked analysis.

The current BAcc-oriented configuration preserves v1/v2 and writes outputs to
`result/Diagnosis/SPE/v3_bacc/`. For every architecture/fold it retains epochs
whose validation BAcc lies within 0.005 of that fold's best value, then spreads
at most five checkpoints across that high-performance band. Architecture
weights use class/patient-balanced loss, and members are forward-selected by
five-fold held-out BAcc.

- `oof_architecture_predictions.csv`: development OOF architecture matrix.
- `architecture_weights.csv`: development-selected status, fitted non-negative
  weights, and diagnostics for every candidate.
- `residual_similarity.csv`: patient-weighted architecture error correlation.
- `architecture_test_predictions.csv`: fold-averaged member probabilities.
- `spe_predictions.csv`: final probability, label and state/fold/architecture
  disagreement for each independent-test WSI.
- `manifest.json`: config hash, exact run directories, checkpoints and fitting
  diagnostics needed to reconstruct the locked ensemble, including every
  fold-held-out forward-selection step.

The manuscript supplied with this project says that exact weighting details
belong in Supplementary Methods but does not contain that equation. The current
transparent BAcc-oriented implementation uses class/patient-balanced OOF
binary cross-entropy plus a configurable residual-correlation penalty, while
membership is selected by held-out-fold BAcc. If the final Supplementary
Methods specifies a different objective, update `ensemble/spe.py`, increment
the experiment name, and rerun development-only locking before any test use.
