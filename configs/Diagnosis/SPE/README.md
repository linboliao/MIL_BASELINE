# Hierarchical SPE

## Variants

All variants use the readable `run.py --variant NAME` entrypoint. Shared
inference, cache, OOF merge, and output logic remains isolated in `_engine.py`.

| Variant | Configuration |
|---|---|
| `hierarchical_bacc` | `hierarchical_spe.yaml` |
| `macro_f1` | `hierarchical_spe_macro_f1.yaml` |
| `hierarchical_v1` | `hierarchical_spe_v1.yaml` |
| `best_state_anchor` | `best_state_anchor.yaml` |
| `best_state_anchor_v1` | `best_state_anchor_v1.yaml` |
| `constrained_linear_stacking` | `constrained_linear_stacking.yaml` |
| `ra_spe` | `ra_spe.yaml` |
| `diversity_veto` | `diversity_veto.yaml` |
| `sensitivity_constrained` | `sensitivity_constrained.yaml` |

## Best-state + Top1-anchor + automatic fallback

`best_state_anchor.yaml` is a risk-controlled alternative to the manuscript
hierarchy. Its story is deliberately deployment-oriented: first align every
member with the direct baseline by using exactly one `Best_EPOCH` per fold;
then identify the strongest development-OOF architecture as an immutable
anchor; finally allow at most two complementary members to use the remaining
probability budget. The anchor always retains at least 70% weight.

The proposed blend is deployed only when both locked development criteria pass:

- pooled fold-held-out OOF BAcc improves by at least 0.005;
- at least four of five OOF folds are non-decreasing versus Top1.

Otherwise the pipeline writes Top1 probabilities as the final prediction and
records the candidate ensemble, fold diagnostics, and fallback reasons in the
CSV/manifest. Internal and external runs use the same development-locked
decision; external labels never enter member selection, weighting, or fallback.
TDA uses deterministic evenly spaced patch subsampling during evaluation, and
GDF replaces Gumbel sampling with the corresponding softmax in evaluation, so
the Best-state baseline and this ensemble see reproducible member predictions.

```bash
python -u scripts/Diagnosis/spe/run.py --variant best_state_anchor --preflight
CUDA_VISIBLE_DEVICES=5,6,7 python -u scripts/Diagnosis/spe/run.py \
  --variant best_state_anchor \
  --devices cuda:0,cuda:1,cuda:2
python -u scripts/Diagnosis/spe/summarize_performance.py \
  --predictions result/Diagnosis/SPE/Internal/best_state_anchor/spe_predictions.csv \
  --skip-center
```

For the external cohort, run inference with the same YAML and locked OOF rule:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 python -u scripts/Diagnosis/spe/run.py \
  --variant best_state_anchor \
  --test-dataset-csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv \
  --output-name best_state_anchor_external --devices cuda:0,cuda:1,cuda:2
```

The legacy 11-member patient-disjoint candidate pool has a separate compatible
configuration, `best_state_anchor_v1.yaml`. It reuses the same implementation
but keeps the original 11 members and latest-complete-run discovery. The former
`MIL_PatientDisjoint` cohort is now `datasets/Diagnosis/MIL` in this workspace.
Its output is written to
`result/Diagnosis/SPE/Internal/best_state_anchor_v1/`.

## Macro-F1 hierarchical SPE

`hierarchical_spe_macro_f1.yaml` keeps the legacy 11-architecture hierarchy
but makes macro F1 the explicit development endpoint. Stable states come from
`val_macro_f1`; architecture membership is forward-selected by pooled
five-fold held-out macro F1 at threshold 0.5. The inner continuous weight fit
uses class/patient-balanced BCE because thresholded macro F1 is not
differentiable. Independent-test labels are never used for either step.

```bash
python -u scripts/Diagnosis/spe/run.py --variant macro_f1 --preflight
python -u scripts/Diagnosis/spe/run.py --variant macro_f1 \
  --devices cuda:0,cuda:1,cuda:2
```

The output manifest records `development_oof_macro_f1`, and
`architecture_weights.csv` includes each candidate's `oof_macro_f1`.

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
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc --preflight
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

For architecture-level multi-GPU inference, assign one worker process to each
visible GPU either from the command line:

```bash
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

When checkpoint and fold predictions already exist under the same checkpoint
policy, selection and weighting can be rebuilt without model inference:

```bash
python -u scripts/Diagnosis/spe/run.py --variant hierarchical_bacc \
  --refit-from result/Diagnosis/SPE/Internal/bacc
```

Do not refit the BAcc run from legacy output directories: those directories
contain probabilities from the old checkpoint policy and may omit TDA_MIL. Cached refitting still uses
only OOF labels; independent-test labels remain excluded.

## Learnable aggregators

The constrained linear stacker retains every architecture and learns one
global non-negative weight vector. It adds shrinkage toward uniform weights, a
per-architecture cap, and a minimum effective-member constraint:

```bash
python -u scripts/Diagnosis/spe/run.py --variant constrained_linear_stacking \
  --refit-from result/Diagnosis/SPE/Internal/bacc
```

RA-SPE fits a small sample-specific gating network and reports meta-level
cross-fitted OOF predictions. Checkpoint-state variance is used as a
reliability feature. When an older merged OOF CSV omitted it, the runner first
recovers it from each architecture's cached `oof_predictions.csv`; only a cache
without those member files falls back to probability, entropy, and
inter-architecture disagreement features:

```bash
python -u scripts/Diagnosis/spe/run.py --variant ra_spe \
  --refit-from result/Diagnosis/SPE/Internal/bacc
```

For the final paper experiment, run RA-SPE without `--refit-from` (or regenerate
the merged OOF cache) so `use_state_variance: true` is effective. The locked
network is saved as `ra_spe_aggregator.pt`; sample-specific weights are written
to both OOF and independent prediction CSVs. Neither method learns a decision
threshold: classification remains fixed at 0.5.

To lock sensitivity to the BAcc development reference and optimize specificity
and BAcc over equal-weight architecture subsets, use:

```bash
python -u scripts/Diagnosis/spe/run.py --variant sensitivity_constrained \
  --refit-from result/Diagnosis/SPE/Internal/bacc
```

The effective threshold is selected from positive-class OOF order statistics
and recorded in `manifest.json`. Pass that threshold to the performance
summarizer; the command is:

```bash
python scripts/Diagnosis/spe/summarize_performance.py \
  --predictions result/Diagnosis/SPE/Internal/sensitivity_constrained/spe_predictions.csv \
  --threshold 0.455923717620198 \
  --skip-center
```

or set `experiment.devices: [cuda:0, cuda:1, cuda:2, cuda:3]` in YAML. Each
GPU evaluates its MIL shard sequentially, while shards run concurrently. Do not
repeat the same GPU ID. `num_workers` applies to every GPU process, so increase
it conservatively to avoid excessive CPU workers and disk contention. Existing
checkpoint prediction CSVs remain reusable unless `--overwrite` is supplied.

For the final paper run, replace every `run_dir: null` with the exact immutable
training-run directory. Automatic latest-run discovery is convenient during
development but is not appropriate for a locked analysis.

SPE outputs are grouped by locked cohort. Internal runs are written below
`result/Diagnosis/SPE/Internal/`, while a standalone test CSV defaults to
`result/Diagnosis/SPE/External/`. The BAcc-oriented configuration writes to
`result/Diagnosis/SPE/Internal/bacc/`. For every architecture/fold it retains epochs
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
