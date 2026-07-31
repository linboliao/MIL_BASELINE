# Diagnosis WSI representation experiments

All experiments use AB-MIL, the same patient-disjoint five-fold partitions,
seed 42, and validation macro-F1 for checkpoint selection and early stopping.

```text
PFM/       7 encoders, 20x-equivalent, no stain normalization
Mag/       H-optimus-1, 20x/10x/5x-equivalent, no stain normalization
Stains/    H-optimus-1, 10x-equivalent, three normalization methods
```

Every YAML reads its folds from `datasets/Diagnosis/` and writes training
artifacts below `result/Diagnosis/Mag/`. `train_mil.py` then appends
`<DATASET_NAME>/AB_MIL/<seed_time>/fold_<n>` to that log root.

The no-normalization 10x baseline for the stain comparison is the `Mag/10x`
experiment; it must not be retrained under a different fold assignment.

The `Mag/20x` baseline is identical to `PFM/h-optimus-1`. Both YAML files use
`DATASET_NAME: h-optimus-1` and the same checkpoint directory, so the training
script runs 12 unique jobs. The OOF workflow evaluates all 13 YAML configs and
reports `Mag/20x` separately.

Feature dimensions:

| Encoder | `in_dim` |
| --- | ---: |
| CONCH | 512 |
| H-optimus-1 | 1536 |
| mSTAR | 1024 |
| OmiCLIP | 768 |
| UNI | 1024 |
| UNI2 | 1536 |
| Virchow2 | 2560 |

Before launching all jobs, inspect one `.pt` file from every PFM directory and
confirm that its last dimension matches the configured `in_dim`.

## Checkpoint policy

Checkpoint behavior is controlled from `General.checkpoint`:

```yaml
General:
    checkpoint:
        save_mode: best_last  # best_last | every_epoch
        spe:
            metric: macro_f1
            stability_threshold: 0.003
            min_consecutive: 5
            max_checkpoints: 5
```

- Use `best_last` for PFM, magnification, and stain-normalization selection.
  These experiments select a representation and do not contribute model-state
  members to the final SPE.
- Use `every_epoch` for every architecture-fold trajectory that will
  participate in Stable-Best-Mean, Stable-State-Mean, hierarchical SPE, or
  ensemble-reconstruction analyses.
- Configs without a `General.checkpoint` block default to `best_last`.

`every_epoch` keeps the existing `Best_EPOCH_*.pth` and `Last_EPOCH_*.pth`
files for backward compatibility and additionally creates:

```text
fold_<n>/
  epoch_checkpoints/Epoch_0001.pth
  epoch_checkpoints/Epoch_0002.pth
  ...
  checkpoint_manifest.json
  spe_checkpoint_selection.json
```

The manifest stores each epoch's validation metrics and checkpoint path. The
selection file records the final qualifying stable interval, whether the
fallback rule was used, and the up to five approximately evenly distributed
checkpoints selected for SPE.

For a one-off override, a config containing this block can be launched with:

```bash
python train_mil.py \
  --yaml_path <config.yaml> \
  --options General.checkpoint.save_mode=every_epoch
```
