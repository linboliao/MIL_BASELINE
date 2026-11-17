# Prostate utility extensions

This branch keeps main's model registry. SPE, PSIR, domain adaptation models,
historical cohorts and old result trees are not imported.

## Evaluation and prediction output

All existing shared evaluation paths support return_predictions=True,
prediction_path=<new CSV>, and slide_paths=<sequential evaluation paths>.
The default return remains (loss, metrics); requesting predictions adds a
DataFrame as the third result. Coverage includes generic MIL, ACMIL, CLAM,
DSMIL, DTFD (AFS / MaxS / MaxMinS), SCMIL and NN_MIL. NN_MIL keeps uncertainty
columns. Feature and attention extraction remain separate operations.

Common scalar columns: sample_id, slide_id, slide_path, label, prediction,
prob_0... and logit_0.... CSVs may provide <split>_sample_id; otherwise full
feature paths identify samples. Stems alone are not unique identifiers.
Automatic identity mapping requires a sequential loader; shuffled loaders
must return explicit IDs via return_metadata=True. That option changes pairs
to triples and is incompatible with NN_MIL's pair-only training collate.
The default remains pairs.

test_mil.py saves Infer_Result.csv for all evaluation paths and accepts
--device cpu / --device cuda:0. Training can enable:

    General:
      save_predictions: true

Each performed validation/test pass writes
predictions/{val,test}_epoch_NNNN.csv in the run directory. This does not add
evaluation on splits that the existing pipeline would leave untouched.
Prediction files and nonempty run directories are rejected, not overwritten.

cal_scores takes probabilities by default; raw logits require
input_kind="logits". Probabilities are never softmaxed twice. AUROC is NaN
when class support is absent. BCE class-index targets get the proper loss
shape. Multi-logit BCE retains main's softmax evaluation convention;
single-logit binary BCE uses sigmoid.

## Bounded feature preload

Example settings (not resource allocations):

    Dataset:
      preload:
        train: true
        val: true
        test: false
      preload_mem_gb:
        train: 16
        val: 4
        test: 0
      preload_workers: 1

Preload is off by default. Fallback budget: 1 GiB per enabled split; omitted
entries in an explicit budget mapping get zero. Workers are bounded to 1–4.
Legacy constructor mem_map is accepted.

Caching is lazy on first access and supports ordinary, coordinate, CDP, LONG
and SC datasets. Admitted tensor payload never exceeds the dataset budget.
This is NOT a process RSS limit: outstanding loads, a cache clone, Python
allocations and GPU transfer require headroom. Start with one preload worker.
Separate splits/models have separate budgets. Read errors fail explicitly.
cache_info() reports completeness, tensor bytes and hit/miss counts.
Partial caching still reloads uncached samples each epoch.

Entry points use a cache-aware DataLoader that forces num_workers=0 when
preload is enabled, avoiding replicated caches across fork/spawn workers.
Direct users should use utils.data_loading.DataLoader too. PT dictionaries
(feats/features), tensors and H5 are supported. Coordinate row counts are
validated. SC pseudo-grid fallback is warned about; it is not measured spatial
information.

## Loss, checkpoint selection and reproducibility

criterion may be ce or {loss: ce, label_smoothing: 0.0}. Smoothing is opt-in.
Main's complete model factory is retained.

    General:
      checkpoint_tie_break: val_loss
      checkpoint_tie_epsilon: 1.0e-9

Default tie-break policy is none. Optional val_loss uses lower validation loss
when the primary metric ties. This changes selection and must be recorded in
the protocol. The first finite validation result always saves a best checkpoint,
including metric zero. best_model_metric: val_loss uses actual validation loss.

Runs save config, Git status/diff hash, split SHA256, interpreter and library
metadata. Stronger determinism is opt-in:

    PYTHONHASHSEED=42 MIL_DETERMINISM=strict /absolute/python train_mil.py ...

MIL_DETERMINISM=1 warns on unsupported deterministic operations; strict raises.
PYTHONHASHSEED must be set before Python starts. Identical results across
different CUDA/GPU/library versions are not guaranteed. Each node retains its
configured interpreter.

## Validation

python -m unittest tests.test_prostate_utils -v uses CPU synthetic data.
Set TMPDIR to the node-specific project operational scratch directory;
synthetic artifacts are retained. Tests cover evaluation contracts, all DTFD
branches, real ABMIL/NN_MIL forwards, identity, budgets, coordinate compatibility,
non-clobbering outputs and process call-site wiring. This is not validation of
every architecture on real WSIs or a GPU performance benchmark.

## DTFD grouping and distillation

Training and evaluation share utils/dtfd_utils.py. A WSI is split into exactly
min(num_Group, patch_count) balanced, nonempty groups. AFS pools one feature per
group and does not use total_instance.

For MaxS/MaxMinS, total_instance is a positive integer distributed across the
actual groups (remainders go to the first groups). Budgets smaller than the
actual group count are rejected. MaxS selects the top quota per group.
MaxMinS selects both extremes: its maximum count is twice total_instance.
Selection is capped at the available patches and extremes never duplicate a
patch. A short group can therefore underspend its budget; it is not padded.
This fixes zero-quota / -0 slicing and overlapping selections. The example
DTFD config uses num_Group=5 and total_instance=5.

These boundary rules differ from legacy torch.chunk/floor/overlapping behavior.
Do not treat predictions for affected bags as bitwise reproduction of old runs.
Batch size must remain one WSI. Unsupported modes, empty bags and nonintegral
parameters fail explicitly.
