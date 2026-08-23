# Diagnosis workflows

Diagnosis scripts are grouped by experiment. Run commands from the repository
root after activating the Python environment.

`scripts/Diagnosis/spe/` holds the production Stability-Prioritized Ensemble
(SPE) system. `scripts/Diagnosis/specimen_type_investigation/` is a separate,
unrelated investigation that reuses SPE's already-computed development-OOF
predictions as raw material for a paper-narrative audit of cross-cohort
model-ranking instability; it shares no code path with the SPE system beyond
that shared input data. Do not confuse the two just because both used to live
under one `spe/` folder.

## Stability-Prioritized Ensemble (production system)

All entrypoints live under `scripts/Diagnosis/spe/infra/`:

```bash
python -u scripts/Diagnosis/spe/infra/run.py --list-variants
python -u scripts/Diagnosis/spe/infra/run.py --variant hierarchical_bacc --preflight
python -u scripts/Diagnosis/spe/infra/run.py --variant hierarchical_bacc \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Custom configurations remain supported:

```bash
python -u scripts/Diagnosis/spe/infra/run.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml
```

The complete sequential workflow trains all configured MIL models, evaluates
their best checkpoints on the locked internal and external cohorts, builds SPE
variants, and writes performance summaries:

```bash
bash scripts/Diagnosis/spe/infra/run_all.sh
```

`run_all.sh` centralizes shared paths, model lists, GPU assignments, and repeated
internal/external commands in small functions. Adjust its constants and job
arrays when moving the workflow to another server.

Post-processing and audit commands:

```bash
python -u scripts/Diagnosis/spe/infra/check_cohort_overlap.py --help
python -u scripts/Diagnosis/spe/infra/test_best_checkpoints.py --help
python -u scripts/Diagnosis/spe/infra/summarize_performance.py --help
python -u scripts/Diagnosis/spe/infra/summarize_member_performance.py --help
python -u scripts/Diagnosis/spe/infra/analyze_tumor_content.py --help
python -u scripts/Diagnosis/spe/infra/audit_checkpoint_basins.py --help
python -u scripts/Diagnosis/spe/infra/build_cp_awa.py --help
```

Model-test results are stored below
`result/Diagnosis/ModelTest/{internal,external}/`. SPE results are grouped under
`result/Diagnosis/SPE/MIL_Mix/trajectory_robust_anchor/<experiment>/`. The
overlap audit stops locked external evaluation when it detects internal
patients; `--allow-overlap` is intended only for diagnostics.

`result/Diagnosis/SPE/_archive/MIL_legacy_dataset/` holds SPE output from the
earlier, now-superseded "MIL" dataset export (pre-`MIL_Mix`). Kept for
reference only; nothing in the current pipeline reads it.

See `configs/Diagnosis/SPE/README.md` for variant definitions and methodology.

## Specimen-type composition investigation (paper narrative audit)

Cross-cohort model-ranking instability audit, organized to mirror the paper's
three-act narrative (see `paper/paper_narrative_progress.md`). Every script
here only reads already-computed predictions from
`result/Diagnosis/SpecimenType_Investigation/_shared_dev_oof/` (the SPE
system's development-OOF `oof_predictions.csv` per architecture) and
`result/Diagnosis/ModelTest/MIL_Mix/` — no new inference.

```
scripts/Diagnosis/specimen_type_investigation/
├── act1_cohort_composition/     # Act 1: descriptive cohort/specimen-type stats
├── act2_ranking_instability/    # Act 2: ranking-instability diagnosis + permutation test
├── act3_selection_criterion/    # Act 3: production type-balanced selection + validation
└── background_excluded/         # explored, not part of the main narrative (see paper doc §"背景/已排除路径")
```

```bash
bash scripts/Diagnosis/specimen_type_investigation/act1_cohort_composition/run_cohort_composition_report.sh
bash scripts/Diagnosis/specimen_type_investigation/act2_ranking_instability/run_specimen_type_selection.sh
bash scripts/Diagnosis/specimen_type_investigation/act2_ranking_instability/run_specimen_type_permutation_test.sh
bash scripts/Diagnosis/specimen_type_investigation/act3_selection_criterion/run_development_oof_report.sh
```

Each act's script writes its own JSON/CSV outputs under a matching
`result/Diagnosis/SpecimenType_Investigation/actN_.../analysis/` directory.
`background_excluded/` scripts (ensemble search, Layer-1/Layer-2 stacking,
cross-PFM feasibility, fold-consistency and class-balance selection
criteria) write to `result/Diagnosis/SpecimenType_Investigation/background_excluded/analysis/`.

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

`scripts/Diagnosis/wsi_representation/plots/config_plots.ipynb` (formerly the
top-level `config.ipynb` — the name never matched its contents, three
plotting cells for PFM/magnification/stain comparisons) lives here too.

Generated checkpoints, predictions, summaries, manifests, and figures belong
under `result/Diagnosis/`; source code stays under `scripts/Diagnosis/`.
