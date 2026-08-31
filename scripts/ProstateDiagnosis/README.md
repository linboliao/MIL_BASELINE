# ProstateDiagnosis workflows

Scripts here support the prostate-diagnosis AB_MIL experiments: the
center-adversarial (CENTERADV_AB_MIL) training added to fight cross-center
generalization drift on the 301 external cohort, and the Virchow2 feature
comparison against the UNI2 baseline. Run commands from the repository root
after activating the Python environment.

- `dataset_build/` — builds the per-fold dataset CSVs these experiments train
  from: `build_centeradv_fold_csvs.py` adds a `_domain` (center id) column to
  the existing 3-center 5-fold splits for CENTERADV_AB_MIL;
  `build_virchow2_fold_csvs.py` / `build_virchow2_fold_csvs_local.py` retarget
  those same splits at Virchow2 features on NAS vs. the local `/data14`
  cache; `patch_virchow2_preload.py` one-time-patches the generated Virchow2
  fold configs to add `preload`/`preload_mem_gb` settings.
- `config_gen/` — writes the per-fold training YAMLs
  (`gen_configs_centeradv.py`, `gen_configs_centeradv_lambda015.py` for the
  two GRL-lambda variants, `gen_configs_virchow2.py` for the Virchow2 AB_MIL
  baseline).
- `eval/` — standalone external/internal test-set evaluation launchers for
  the various trained checkpoint versions (see each script's header comment
  for which checkpoint/test-set pair it targets).
- `calibration/` — post-hoc calibration experiments run against existing
  `Infer_Result.csv` outputs (Platt scaling, temperature scaling) while
  diagnosing the 301 fold-to-fold instability; both were negative results,
  kept for reference.

See the project's dataset build script at
`split_scripts/build_prostate_diagnosis_dataset.py` for how the underlying
dev/internal_test/external_test splits themselves are constructed.
