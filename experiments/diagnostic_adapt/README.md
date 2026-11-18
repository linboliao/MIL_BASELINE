# Diagnostic adaptation, stage 2

Run `bash experiments/diagnostic_adapt/run.sh` on 195 to check the
frozen inputs without starting training. To execute stage 2 in an allocated
session, pass a new dated run ID and a physical GPU index.

The local `run_stage2.py` takes explicit config, dataset and PrePATH roots.
The shell recipe points to the NAS diagnostic_adapt input package. The runner
resolves the three repeat0 input sets from that package and verifies their
frozen hashes. The copied policy is checked against the frozen lineage.
The runner records hashes for the local implementation and AB-MIL module.
The original release remains the reference for historical results.

Execution locks the selected action before reading the unlabelled evaluation
roster. Stage 2 ends with predictions. Evaluation truth and stage 3 metrics
are handled separately.

`run_stage2.py` owns the command line and serial dispatch. `run_fold.py`
owns per-fold sequencing. `inputs.py` checks frozen inputs and manages
feature access. `scoring.py` handles WSI scores and patient aggregation.
`policy.py` holds the unchanged frozen decision rules.
