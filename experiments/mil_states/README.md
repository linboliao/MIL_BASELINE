# MIL weight states

For standard AB-MIL, set `General.save_epochs: true` to keep every completed
epoch's weights, including the stopping epoch. Default is off. These are
model weights and do not contain optimizer or RNG state.

Build Best, Last, SWA-last5 and epoch EMA (0.9) from a completed standard
Train/Val run using `python -m experiments.mil_states.build_states --run
<fold-output> --output <new-state-directory>`. SWA uses up to the last five
actual epochs. The native R2 trainer generates these states itself.

The R2 recipe in run.sh calls the repository-local train_cell.py, adapted from
the NAS train_pfm_v2.py. The trainer still imports the historical framework
checkout for the selected host and records its dependency hashes. Staging, cell QA and evaluation use the local modules described below. The mil_states NAS
folder is the review copy with relocated inputs. The recipe requires a new
prepared run root with RUN_META.json, and covers one mSTAR/SL/42 example.
Native v2 accepts seeds 42 and 43. It does not implement the entire 120-cell
extension with seed 44.

R2 uses its own stopping and checkpoint-selection rules and resume bundles.
Using standard AB-MIL with save_epochs produces a weight trajectory under the
standard trainer's rules. Source-only selection and held-out evaluation remain
in the local selection.py and evaluate_cell.py modules. Evaluation persists
the source-only decision before reading held-out inputs.


Execution modules:
- stage_features.py preserves the native R2 full-PFM cache verification.
- training_inputs.py holds fold-local resident features and sampler.
- train_cell.py trains one cell and writes the original R2 state format.
- audit_cell.py checks cell artifacts.
- evaluate_cell.py uses selection.py and locks the source-only choice before held-out reads.

Use `--prep` to point these commands at a frozen guard-preparation package.
The NAS mil_states review copy is not a guard-preparation package. Historical
preparation scripts remain with provenance. No run directory was allocated
by the migration.
