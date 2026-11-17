# NAS-backed PFM LOCO orchestration

The `orchestration` package separates experiment identity (`Study`, `Cell`), frozen
input adaptation (`FrozenInputs`), host-local execution (`LocalExecutor`), and
artifact validation (`CellAudit`). `python -m orchestration` is the composition
root. None of these components imports model implementations; training/testing
run in isolated subprocesses through the existing repository entry points.

The checked-in profile references the existing NAS PFM_LOCO configuration and
split directories. They remain unchanged. The five domains are SL, XC, CNB,
TURP, RP, not five CV folds. The default matrix is eight PFMs by five domains.
Source configuration retains architecture, optimizer, epochs, seed and early
stopping. Runtime adaptation explicitly sets selection macro-F1 with validation
loss tie-break (epsilon 1e-9), retains checkpoints, and maps selection to val.
Calibration is snapshotted but unused. Held-out test paths are absent from the
training CSV and evaluated once after training with the selected checkpoint.
This replays the design on the current implementation; it is not a claim of
bitwise reproduction of commit 060f989.

## Read-only planning

From the repository on 138:

```bash
/data12/jing/anaconda3/envs/PrePATH/bin/python -B -m orchestration plan
/data12/jing/anaconda3/envs/PrePATH/bin/python -B -m orchestration plan --pfms conch --domains SL --check-features
```

Planning checks source configuration, labels, duplicate paths, patient-role
separation, role/PFM identity and hashes. Optional feature checks verify existence
and recorded byte size, not tensor contents or original feature hashes.

## Explicit execution

Only after inspecting live jobs, GPU memory, RAM, storage and source access.
Run long executions inside a dedicated tmux/session with a unique preserved log,
not foreground SSH. The command below is the session command:

```bash
/data12/jing/anaconda3/envs/PrePATH/bin/python -B -m orchestration run \
  --pfms conch --domains SL --gpu 1 --run-id seed42-conch-sl-repro-r1
```

On 195 use `/home/lbliao/anaconda3/envs/clam/bin/python` with its lib directory in
LD_LIBRARY_PATH. The executor verifies the host/interpreter and a committed clean
checkout. It executes sequentially on one explicitly selected physical GPU.
Separate nodes require non-overlapping cells AND unique run IDs. It does not
select resources, submit SSH jobs, terminate processes, retry, resume, or delete
caches. Re-execution at an existing attempt path is refused. Failure stops the
matrix, preserves partial artifacts and writes failed.json. Unstarted cells have
no completed marker; completed.json at the run root is written only after all
selected cells pass. run_meta.json records creation state; completion/failure
markers, rather than that initial state, establish terminal status.

Use node-specific TMPDIR/TMP/TEMP as required by project instructions. Attempts
go to provenance/generalization/pfm-loco-prostate-reproduction/<run-id> by the
profile. No outputs are written to canonical studies or the source experiment.
Each cell includes source snapshots, input hashes, generated train/val/test CSVs,
runtime YAML, logs, checkpoints, predictions and completion evidence. Default
preload is off, reading original NAS features; `--preload-gb` sets a tensor payload
budget separately for EACH train and selection split, not an overall RSS cap.
No local feature staging or conversion is performed.

```text
<run-id>/
  run_meta.json
  <pfm>/<domain>/
    inputs_meta.json
    inputs/                     # immutable source snapshots and adapted CSVs
    model.yaml                  # runtime configuration
    training/                   # native framework outputs
    evaluation/                 # single held-out pass
    train.log, test.log
    completed.json OR failed.json
  completed.json                # all selected cells passed
```

## Independent audit

```bash
/data12/jing/anaconda3/envs/PrePATH/bin/python -B -m orchestration audit --cell /absolute/attempt/pfm/domain
```

Audit checks one selected-epoch log, checkpoint existence, exact held-out path and
label coverage, finite normalized binary probabilities and argmax consistency
(class 0 wins exact ties). It does not certify scientific equivalence, regenerate
metrics, or approve canonical publication. Existing frozen NAS inputs and original
results remain the authority for historical reproduction.
