# Diagnosis external test

This package evaluates the reviewed 521-slide external cohort with the
`h-optimus-1` (1536-dimensional) representation, 10x magnification and
Reinhard stain normalization.

## Files

- Dataset: `datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv`
- Manifest: `datasets/Diagnosis/External/h-optimus-1/dataset_manifest.json`
- Config: `configs/Diagnosis/External/h-optimus-1_AB_MIL.yaml`

The dataset contains only `test_*` columns and must be used with `test_mil.py`,
not `train_mil.py`. The four reviewed slides from case `202302334` retain their
original labels and have specimen type corrected from CNB to RP in this copy.
The source `datasets/Diagnosis/external_test.csv` is not overwritten.

## Run

```powershell
python test_mil.py `
  --yaml_path configs/Diagnosis/External/h-optimus-1_AB_MIL.yaml `
  --test_dataset_csv datasets/Diagnosis/External/h-optimus-1/external_test_h-optimus-1.csv `
  --model_weight_path <h-optimus-1_AB_MIL_checkpoint.pth> `
  --test_log_dir result/Diagnosis/External/h-optimus-1/AB_MIL
```

The CSV currently references the NAS/Linux prefix
`/NAS145/liaolinbo/Data/MXB/外部测试/.../h-optimus-1`. On Windows, replace that
prefix with the actual mounted feature directory before running. Do not use a
virchow2 checkpoint with this CSV: virchow2 expects 2560-dimensional features,
while these paths identify h-optimus-1 features with dimension 1536.
