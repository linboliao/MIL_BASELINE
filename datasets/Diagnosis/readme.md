# MIL patient-disjoint independent test set

This directory is the patient-level de-overlapped version of the independent
test dataset used by the Diagnosis MIL experiments.

## Construction

- The original `datasets/Diagnosis/test_wo_type.csv` is retained unchanged
  (345 slides).
- Patient identifiers are normalized from slide filenames and compared with
  the union of the development train/validation patients.
- All test slides belonging to any overlapping patient are removed. This
  removes 47 slides from 17 patients: 35 exact-WSI overlaps and 12 additional
  slides that share an overlapping patient identifier.
- The final independent test cohort contains 298 slides and has zero patient
  overlap with the development cohort.

## Final cohort

- Labels: 30 negative (`0`), 268 positive (`1`).
- Specimen types: CNB 27, RP 215, TURP 56.
- Feature path prefix:
  `/NAS145/liaolinbo/Data/MXB/CLS测试/feat_0_448/stains/Reinhard/pt_files/h-optimus-1/`

All five fold files retain their original development train/validation split
and contain the same locked 298-slide independent test set. Use this directory,
rather than `datasets/Diagnosis/MIL`, for patient-independent final evaluation.

## Files

- `Total_5-fold_MIL_PatientDisjoint_{1..5}fold.csv`: five experiment datasets.
- `removed_test_overlap_audit.csv`: every removed test slide and its matching
  development record.
- `dataset_manifest.json`: cohort counts, independence check, and per-file
  SHA-256 hashes.
- `../test_patient_disjoint.csv`: portable 298-slide test source without the
  NAS feature prefix.
