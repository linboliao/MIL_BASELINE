"""Check which serial-panel cases actually appear in the 5-fold training splits.

The panel is only usable as an unbiased robustness probe if the model has not
seen that tissue. The 3-centre design draws dev/internal_test from the dev,
oldtest AND ext_sl pools, so filtering on `pool == 'dev'` alone is not enough --
verify directly against the fold CSVs, and also against patient_id (a panel case
may share a patient with a different slide that IS in training).
"""
import os

import pandas as pd

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'
SER = f'{ROOT}/datasets/ProstateDiagnosis/serial_sections'
FOLDS = f'{ROOT}/datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center'

case_labels = pd.read_csv(f'{SER}/serial_case_labels.csv')
panel_cases = set(case_labels['case_id'].astype(str))
panel_patients = set(case_labels['main_patient_id'].dropna().astype(str))

print(f'panel cases: {len(panel_cases)} | panel patients known in main cohort: {len(panel_patients)}')
print()
print('panel cases by main-cohort pool:')
print(case_labels['main_pool'].value_counts(dropna=False).to_string())
print()

for fold in range(1, 6):
    path = f'{FOLDS}/fold_{fold}/prostate_dev_uni2_{fold}fold.csv'
    if not os.path.exists(path):
        print(f'fold{fold}: MISSING {path}')
        continue
    df = pd.read_csv(path)
    hits = {}
    for group in ['train', 'val', 'test']:
        stems = df[f'{group}_slide_path'].dropna().map(
            lambda p: os.path.splitext(os.path.basename(p))[0])
        hits[group] = len(set(stems) & panel_cases)
    print(f'fold{fold}: panel slides present -> train={hits["train"]}  '
          f'val={hits["val"]}  test={hits["test"]}')

print()
print('=== patient-level leakage check (fold 1) ===')
man = pd.read_csv(f'{ROOT}/datasets/ProstateDiagnosis/manifest.csv')
man['slide_id'] = man['slide_id'].astype(str)
slide2pat = dict(zip(man['slide_id'], man['patient_id'].astype(str)))
df = pd.read_csv(f'{FOLDS}/fold_1/prostate_dev_uni2_1fold.csv')
train_stems = df['train_slide_path'].dropna().map(lambda p: os.path.splitext(os.path.basename(p))[0])
train_patients = set(slide2pat.get(s) for s in train_stems) - {None}
shared = panel_patients & train_patients
print(f'panel patients also appearing in fold1 TRAIN (any slide): {len(shared)}')
print(sorted(shared)[:20])
print()

# the exclusion list that is actually needed: every slide of every panel patient
need_exclude_slides = man[man['patient_id'].astype(str).isin(panel_patients)]['slide_id'].tolist()
print(f'=> slides that must be removed from training to keep the panel clean: '
      f'{len(need_exclude_slides)}')
pd.DataFrame({'slide_id': sorted(need_exclude_slides)}).to_csv(
    f'{SER}/exclude_from_training_slides.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'patient_id': sorted(panel_patients)}).to_csv(
    f'{SER}/exclude_from_training_patients.csv', index=False, encoding='utf-8-sig')
print(f'written: exclude_from_training_slides.csv / exclude_from_training_patients.csv')
