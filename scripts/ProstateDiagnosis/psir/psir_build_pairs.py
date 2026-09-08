"""PSIR step 1: build the case-level train-signal / held-out split for Panel A.

Panel A = 6 hospitals x ~60 cases, consecutive tissue sections of the same
case stained/scanned at each institution. We split by CASE (never by slide)
into 5 folds: each fold uses 80% of cases as the contrastive training-signal
source, the other 20% is held out and never touched during PSIR training --
this is what keeps the later "does PSIR reduce cross-center disagreement"
evaluation from being a circular self-certification.

Panel B is NOT split here -- it stays a fully independent, never-trained-on
validation set (handled directly by the eval script later).
"""
import json
import os

import pandas as pd
from sklearn.model_selection import GroupKFold

SER = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE/datasets/ProstateDiagnosis/serial_sections'
OUT_DIR = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE/datasets/ProstateDiagnosis/psir'
N_SPLITS = 5
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f'{SER}/panel_A_6center.csv', dtype={'case_id': str})
df = df[df['excluded'] == False].copy()  # noqa: E712 -- pandas bool column

# keep only cases that actually have >=2 usable centers (need pairs for contrastive loss)
centers_per_case = df.groupby('case_id')['center'].nunique()
usable_cases = centers_per_case[centers_per_case >= 2].index
dropped = set(centers_per_case.index) - set(usable_cases)
if dropped:
    print(f'dropping {len(dropped)} cases with <2 usable centers: {sorted(dropped)}')
df = df[df['case_id'].isin(usable_cases)].reset_index(drop=True)

print(f'usable Panel A rows: {len(df)}, cases: {df["case_id"].nunique()}, '
      f'centers/case: min={centers_per_case[usable_cases].min()} '
      f'max={centers_per_case[usable_cases].max()}')

case_label = df.groupby('case_id')['label'].first()
cases = sorted(df['case_id'].unique())
gkf = GroupKFold(n_splits=N_SPLITS)
# GroupKFold needs one row per group to split; use the case-level frame
case_df = pd.DataFrame({'case_id': cases})
case_df['label'] = case_df['case_id'].map(case_label)

folds = list(gkf.split(case_df, groups=case_df['case_id']))
# GroupKFold on a case-per-row frame with groups=case_id just partitions
# cases into N_SPLITS folds (deterministic order, not shuffled) -- fine here,
# we don't need stratification for a ~50-60 case contrastive-signal split.

assignment = {}
for k, (train_idx, held_idx) in enumerate(folds, start=1):
    held_cases = set(case_df.loc[held_idx, 'case_id'])
    for c in cases:
        assignment.setdefault(c, {})[f'fold{k}'] = 'held_out' if c in held_cases else 'train_signal'

out_rows = []
for c in cases:
    row = {'case_id': c, 'label': case_label[c], 'n_centers': int(centers_per_case[c])}
    row.update(assignment[c])
    out_rows.append(row)
out_df = pd.DataFrame(out_rows)
out_df.to_csv(f'{OUT_DIR}/panel_a_case_folds.csv', index=False, encoding='utf-8-sig')

for k in range(1, N_SPLITS + 1):
    n_held = (out_df[f'fold{k}'] == 'held_out').sum()
    n_train = (out_df[f'fold{k}'] == 'train_signal').sum()
    print(f'fold{k}: train_signal={n_train} cases, held_out={n_held} cases')

df.to_csv(f'{OUT_DIR}/panel_a_usable_slides.csv', index=False, encoding='utf-8-sig')
meta = {'n_splits': N_SPLITS, 'n_cases': len(cases), 'n_slides': len(df)}
with open(f'{OUT_DIR}/panel_a_split_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(f'\nwritten -> {OUT_DIR}/panel_a_case_folds.csv, panel_a_usable_slides.csv')
