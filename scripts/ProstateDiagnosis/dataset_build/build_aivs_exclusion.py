"""Build the exclusion list for slides leaked into the 110-case human-vs-AI set.

Matches were established by exact level-0 dimensions + mpp, then confirmed by
thumbnail correlation; only pairs with corr >= 0.90 are kept. Exclusion is done
at PATIENT level, because a leaked slide's patient usually contributes other
slides to the development pool as well.
"""
import json
import os

import pandas as pd

OUT = '/tmp/aivs_metadata_match'
SER = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis'
DEST = os.path.join(SER, 'aivs_leakage')
MANIFEST = os.path.join(SER, 'manifest.csv')
FOLDS = os.path.join(SER, 'DataAnalysis/AB_MIL_uni2_5fold_3center')
CORR_MIN = 0.90

os.makedirs(DEST, exist_ok=True)

ver = pd.read_csv(f'{OUT}/thumbnail_verification.csv',
                  dtype={'renamed_id': str, 'candidate': str})
best = (ver[ver['corr'] >= CORR_MIN]
        .sort_values('corr', ascending=False)
        .drop_duplicates('renamed_id'))
print(f'confirmed matches (corr >= {CORR_MIN}): {len(best)}')

man = pd.read_csv(MANIFEST, dtype={'slide_id': str, 'patient_id': str})
lut = man.drop_duplicates('slide_id').set_index('slide_id')

best = best.rename(columns={'candidate': 'slide_id'})
for c in ['patient_id', 'label', 'type', 'center', 'pool', 'raw_path']:
    best[c] = best['slide_id'].map(lut[c])
best = best.sort_values('renamed_id')
best.to_csv(os.path.join(DEST, 'confirmed_matches.csv'), index=False, encoding='utf-8-sig')

leaked_slides = set(best['slide_id'])
leaked_patients = set(best['patient_id'].dropna())
print(f'leaked slides: {len(leaked_slides)} | distinct patients: {len(leaked_patients)}')
print()
print('slides per leaked patient in the whole cohort:')
per_pat = man[man['patient_id'].isin(leaked_patients)].groupby(
    ['patient_id', 'center', 'pool']).size().rename('slides_in_cohort')
print(per_pat.to_string())
print()

all_slides_of_patients = sorted(man[man['patient_id'].isin(leaked_patients)]['slide_id'])
pd.DataFrame({'slide_id': sorted(leaked_slides)}).to_csv(
    os.path.join(DEST, 'exclude_slides_direct.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'slide_id': all_slides_of_patients}).to_csv(
    os.path.join(DEST, 'exclude_slides_patientlevel.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'patient_id': sorted(leaked_patients)}).to_csv(
    os.path.join(DEST, 'exclude_patients.csv'), index=False, encoding='utf-8-sig')

print(f'direct leaked slides       : {len(leaked_slides)}')
print(f'patient-level slide removal: {len(all_slides_of_patients)}')
print()

# --- how much of the current folds does this touch? ---
print('=== presence in the current 5-fold splits ===')
impact = []
for fold in range(1, 6):
    p = f'{FOLDS}/fold_{fold}/prostate_dev_uni2_{fold}fold.csv'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    row = {'fold': fold}
    for group in ['train', 'val', 'test']:
        stems = df[f'{group}_slide_path'].dropna().map(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        row[f'{group}_direct'] = len(set(stems) & leaked_slides)
        row[f'{group}_patientlevel'] = len(set(stems) & set(all_slides_of_patients))
    impact.append(row)
imp = pd.DataFrame(impact)
print(imp.to_string(index=False))
imp.to_csv(os.path.join(DEST, 'fold_impact.csv'), index=False, encoding='utf-8-sig')

summary = {
    'confirmed_matches': int(len(best)),
    'correlation_threshold': CORR_MIN,
    'leaked_slides': len(leaked_slides),
    'leaked_patients': len(leaked_patients),
    'slides_removed_at_patient_level': len(all_slides_of_patients),
    'by_pool': best['pool'].value_counts().to_dict(),
    'by_center': best['center'].value_counts().to_dict(),
}
with open(os.path.join(DEST, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print()
print(json.dumps(summary, ensure_ascii=False, indent=2))
print()
print('written to', DEST)
