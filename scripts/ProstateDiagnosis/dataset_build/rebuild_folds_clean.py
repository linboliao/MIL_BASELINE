"""Rebuild the 5-fold MIL splits after removing every contaminated patient.

Exclusion is at PATIENT level: if any of a patient's slides turns up in the
serial-section panel, the human-vs-AI comparison set, or the duplicate-slide
list, all of that patient's slides are dropped from both the development pool
and the fixed internal test set. Tissue from one patient is too correlated
across blocks to leave the rest of it in training.

The dev / internal_test boundary is NOT re-randomised - that split was agreed
earlier and re-drawing it would invalidate comparisons against existing runs.
Only the 5-fold CV inside the surviving dev pool is regenerated, keeping the
same (label|center)-stratified, patient-grouped scheme.
"""
import json
import os
import shutil
from datetime import datetime

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'
DS = f'{ROOT}/datasets/ProstateDiagnosis'
OLD_FOLDS = f'{DS}/DataAnalysis/AB_MIL_uni2_5fold_3center'
NEW_FOLDS = f'{DS}/DataAnalysis/AB_MIL_uni2_5fold_3center'
BAK_ROOT = f'{DS}/DataAnalysis/_bak_{datetime.now():%Y%m%d_%H%M%S}'
SEED = 42
N_SPLITS = 5

CACHE = '/data5/lbliao_prostate_cache'
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试'}

# ---------------------------------------------------------------- exclusions
excl = pd.read_csv(f'{DS}/exclusions_master.csv', dtype={'slide_id': str, 'patient_id': str})
excl_patients = set(excl['patient_id'].dropna())
print(f'contaminated patients: {len(excl_patients)}')

dev = pd.read_csv(f'{DS}/dev.csv', dtype={'slide_id': str, 'patient_id': str})
itest = pd.read_csv(f'{DS}/internal_test.csv', dtype={'slide_id': str, 'patient_id': str})
print(f'before  -> dev {len(dev)} slides/{dev["patient_id"].nunique()} pts | '
      f'internal_test {len(itest)} slides/{itest["patient_id"].nunique()} pts')

dev_c = dev[~dev['patient_id'].isin(excl_patients)].copy()
itest_c = itest[~itest['patient_id'].isin(excl_patients)].copy()
print(f'after   -> dev {len(dev_c)} slides/{dev_c["patient_id"].nunique()} pts '
      f'(-{len(dev) - len(dev_c)}) | internal_test {len(itest_c)} slides/'
      f'{itest_c["patient_id"].nunique()} pts (-{len(itest) - len(itest_c)})')
print()
print('dev label balance :', dict(dev_c['label'].value_counts()))
print('test label balance:', dict(itest_c['label'].value_counts()))
print('dev by centre     :', dict(dev_c['center'].value_counts()))
print('test by centre    :', dict(itest_c['center'].value_counts()))
print()

# ---------------------------------------------------------------- feature paths
def feat_path(row):
    d = POOL_DIR.get(row['pool'])
    if d is None:
        return None
    stem = os.path.splitext(str(row['filename']))[0]
    return f'{CACHE}/{d}/feat_0_224/pt_files/uni2/{stem}.pt'


for df in (dev_c, itest_c):
    df['feat'] = df.apply(feat_path, axis=1)

missing_dev = dev_c[~dev_c['feat'].map(lambda p: bool(p) and os.path.exists(p))]
missing_test = itest_c[~itest_c['feat'].map(lambda p: bool(p) and os.path.exists(p))]
print(f'feature files missing: dev {len(missing_dev)}, internal_test {len(missing_test)}')
if len(missing_dev):
    print('  dev missing:', missing_dev['slide_id'].tolist()[:10])
if len(missing_test):
    print('  test missing:', missing_test['slide_id'].tolist()[:10])
dev_c = dev_c[dev_c['feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
itest_c = itest_c[itest_c['feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
print(f'usable  -> dev {len(dev_c)} | internal_test {len(itest_c)}')
print()

# ---------------------------------------------------------------- backup
os.makedirs(BAK_ROOT, exist_ok=True)
if os.path.isdir(OLD_FOLDS):
    shutil.copytree(OLD_FOLDS, f'{BAK_ROOT}/AB_MIL_uni2_5fold_3center')
    print(f'old folds backed up -> {BAK_ROOT}/AB_MIL_uni2_5fold_3center')
for f in ['dev.csv', 'internal_test.csv']:
    shutil.copyfile(f'{DS}/{f}', f'{BAK_ROOT}/{f}')
print(f'dev.csv / internal_test.csv backed up -> {BAK_ROOT}')
print()

# ---------------------------------------------------------------- 5-fold CV
pat = (dev_c.groupby('patient_id')
       .agg(label=('label', lambda s: int(s.max())),
            center=('center', lambda s: s.mode().iat[0]))
       .reset_index())
pat['stratum'] = pat['label'].astype(str) + '|' + pat['center'].astype(str)
vc = pat['stratum'].value_counts()
pat.loc[pat['stratum'].isin(vc[vc < N_SPLITS].index), 'stratum'] = pat['label'].astype(str)
print('patient strata:', dict(pat['stratum'].value_counts()))
print()

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(pat, pat['stratum'], groups=pat['patient_id']))

test_paths = itest_c['feat'].tolist()
test_labels = itest_c['label'].tolist()

summary = []
for k, (tr_idx, va_idx) in enumerate(folds, start=1):
    tr_pat = set(pat.loc[tr_idx, 'patient_id'])
    va_pat = set(pat.loc[va_idx, 'patient_id'])
    assert not (tr_pat & va_pat), 'patient leaked between train and val'

    tr = dev_c[dev_c['patient_id'].isin(tr_pat)]
    va = dev_c[dev_c['patient_id'].isin(va_pat)]

    n = max(len(tr), len(va), len(test_paths))
    out = pd.DataFrame({
        'train_slide_path': tr['feat'].tolist() + [None] * (n - len(tr)),
        'train_label': tr['label'].tolist() + [None] * (n - len(tr)),
        'val_slide_path': va['feat'].tolist() + [None] * (n - len(va)),
        'val_label': va['label'].tolist() + [None] * (n - len(va)),
        'test_slide_path': test_paths + [None] * (n - len(test_paths)),
        'test_label': test_labels + [None] * (n - len(test_paths)),
    })
    d = f'{NEW_FOLDS}/fold_{k}'
    os.makedirs(d, exist_ok=True)
    for old in os.listdir(d):
        if old.endswith('.csv'):
            os.remove(os.path.join(d, old))
    out.to_csv(f'{d}/prostate_dev_uni2_{k}fold.csv', index=False)

    row = {'fold': k, 'train_slides': len(tr), 'train_patients': len(tr_pat),
           'val_slides': len(va), 'val_patients': len(va_pat),
           'test_slides': len(test_paths),
           'train_pos_rate': round(tr['label'].mean(), 3),
           'val_pos_rate': round(va['label'].mean(), 3)}
    for c in ['新昌', '迈新', '省立']:
        row[f'train_{c}'] = int((tr['center'] == c).sum())
        row[f'val_{c}'] = int((va['center'] == c).sum())
    summary.append(row)
    print(f'fold{k}: train {len(tr):4d} slides/{len(tr_pat):4d} pts | '
          f'val {len(va):4d}/{len(va_pat):3d} | test {len(test_paths)}')

sm = pd.DataFrame(summary)
print()
print(sm.to_string(index=False))
sm.to_csv(f'{NEW_FOLDS}/fold_summary.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- verification
print()
print('=== verification ===')
all_dev_pat = set(dev_c['patient_id'])
all_test_pat = set(itest_c['patient_id'])
print('dev/internal_test patient overlap :', len(all_dev_pat & all_test_pat))
print('excluded patients still present    :',
      len((all_dev_pat | all_test_pat) & excl_patients))
for k in range(1, N_SPLITS + 1):
    df = pd.read_csv(f'{NEW_FOLDS}/fold_{k}/prostate_dev_uni2_{k}fold.csv')
    stems = {g: set(df[f'{g}_slide_path'].dropna().map(
        lambda p: os.path.splitext(os.path.basename(p))[0])) for g in ['train', 'val', 'test']}
    print(f'  fold{k}: train∩val={len(stems["train"] & stems["val"])} '
          f'train∩test={len(stems["train"] & stems["test"])} '
          f'val∩test={len(stems["val"] & stems["test"])}')

meta = {
    'seed': SEED, 'n_splits': N_SPLITS,
    'excluded_patients': len(excl_patients),
    'dev_slides_before': int(len(dev)), 'dev_slides_after': int(len(dev_c)),
    'internal_test_slides_before': int(len(itest)), 'internal_test_slides_after': int(len(itest_c)),
    'dev_patients_after': int(dev_c['patient_id'].nunique()),
    'internal_test_patients_after': int(itest_c['patient_id'].nunique()),
    'backup_dir': BAK_ROOT,
}
with open(f'{NEW_FOLDS}/rebuild_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print()
print(json.dumps(meta, ensure_ascii=False, indent=2))

dev_c.drop(columns=['feat']).to_csv(f'{DS}/dev_clean.csv', index=False, encoding='utf-8-sig')
itest_c.drop(columns=['feat']).to_csv(f'{DS}/internal_test_clean.csv', index=False, encoding='utf-8-sig')
print()
print('written dev_clean.csv / internal_test_clean.csv')
