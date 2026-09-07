"""Build 5-fold CONCH splits on the cleaned cohort (dev_clean.csv / internal_test_clean.csv).

Reuses the exact same patient-level StratifiedGroupKFold (seed=42) as the uni2
clean rebuild, so folds are reproducible from the already-finalized patient
sets. Feature paths point at a LOCAL disk cache (copied from NAS145 first)
instead of NAS145 directly, to avoid the NFS single-connection bottleneck
during training.
"""
import json
import os
import shutil
import subprocess
from datetime import datetime

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'
DS = f'{ROOT}/datasets/ProstateDiagnosis'
NAS_FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis'
LOCAL_CACHE = '/data5/lbliao_prostate_cache'
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试'}
MODEL = 'uni2'
NEW_FOLDS = f'{DS}/DataAnalysis/AB_MIL_uni2_5fold_3center'
SEED = 42
N_SPLITS = 5

dev_c = pd.read_csv(f'{DS}/dev_clean.csv', dtype={'slide_id': str, 'patient_id': str})
itest_c = pd.read_csv(f'{DS}/internal_test_clean.csv', dtype={'slide_id': str, 'patient_id': str})
print(f'dev_clean {len(dev_c)} slides/{dev_c["patient_id"].nunique()} pts | '
      f'internal_test_clean {len(itest_c)} slides/{itest_c["patient_id"].nunique()} pts')
print('dev pool values:', dict(dev_c['pool'].value_counts()))
print('itest pool values:', dict(itest_c['pool'].value_counts()))


def nas_path(row):
    d = POOL_DIR.get(row['pool'])
    if d is None:
        return None
    stem = os.path.splitext(str(row['filename']))[0]
    return f'{NAS_FEAT_ROOT}/{d}/feat_0_224/pt_files/{MODEL}/{stem}.pt'


def local_path(row):
    d = POOL_DIR.get(row['pool'])
    stem = os.path.splitext(str(row['filename']))[0]
    return f'{LOCAL_CACHE}/{d}/feat_0_224/pt_files/{MODEL}/{stem}.pt'


for df in (dev_c, itest_c):
    df['nas_feat'] = df.apply(nas_path, axis=1)
    df['local_feat'] = df.apply(local_path, axis=1)

missing_dev = dev_c[~dev_c['nas_feat'].map(lambda p: bool(p) and os.path.exists(p))]
missing_test = itest_c[~itest_c['nas_feat'].map(lambda p: bool(p) and os.path.exists(p))]
print(f'NAS conch feature missing: dev {len(missing_dev)}, internal_test {len(missing_test)}')
dev_c = dev_c[dev_c['nas_feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
itest_c = itest_c[itest_c['nas_feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
print(f'usable -> dev {len(dev_c)} | internal_test {len(itest_c)}')

# ---------------------------------------------------------------- local disk cache
print('\ncaching needed .pt files to local disk ...')
needed_dirs = set()
for df in (dev_c, itest_c):
    for p in df['local_feat']:
        needed_dirs.add(os.path.dirname(p))
for d in needed_dirs:
    os.makedirs(d, exist_ok=True)

to_copy = []
for df in (dev_c, itest_c):
    for nas_p, loc_p in zip(df['nas_feat'], df['local_feat']):
        if not os.path.exists(loc_p) or os.path.getsize(loc_p) != os.path.getsize(nas_p):
            to_copy.append((nas_p, loc_p))
print(f'{len(to_copy)} files to copy (of {len(dev_c) + len(itest_c)} total)')

for i, (src, dst) in enumerate(to_copy):
    shutil.copyfile(src, dst)
    if (i + 1) % 200 == 0:
        print(f'  copied {i + 1}/{len(to_copy)}')
print('cache copy done.')

# ---------------------------------------------------------------- 5-fold CV (identical scheme to uni2 clean rebuild)
pat = (dev_c.groupby('patient_id')
       .agg(label=('label', lambda s: int(s.max())),
            center=('center', lambda s: s.mode().iat[0]))
       .reset_index())
pat['stratum'] = pat['label'].astype(str) + '|' + pat['center'].astype(str)
vc = pat['stratum'].value_counts()
pat.loc[pat['stratum'].isin(vc[vc < N_SPLITS].index), 'stratum'] = pat['label'].astype(str)

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(pat, pat['stratum'], groups=pat['patient_id']))

test_paths = itest_c['local_feat'].tolist()
test_labels = itest_c['label'].tolist()

os.makedirs(NEW_FOLDS, exist_ok=True)
summary = []
for k, (tr_idx, va_idx) in enumerate(folds, start=1):
    tr_pat = set(pat.loc[tr_idx, 'patient_id'])
    va_pat = set(pat.loc[va_idx, 'patient_id'])
    assert not (tr_pat & va_pat)

    tr = dev_c[dev_c['patient_id'].isin(tr_pat)]
    va = dev_c[dev_c['patient_id'].isin(va_pat)]

    n = max(len(tr), len(va), len(test_paths))
    out = pd.DataFrame({
        'train_slide_path': tr['local_feat'].tolist() + [None] * (n - len(tr)),
        'train_label': tr['label'].tolist() + [None] * (n - len(tr)),
        'val_slide_path': va['local_feat'].tolist() + [None] * (n - len(va)),
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
    summary.append(row)
    print(f'fold{k}: train {len(tr):4d}/{len(tr_pat):4d}pts | val {len(va):4d}/{len(va_pat):3d} | test {len(test_paths)}')

sm = pd.DataFrame(summary)
print(sm.to_string(index=False))
sm.to_csv(f'{NEW_FOLDS}/fold_summary.csv', index=False, encoding='utf-8-sig')

# verification
print('\n=== verification ===')
for k in range(1, N_SPLITS + 1):
    df = pd.read_csv(f'{NEW_FOLDS}/fold_{k}/prostate_dev_uni2_{k}fold.csv')
    stems = {g: set(df[f'{g}_slide_path'].dropna().map(lambda p: os.path.splitext(os.path.basename(p))[0]))
             for g in ['train', 'val', 'test']}
    print(f'  fold{k}: train∩val={len(stems["train"] & stems["val"])} '
          f'train∩test={len(stems["train"] & stems["test"])} val∩test={len(stems["val"] & stems["test"])}')

meta = {'seed': SEED, 'n_splits': N_SPLITS, 'model': MODEL,
        'dev_slides': int(len(dev_c)), 'internal_test_slides': int(len(itest_c)),
        'local_cache': LOCAL_CACHE}
with open(f'{NEW_FOLDS}/rebuild_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))
