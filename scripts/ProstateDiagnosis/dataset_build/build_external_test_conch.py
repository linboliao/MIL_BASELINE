"""Build CONCH-feature external test CSVs (301, ynzl) pointing at the
MPP-corrected feature root, mirroring the existing uni2 external_test/*.csv
but with fresh paths + the conch model dir. No patient overlap with the
CONCH training cohort (verified separately: 0/147 and 0/183)."""
import os

import pandas as pd

DS = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis'
OUT = f'{DS}/DataAnalysis/external_test'
FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis'
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试',
            'ext_301': 'MIL外部测试', 'ext_ynzl': 'MIL外部测试'}
MODEL = 'conch'

for name in ['301', 'ynzl']:
    df = pd.read_csv(f'{DS}/external_test_{name}.csv', dtype=str)
    df['label'] = df['label'].astype(int)

    def feat_path(row):
        d = POOL_DIR.get(row['pool'])
        stem = os.path.splitext(str(row['filename']))[0]
        return f'{FEAT_ROOT}/{d}/feat_0_224/pt_files/{MODEL}/{stem}.pt'

    df['test_slide_path'] = df.apply(feat_path, axis=1)
    missing = df[~df['test_slide_path'].map(os.path.exists)]
    print(f'{name}: {len(df)} rows, missing conch feature: {len(missing)}')
    if len(missing):
        print(missing[['slide_id', 'pool', 'test_slide_path']].to_string())
    df = df[df['test_slide_path'].map(os.path.exists)].reset_index(drop=True)

    out = df[['test_slide_path']].copy()
    out['test_label'] = df['label']
    out_path = f'{OUT}/external_test_{name}_conch.csv'
    out.to_csv(out_path, index=False)
    print(f'  usable: {len(out)}  pos_rate={out["test_label"].mean():.3f}  -> {out_path}')
