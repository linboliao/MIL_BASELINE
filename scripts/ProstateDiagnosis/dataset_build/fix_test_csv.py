"""Apply the adjudicated label corrections to test.csv.

Decisions (from the pathologist review):
  - 1740359.8 : the panel sections match the original slide, and the original
                is benign (manifest=0, source folder 无癌, 20例 Sheet1 label=0).
                The model calls it positive at every centre because of
                urothelium / squamous epithelium. Correct 1 -> 0.
  - 124714.1  : four centres scanned it out of focus, so the case is unusable
                rather than mislabelled. Drop the row entirely.
  - C2105119N : slide re-checked and found sound; keep as-is (label 0).
"""
import os
import shutil

import pandas as pd

PATH = '/NAS145/linboliao/Data/迈新生物/连续切片跨中心染色/6家医院各60张/test.csv'
BAK = PATH + '.bak_before_label_fix'

df = pd.read_csv(PATH)
print('before: %d rows, balance=%s' % (len(df), df['label'].value_counts().to_dict()))

if not os.path.exists(BAK):
    shutil.copyfile(PATH, BAK)
    print('backup written ->', BAK)
else:
    print('backup already exists ->', BAK)

ids = df['slide_id'].astype(str).str.strip()

before_1740 = df.loc[ids == '1740359.8', 'label'].tolist()
df.loc[ids == '1740359.8', 'label'] = 0
print(f'1740359.8: {before_1740} -> [0]')

n_before = len(df)
df = df[ids != '124714.1'].reset_index(drop=True)
print(f'124714.1: removed {n_before - len(df)} row(s) (out-of-focus scans at 4 centres)')

kept = df[df['slide_id'].astype(str).str.strip() == 'C2105119N']
print('C2105119N kept as-is:', kept['label'].tolist())

df.to_csv(PATH, index=False)
print()
print('after: %d rows, balance=%s' % (len(df), df['label'].value_counts().to_dict()))
print('written ->', PATH)
