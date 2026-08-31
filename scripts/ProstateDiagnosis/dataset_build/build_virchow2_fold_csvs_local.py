import os
import pandas as pd

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'
SRC_DIR = os.path.join(ROOT, 'datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center')
DST_DIR = os.path.join(ROOT, 'datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center')

OLD_PREFIX = '/data5/lbliao_prostate_cache/'
NEW_PREFIX = '/data14/lbliao_prostate_cache_virchow2/'


def convert_path(path):
    if not isinstance(path, str) or not path:
        return path
    assert path.startswith(OLD_PREFIX), f'unexpected path prefix: {path}'
    rest = path[len(OLD_PREFIX):]
    rest = rest.replace('/pt_files/uni2/', '/pt_files/virchow2/')
    return NEW_PREFIX + rest


missing = []
for fold in range(1, 6):
    src_path = os.path.join(SRC_DIR, f'fold_{fold}', f'prostate_dev_uni2_{fold}fold.csv')
    df = pd.read_csv(src_path)
    for group in ['train', 'val', 'test']:
        col = f'{group}_slide_path'
        df[col] = df[col].apply(convert_path)

    dst_dir = os.path.join(DST_DIR, f'fold_{fold}')
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f'prostate_dev_virchow2_{fold}fold.csv')
    df.to_csv(dst_path, index=False)

    n_checked = 0
    n_missing = 0
    for group in ['train', 'val', 'test']:
        for p in df[f'{group}_slide_path'].dropna():
            n_checked += 1
            if not os.path.exists(p):
                n_missing += 1
                missing.append(p)
    print(f'fold{fold}: wrote {dst_path} ({len(df)} rows), checked {n_checked} paths, missing {n_missing}')

if missing:
    print(f'\nWARNING: {len(missing)} missing files total, first 10:')
    for p in missing[:10]:
        print(' ', p)
else:
    print('\nAll local paths exist. Good to go.')
