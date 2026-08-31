import os
import pandas as pd

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'
MANIFEST = os.path.join(ROOT, 'datasets/ProstateDiagnosis/manifest.csv')
SRC_DIR = os.path.join(ROOT, 'datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center')
DST_DIR = os.path.join(ROOT, 'datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_centeradv')

CENTER_TO_DOMAIN = {'新昌': 0, '迈新': 1, '省立': 2}

manifest = pd.read_csv(MANIFEST)
# The .pt feature filename mirrors the *raw* filename stem (e.g.
# "14074504.1有癌.pt"), not the normalized `slide_id` column (which strips
# the trailing 有癌/无癌 marker to "14074504.1") -- match on `filename`
# (minus its original extension) instead, and fall back to `slide_id` for
# anything that doesn't hit (belt-and-braces for pools built differently).
filename_stem_to_center = dict(zip(
    manifest['filename'].astype(str).apply(lambda f: os.path.splitext(f)[0]),
    manifest['center'],
))
slide_id_to_center = dict(zip(manifest['slide_id'].astype(str), manifest['center']))

missing = set()


def domain_for_path(path):
    if not isinstance(path, str) or not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    center = filename_stem_to_center.get(stem)
    if center is None:
        center = slide_id_to_center.get(stem)
    if center is None:
        missing.add(stem)
        return None
    return CENTER_TO_DOMAIN[center]


for fold in range(1, 6):
    src_path = os.path.join(SRC_DIR, f'fold_{fold}', f'prostate_dev_uni2_{fold}fold.csv')
    df = pd.read_csv(src_path)
    for group in ['train', 'val', 'test']:
        path_col = f'{group}_slide_path'
        label_col = f'{group}_label'
        domain_col = f'{group}_domain'
        df[domain_col] = df[path_col].apply(domain_for_path)
        # Safety net: WSI_Domain_Dataset does an independent dropna() per
        # column, so a row with a populated slide_path/label but a NaN
        # domain would silently shift every later index out of alignment.
        # Blank out slide_path+label together with any unmatched domain so
        # the three columns stay dropna()-aligned (should be a no-op once
        # domain_for_path matches everything, but never leave this to chance).
        unmatched = df[path_col].notna() & df[domain_col].isna()
        if unmatched.any():
            df.loc[unmatched, [path_col, label_col, domain_col]] = None
    dst_dir = os.path.join(DST_DIR, f'fold_{fold}')
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f'prostate_dev_uni2_centeradv_{fold}fold.csv')
    df.to_csv(dst_path, index=False)
    n_rows = len(df)
    print(f'fold{fold}: wrote {dst_path} ({n_rows} rows)')
    for group in ['train', 'val', 'test']:
        vc = df[f'{group}_domain'].dropna().value_counts().to_dict()
        print(f'  {group}_domain counts: {vc}')

if missing:
    print(f'\nWARNING: {len(missing)} slide_ids not found in manifest: {sorted(missing)[:20]}...')
else:
    print('\nAll slide_ids matched to a center. No missing domain labels.')
