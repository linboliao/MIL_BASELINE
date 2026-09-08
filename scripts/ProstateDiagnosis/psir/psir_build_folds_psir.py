"""PSIR stage-2 fold builder: same 5-fold CV split as the CONCH baseline
(dev_clean/internal_test_clean, seed=42), but feature paths point at one
PSIR fold's projected features (conch_psir_fold{K}) instead of raw conch.
No local disk caching here (portable across servers) -- points straight at
NAS145. Add caching yourself if your server has a fast local scratch disk.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'  # adjust to your MIL_BASELINE checkout
DS = f'{ROOT}/datasets/ProstateDiagnosis'
NAS_FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis'
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试'}
SEED = 42
N_SPLITS = 5


def main(psir_fold):
    model = f'uni_psir_fold{psir_fold}'
    new_folds = f'{DS}/DataAnalysis/AB_MIL_{model}_5fold_3center'

    dev_c = pd.read_csv(f'{DS}/dev_clean.csv', dtype={'slide_id': str, 'patient_id': str})
    itest_c = pd.read_csv(f'{DS}/internal_test_clean.csv', dtype={'slide_id': str, 'patient_id': str})
    print(f'dev_clean {len(dev_c)} slides/{dev_c["patient_id"].nunique()} pts | '
          f'internal_test_clean {len(itest_c)} slides/{itest_c["patient_id"].nunique()} pts')

    def feat_path(row):
        d = POOL_DIR.get(row['pool'])
        if d is None:
            return None
        stem = os.path.splitext(str(row['filename']))[0]
        return f'{NAS_FEAT_ROOT}/{d}/feat_0_224/pt_files/{model}/{stem}.pt'

    for df in (dev_c, itest_c):
        df['feat'] = df.apply(feat_path, axis=1)

    missing_dev = dev_c[~dev_c['feat'].map(lambda p: bool(p) and os.path.exists(p))]
    missing_test = itest_c[~itest_c['feat'].map(lambda p: bool(p) and os.path.exists(p))]
    print(f'PSIR feature missing: dev {len(missing_dev)}, internal_test {len(missing_test)}')
    dev_c = dev_c[dev_c['feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
    itest_c = itest_c[itest_c['feat'].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)
    print(f'usable -> dev {len(dev_c)} | internal_test {len(itest_c)}')
    if len(dev_c) == 0:
        raise SystemExit(f'no usable dev features for {model} -- run psir_apply_projection.py --fold {psir_fold} first')

    pat = (dev_c.groupby('patient_id')
           .agg(label=('label', lambda s: int(s.max())),
                center=('center', lambda s: s.mode().iat[0]))
           .reset_index())
    pat['stratum'] = pat['label'].astype(str) + '|' + pat['center'].astype(str)
    vc = pat['stratum'].value_counts()
    pat.loc[pat['stratum'].isin(vc[vc < N_SPLITS].index), 'stratum'] = pat['label'].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(pat, pat['stratum'], groups=pat['patient_id']))

    test_paths = itest_c['feat'].tolist()
    test_labels = itest_c['label'].tolist()

    os.makedirs(new_folds, exist_ok=True)
    summary = []
    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr_pat = set(pat.loc[tr_idx, 'patient_id'])
        va_pat = set(pat.loc[va_idx, 'patient_id'])
        assert not (tr_pat & va_pat)
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
        d = f'{new_folds}/fold_{k}'
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            if old.endswith('.csv'):
                os.remove(os.path.join(d, old))
        out.to_csv(f'{d}/prostate_dev_{model}_{k}fold.csv', index=False)
        summary.append({'fold': k, 'train_slides': len(tr), 'val_slides': len(va), 'test_slides': len(test_paths)})
        print(f'fold{k}: train {len(tr)} | val {len(va)} | test {len(test_paths)}')

    pd.DataFrame(summary).to_csv(f'{new_folds}/fold_summary.csv', index=False, encoding='utf-8-sig')
    meta = {'seed': SEED, 'n_splits': N_SPLITS, 'model': model,
            'dev_slides': int(len(dev_c)), 'internal_test_slides': int(len(itest_c))}
    with open(f'{new_folds}/rebuild_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f'\nwritten -> {new_folds}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    main(args.fold)
