"""Match the 11 still-unidentified AI-VS slides against the cohorts that the
first pass did not cover: the 301 and 云南肿瘤 external test sets, plus the
internal_test list, in case it holds slides absent from manifest.csv.

Same two-stage approach as before: exact level-0 dimensions + mpp as a
fingerprint, then thumbnail correlation to confirm.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, '/NAS3/lbliao/Code-138/PrePATH')

DS = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis'
OUT = '/tmp/aivs_metadata_match'
SIZE = (256, 256)

UNMATCHED = ['202603311.1', '202603311.5', '202603311.6', '202603312.1', '202603312.2',
             '202603312.3', '202603313.3', '202603313.5', '202603314.3', '202603315.3',
             '202603315.5']


def open_slide(path):
    ext = path.rsplit('.', 1)[-1].lower()
    if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
        import openslide
        return openslide.OpenSlide(path), 'openslide'
    from wsi_core.Aslide.aslide import Slide
    return Slide(path), 'aslide'


def read_meta(path):
    try:
        h, kind = open_slide(path)
        try:
            w, hh = h.level_dimensions[0]
            mpp = h.properties.get('openslide.mpp-x') if kind == 'openslide' else getattr(h, 'mpp', None)
        finally:
            try:
                h.close()
            except Exception:
                pass
        return int(w), int(hh), round(float(mpp), 6) if mpp else None
    except Exception:
        return None, None, None


def thumb_vec(path):
    h, kind = open_slide(path)
    try:
        if kind == 'openslide':
            im = h.get_thumbnail(SIZE)
        else:
            lvl = h.level_count - 1
            w, hh = h.level_dimensions[lvl]
            im = h.read_region((0, 0), lvl, (w, hh))
            if not hasattr(im, 'convert'):
                im = Image.fromarray(im)
            im.thumbnail(SIZE)
        a = np.asarray(im.convert('L').resize(SIZE), dtype=np.float32).ravel()
        return (a - a.mean()) / (a.std() + 1e-6)
    finally:
        try:
            h.close()
        except Exception:
            pass


# ---- build the extra search space ----
extra = []
for name, path in [('301', f'{DS}/external_test_301.csv'),
                   ('ynzl', f'{DS}/external_test_ynzl.csv')]:
    if not os.path.exists(path):
        print('missing', path)
        continue
    d = pd.read_csv(path, dtype={'slide_id': str, 'patient_id': str})
    for _, r in d.iterrows():
        extra.append({'slide_id': r['slide_id'], 'path': r['raw_path'], 'cohort': name,
                      'label': r.get('label'), 'type': r.get('type'),
                      'center': r.get('center'), 'patient_id': r.get('patient_id')})

it = f'{DS}/internal_test.csv'
if os.path.exists(it):
    d = pd.read_csv(it, dtype={'slide_id': str, 'patient_id': str})
    known = {e['slide_id'] for e in extra}
    man_ids = set(pd.read_csv(f'{DS}/manifest.csv', dtype={'slide_id': str})['slide_id'])
    for _, r in d.iterrows():
        if r['slide_id'] in known or r['slide_id'] in man_ids:
            continue
        extra.append({'slide_id': r['slide_id'], 'path': r['raw_path'], 'cohort': 'internal_test_only',
                      'label': r.get('label'), 'type': r.get('type'),
                      'center': r.get('center'), 'patient_id': r.get('patient_id')})

edf = pd.DataFrame(extra)
print(f'extra search space: {len(edf)} slides')
print(edf['cohort'].value_counts().to_string())
print()

for i, r in edf.iterrows():
    w, h, mpp = read_meta(r['path'])
    edf.loc[i, ['w', 'h', 'mpp']] = [w, h, mpp]
    if (i + 1) % 100 == 0:
        print(f'  ...scanned {i + 1}/{len(edf)}', flush=True)
edf.to_csv(f'{OUT}/external_meta.csv', index=False, encoding='utf-8-sig')
print(f'readable: {edf["w"].notna().sum()}/{len(edf)}')
print()

targets = pd.read_csv(f'{OUT}/targets_meta.csv', dtype={'renamed_id': str})
targets = targets[targets['renamed_id'].isin(UNMATCHED)]

rows = []
for _, t in targets.iterrows():
    cands = edf[(edf['w'] == t['w']) & (edf['h'] == t['h'])]
    rid = str(t['renamed_id'])
    if len(cands) == 0:
        print(f'  {rid:<14} {int(t["w"])}x{int(t["h"])} mpp={t["mpp"]} -> NO dimension match')
        rows.append({'renamed_id': rid, 'match': None, 'corr': None, 'cohort': None})
        continue
    tv = thumb_vec(t['path'])
    best, bestc = None, -2
    for _, c in cands.iterrows():
        try:
            corr = float(np.dot(tv, thumb_vec(c['path'])) / len(tv))
        except Exception:
            corr = float('nan')
        if corr == corr and corr > bestc:
            bestc, best = corr, c
    print(f'  {rid:<14} -> {best["slide_id"]:<18} [{best["cohort"]}] corr={bestc:.3f}', flush=True)
    rows.append({'renamed_id': rid, 'match': best['slide_id'], 'corr': bestc,
                 'cohort': best['cohort'], 'label': best.get('label'),
                 'type': best.get('type'), 'center': best.get('center'),
                 'patient_id': best.get('patient_id')})

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/external_match_result.csv', index=False, encoding='utf-8-sig')
print()
print(res.to_string(index=False))
print()
ok = res[res['corr'].notna() & (res['corr'] >= 0.90)]
print(f'confirmed (corr>=0.90): {len(ok)} / {len(res)}')
if len(ok):
    print(ok.groupby('cohort').size().to_string())
