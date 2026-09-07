"""Verify the metadata-based matches by comparing actual image content, and
work out where the 11 unmatched slides came from.

Dimension+mpp agreement is suggestive but could coincide, so every candidate
pair is confirmed by correlating low-resolution grayscale thumbnails. For the
unmatched targets, the mpp values are compared against the per-centre mpp
profile of the study to at least establish which cohort they plausibly belong to.

Note: ids such as `202603311.1` look numeric, so every id column is read as str.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, '/NAS3/lbliao/Code-138/PrePATH')

OUT = '/tmp/aivs_metadata_match'
SIZE = (256, 256)


def open_slide(path):
    ext = path.rsplit('.', 1)[-1].lower()
    if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
        import openslide
        return openslide.OpenSlide(path), 'openslide'
    from wsi_core.Aslide.aslide import Slide
    return Slide(path), 'aslide'


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
        g = im.convert('L').resize(SIZE)
        a = np.asarray(g, dtype=np.float32).ravel()
        return (a - a.mean()) / (a.std() + 1e-6)
    finally:
        try:
            h.close()
        except Exception:
            pass


STR_COLS = {'renamed_id': str, 'matched_slide_ids': str, 'slide_id': str, 'patient_id': str}
res = pd.read_csv(f'{OUT}/match_result.csv', dtype=STR_COLS)
targets = pd.read_csv(f'{OUT}/targets_meta.csv', dtype=STR_COLS)
cohort = pd.read_csv(f'{OUT}/cohort_meta.csv', dtype=STR_COLS)

tpath = dict(zip(targets['renamed_id'], targets['path']))
cpath = dict(zip(cohort['slide_id'], cohort['path']))

print('=== verifying matched pairs by thumbnail correlation ===', flush=True)
rows = []
for _, r in res.iterrows():
    ids = r.get('matched_slide_ids')
    if not isinstance(ids, str) or not ids.strip() or ids == 'nan':
        continue
    rid = str(r['renamed_id'])
    try:
        tv = thumb_vec(tpath[rid])
    except Exception as e:
        print(f'  {rid}: target thumb failed {repr(e)[:60]}', flush=True)
        continue
    for sid in ids.split(';'):
        sid = sid.strip()
        if not sid or sid not in cpath:
            continue
        try:
            cv = thumb_vec(cpath[sid])
            corr = float(np.dot(tv, cv) / len(tv))
        except Exception as e:
            corr = float('nan')
            print(f'  thumb fail {sid} {repr(e)[:60]}', flush=True)
        rows.append({'renamed_id': rid, 'candidate': sid, 'corr': corr})
        print(f'  {rid:<14} vs {sid:<18} corr={corr:.3f}', flush=True)

ver = pd.DataFrame(rows)
ver.to_csv(f'{OUT}/thumbnail_verification.csv', index=False, encoding='utf-8-sig')
print()
if len(ver):
    print('correlation distribution:')
    print(ver['corr'].describe().to_string())
    print()
    print(f'pairs with corr >= 0.90: {(ver["corr"] >= 0.90).sum()}')
    print(f'pairs with corr in [0.70,0.90): {((ver["corr"] >= 0.70) & (ver["corr"] < 0.90)).sum()}')
    print(f'pairs with corr <  0.70: {(ver["corr"] < 0.70).sum()}')
    print()
    print('best candidate per renamed slide:')
    best = ver.sort_values('corr', ascending=False).drop_duplicates('renamed_id')
    print(best.sort_values('renamed_id').to_string(index=False))
print()

print('=== cohort mpp profile by centre ===')
cohort['mpp'] = pd.to_numeric(cohort['mpp'], errors='coerce')
print(cohort.dropna(subset=['mpp']).groupby(['center', 'mpp']).size().to_string())
print()

print('=== unmatched targets ===')
res['n_dim_mpp_matches'] = pd.to_numeric(res['n_dim_mpp_matches'], errors='coerce').fillna(0)
targets['mpp'] = pd.to_numeric(targets['mpp'], errors='coerce')
tm = targets.set_index('renamed_id')
for _, r in res[res['n_dim_mpp_matches'] == 0].iterrows():
    rid = str(r['renamed_id'])
    if rid not in tm.index:
        continue
    m = tm.loc[rid]
    mpp = m['mpp']
    if pd.isna(mpp):
        print(f'  {rid:<14} (mpp unknown)')
        continue
    same = cohort[(cohort['mpp'].notna()) & (abs(cohort['mpp'] - mpp) < 1e-4)]
    centers = same['center'].value_counts().to_dict() if len(same) else {}
    print(f'  {rid:<14} {m["w"]}x{m["h"]} mpp={mpp} -> cohort slides w/ same mpp: {centers}')
