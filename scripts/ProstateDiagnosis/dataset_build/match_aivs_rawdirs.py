"""Last pass for the 7 still-unidentified AI-VS slides.

Earlier passes searched the curated tables (manifest.csv, external_test_*.csv),
which exclude slides dropped during dataset construction. This pass walks the
RAW directory trees instead, so slides that never made it into a table are
still findable.
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, '/NAS3/lbliao/Code-138/PrePATH')

OUT = '/tmp/aivs_metadata_match'
SIZE = (256, 256)
ROOTS = [
    '/NAS145/linboliao/Data/迈新生物_svs重复',
    '/NAS145/linboliao/Data/迈新生物/MIL外部测试',
    '/NAS145/linboliao/Data/迈新生物/MIL测试数据',
    '/NAS145/linboliao/Data/迈新生物/MIL训练数据',
]
EXTS = ('.svs', '.kfb', '.ndpi', '.tif', '.tiff', '.mrxs', '.sdpc')

REMAINING = ['202603311.5', '202603311.6', '202603312.2', '202603312.3',
             '202603313.3', '202603314.3', '202603315.5']


def open_slide(path):
    ext = path.rsplit('.', 1)[-1].lower()
    if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
        import openslide
        return openslide.OpenSlide(path), 'openslide'
    from wsi_core.Aslide.aslide import Slide
    return Slide(path), 'aslide'


def read_dims(path):
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
        return int(w), int(hh), (round(float(mpp), 6) if mpp else None)
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


targets = pd.read_csv(f'{OUT}/targets_meta.csv', dtype={'renamed_id': str})
targets = targets[targets['renamed_id'].isin(REMAINING)]
want = {(int(r['w']), int(r['h'])) for _, r in targets.iterrows() if pd.notna(r['w'])}
print('target dimensions to look for:', sorted(want))
print()

# already-known paths, so we can report whether a hit is a *new* file
known = set()
for f in ['cohort_meta.csv', 'external_meta.csv']:
    p = f'{OUT}/{f}'
    if os.path.exists(p):
        known |= set(pd.read_csv(p)['path'].dropna())

files = []
for root in ROOTS:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                files.append(os.path.join(dp, fn))
print(f'raw files found: {len(files)} ({len(set(files) - known)} not seen in earlier passes)')
print()

hits = []
for i, p in enumerate(files):
    w, h, mpp = read_dims(p)
    if w is None:
        continue
    if (w, h) in want:
        hits.append({'path': p, 'w': w, 'h': h, 'mpp': mpp,
                     'new_file': p not in known})
        print(f'  dim hit: {os.path.basename(p)} {w}x{h} mpp={mpp} new={p not in known}', flush=True)
    if (i + 1) % 500 == 0:
        print(f'  ...scanned {i + 1}/{len(files)}', flush=True)

print()
print(f'dimension hits: {len(hits)}')
rows = []
for _, t in targets.iterrows():
    rid = str(t['renamed_id'])
    cands = [h for h in hits if h['w'] == int(t['w']) and h['h'] == int(t['h'])]
    if not cands:
        print(f'  {rid:<14} -> still nothing')
        rows.append({'renamed_id': rid, 'match_path': None, 'corr': None})
        continue
    tv = thumb_vec(t['path'])
    best, bestc = None, -2
    for c in cands:
        try:
            corr = float(np.dot(tv, thumb_vec(c['path'])) / len(tv))
        except Exception:
            continue
        if corr > bestc:
            bestc, best = corr, c
    print(f'  {rid:<14} -> {os.path.basename(best["path"]) if best else None} corr={bestc:.3f}', flush=True)
    rows.append({'renamed_id': rid,
                 'match_path': best['path'] if best else None,
                 'match_name': os.path.basename(best['path']) if best else None,
                 'corr': bestc, 'new_file': best['new_file'] if best else None})

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/rawdir_match_result.csv', index=False, encoding='utf-8-sig')
print()
print(res.to_string(index=False))
