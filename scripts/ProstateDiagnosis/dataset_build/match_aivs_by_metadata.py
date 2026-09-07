"""Match the 30 renamed AI-VS slides to their originals using WSI metadata.

Byte comparison only recovered 1/30 because the rest were re-exported, but a
re-export normally preserves the pixel grid, so exact level-0 dimensions plus
mpp act as a fingerprint. Reads headers only (no pixel data).
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, '/NAS3/lbliao/Code-138/PrePATH')

AIVS = '/NAS145/linboliao/Data/迈新生物/AI VS 中级医生病灶识别'
MANIFEST = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/manifest.csv'
OUT = '/tmp/aivs_metadata_match'
os.makedirs(OUT, exist_ok=True)


def read_meta(path):
    ext = path.rsplit('.', 1)[-1].lower()
    try:
        if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
            import openslide
            h = openslide.OpenSlide(path)
            try:
                w, hh = h.level_dimensions[0]
                mpp = h.properties.get('openslide.mpp-x')
                mag = h.properties.get('openslide.objective-power')
                t = h.properties.get('aperio.Time')
                dt = h.properties.get('aperio.Date')
            finally:
                h.close()
        else:
            from wsi_core.Aslide.aslide import Slide
            h = Slide(path)
            try:
                w, hh = h.level_dimensions[0]
                mpp = getattr(h, 'mpp', None)
                mag = getattr(h, 'objective_power', None)
                t = dt = None
            finally:
                try:
                    h.close()
                except Exception:
                    pass
        return {'w': int(w), 'h': int(hh),
                'mpp': round(float(mpp), 6) if mpp else None,
                'mag': mag, 'scan_date': dt, 'scan_time': t}
    except Exception as e:
        return {'error': repr(e)[:120]}


# ---- targets ----
targets = []
for fn in sorted(os.listdir(AIVS)):
    if not fn.lower().endswith(('.svs', '.kfb', '.ndpi', '.tif', '.tiff')):
        continue
    p = os.path.join(AIVS, fn)
    m = read_meta(p)
    m.update({'renamed_id': os.path.splitext(fn)[0], 'path': p,
              'size': os.path.getsize(p)})
    targets.append(m)
    print(f'[target] {fn}: {m.get("w")}x{m.get("h")} mpp={m.get("mpp")}', flush=True)
tdf = pd.DataFrame(targets)
tdf.to_csv(f'{OUT}/targets_meta.csv', index=False, encoding='utf-8-sig')

# ---- main cohort ----
man = pd.read_csv(MANIFEST)
man['slide_id'] = man['slide_id'].astype(str)
recs = []
for i, r in man.iterrows():
    m = read_meta(r['raw_path'])
    m.update({'slide_id': r['slide_id'], 'path': r['raw_path'], 'label': r['label'],
              'type': r['type'], 'center': r['center'], 'pool': r['pool'],
              'patient_id': r['patient_id']})
    try:
        m['size'] = os.path.getsize(r['raw_path'])
    except OSError:
        m['size'] = None
    recs.append(m)
    if (i + 1) % 200 == 0:
        print(f'  ...scanned {i + 1}/{len(man)} cohort slides', flush=True)
sdf = pd.DataFrame(recs)
sdf.to_csv(f'{OUT}/cohort_meta.csv', index=False, encoding='utf-8-sig')
print(f'cohort slides with readable dims: {sdf["w"].notna().sum()}/{len(sdf)}', flush=True)

# ---- match on exact (w,h) ----
idx = defaultdict(list)
for _, r in sdf[sdf['w'].notna()].iterrows():
    idx[(int(r['w']), int(r['h']))].append(r)

rows = []
for _, t in tdf.iterrows():
    if pd.isna(t.get('w')):
        rows.append({'renamed_id': t['renamed_id'], 'n_matches': 0, 'note': 'unreadable target'})
        continue
    cands = idx.get((int(t['w']), int(t['h'])), [])
    same_mpp = [c for c in cands
                if t['mpp'] is None or c['mpp'] is None or abs(c['mpp'] - t['mpp']) < 1e-4]
    best = same_mpp or cands
    rows.append({
        'renamed_id': t['renamed_id'], 'target_dims': f'{int(t["w"])}x{int(t["h"])}',
        'target_mpp': t['mpp'], 'n_dim_matches': len(cands), 'n_dim_mpp_matches': len(same_mpp),
        'matched_slide_ids': ';'.join(c['slide_id'] for c in best),
        'matched_labels': ';'.join(str(c['label']) for c in best),
        'matched_types': ';'.join(str(c['type']) for c in best),
        'matched_centers': ';'.join(str(c['center']) for c in best),
        'matched_pools': ';'.join(str(c['pool']) for c in best),
        'matched_patients': ';'.join(str(c['patient_id']) for c in best),
    })

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/match_result.csv', index=False, encoding='utf-8-sig')
uniq = res[res['n_dim_mpp_matches'] == 1]
print()
print(f'=== unique matches: {len(uniq)} / {len(res)} ===')
print(res.to_string(index=False))
print()
print(json.dumps({'unique': int(len(uniq)),
                  'ambiguous': int((res['n_dim_mpp_matches'] > 1).sum()),
                  'none': int((res['n_dim_mpp_matches'] == 0).sum())}, indent=2))
print('written to', OUT)
