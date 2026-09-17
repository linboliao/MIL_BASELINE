#!/usr/bin/env python
"""Audit every extracted patch set for magnification consistency.

Bug found: create_patches_fp.py adjusts patch_size by magnification via
adjust_size() ONLY in the single-process path. mp_seg_and_patch (--use_mp) does
NOT -> 40x slides get the same fixed patch_size as 20x slides -> 40x tissue ends
up at 2x zoom for the encoder.

For each slide with an h5 coord file:
  * read slide objective power + mpp
  * infer the patch step actually used, from the coords
  * effective FOV per patch (um) = step_px * mpp   (should be ~constant if OK)
  * effective um/encoder-pixel = FOV_um / 224

Report per cohort + flag slides whose effective um/px deviates from the training
median by >20%.

env: LD_LIBRARY_PATH=<clam>/lib   (openslide + Aslide)
out: /NAS2/Data1/lbliao/Code-195/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis/patch_mpp_audit/
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, '/NAS2/Data1/lbliao/Code-195/PrePATH')
MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
F = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
WR = '/NAS145/linboliao/Data/迈新生物'
ENC_IN = 224
OUT = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/patch_mpp_audit'


def open_wsi(p):
    ext = p.rsplit('.', 1)[-1].lower()
    if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
        import openslide
        s = openslide.OpenSlide(p)
        mpp = s.properties.get('openslide.mpp-x') or s.properties.get('aperio.MPP')
        mag = s.properties.get('openslide.objective-power') or s.properties.get('aperio.AppMag')
        return float(mpp) if mpp else None, float(mag) if mag else None
    if ext in ('kfb', 'tmap', 'sdpc'):
        from wsi_core.Aslide.aslide import Slide
        s = Slide(p)
        try:
            mpp = float(s.mpp) if s.mpp else None
        except Exception:
            mpp = None
        try:
            mag = float(s.objective_power) if s.objective_power else None
        except Exception:
            mag = None
        return mpp, mag
    return None, None


def infer_step(coords):
    """most common small positive gap between sorted-unique x (and y), = step_size."""
    out = []
    for ax in (0, 1):
        v = np.unique(coords[:, ax])
        d = np.diff(v)
        d = d[(d > 0) & (d < 5000)]
        if len(d):
            vals, cnts = np.unique(d, return_counts=True)
            out.append(int(vals[np.argmax(cnts)]))
    return min(out) if out else None


def wsi_index(root):
    idx = {}
    for f in glob.glob(os.path.join(root, '**'), recursive=True):
        e = f.rsplit('.', 1)[-1].lower()
        if e in ('svs', 'kfb', 'ndpi', 'tif', 'tiff', 'mrxs'):
            idx['.'.join(os.path.basename(f).split('.')[:-1])] = f
    return idx


def audit(cohort, pool, id_source, wsi_root, limit=None):
    coords_dir = f'{F}/{pool}/patches_0_224/patches'
    widx = wsi_index(wsi_root)
    rows = []
    ids = id_source if limit is None else id_source[:limit]
    for i, (sid, typ) in enumerate(ids):
        h5 = f'{coords_dir}/{sid}.h5'
        if not os.path.exists(h5) or sid not in widx:
            continue
        try:
            with h5py.File(h5, 'r') as h:
                c = h['coords'][:]
            mpp, mag = open_wsi(widx[sid])
        except Exception as e:  # noqa
            print('  err', sid, repr(e)[:80])
            continue
        step = infer_step(c)
        fov_um = step * mpp if (step and mpp) else None
        rows.append(dict(cohort=cohort, sid=sid, type=typ, mpp=mpp, mag=mag,
                         n_patch=len(c), step_px=step,
                         fov_um=round(fov_um, 1) if fov_um else None,
                         um_per_encpx=round(fov_um / ENC_IN, 3) if fov_um else None))
        if i % 40 == 0:
            print(f'  [{cohort}] {i}/{len(ids)}', flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []

    d3 = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/external_test_301.csv')
    rows += audit('301', 'MIL外部测试', list(zip(d3.slide_id.astype(str), d3['type'])),
                  f'{WR}/MIL外部测试')
    dy = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/external_test_ynzl.csv')
    rows += audit('ynzl', 'MIL外部测试', list(zip(dy.slide_id.astype(str), dy['type'])),
                  f'{WR}/MIL外部测试')
    it = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/internal_test.csv').sample(120, random_state=1)
    rows += audit('internal', 'MIL训练数据', list(zip(it.slide_id.astype(str), it['type'])),
                  f'{WR}/MIL训练数据')
    dev = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/dev.csv').sample(200, random_state=1)
    rows += audit('train', 'MIL训练数据', list(zip(dev.slide_id.astype(str), dev['type'])),
                  f'{WR}/MIL训练数据')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/per_slide.csv', index=False, encoding='utf-8-sig')

    tr = df[df.cohort == 'train'].um_per_encpx.dropna()
    ref = float(tr.median()) if len(tr) else None
    df['deviation_x'] = df.um_per_encpx / ref if ref else np.nan
    df['FLAG_scale_mismatch'] = df.deviation_x.apply(lambda x: bool(x and (x < 0.8 or x > 1.25)))

    L = ['PATCH MAGNIFICATION AUDIT', '=' * 70, '',
         f'encoder input {ENC_IN}px ; training median um/encoder-px = {ref}', '']
    for c in ['train', 'internal', 'ynzl', '301']:
        s = df[df.cohort == c]
        if not len(s):
            continue
        L.append(f'### {c}  (n={len(s)})')
        L.append(f'  mpp:          {s.mpp.describe()[["min","50%","max"]].round(3).to_dict()}')
        L.append(f'  objective:    {s.mag.value_counts().to_dict()}')
        L.append(f'  step_px used: {s.step_px.value_counts().to_dict()}')
        L.append(f'  FOV um/patch: {s.fov_um.describe()[["min","50%","max"]].round(1).to_dict()}')
        L.append(f'  um/encoder-px: median {s.um_per_encpx.median():.3f}  (x{s.um_per_encpx.median()/ref:.2f} vs training)')
        L.append(f'  FLAGGED scale-mismatch: {int(s.FLAG_scale_mismatch.sum())}/{len(s)}')
        L.append('')
    L.append('by cohort x type - median um/encoder-px:')
    piv = df.pivot_table(index='cohort', columns='type', values='um_per_encpx', aggfunc='median').round(3)
    L.append(piv.to_string())
    txt = '\n'.join(L)
    open(f'{OUT}/SUMMARY.txt', 'w').write(txt + '\n')
    json.dump({'training_ref_um_per_encpx': ref,
               'n_flagged': int(df.FLAG_scale_mismatch.sum()),
               'flagged_by_cohort': df[df.FLAG_scale_mismatch].cohort.value_counts().to_dict()},
              open(f'{OUT}/summary.json', 'w'), indent=2)
    print('\n' + txt)
    print(f'\nsaved -> {OUT}/  (per_slide.csv, SUMMARY.txt, summary.json)')


if __name__ == '__main__':
    main()
