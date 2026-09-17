#!/usr/bin/env python
"""Full magnification scan of every WSI in the three prostate pools.

Reads objective power for every slide, flags the ones NOT ~20x (i.e. the ones
whose current fixed-224px patches are at the wrong scale). Also joins which
cohort/split each slide belongs to (dev / internal_test / 301 / ynzl / other).

env: LD_LIBRARY_PATH=<clam>/lib
out: /NAS2/Data1/lbliao/Code-195/MIL_BASELINE/result/ProstateDiagnosis/DataAnalysis/mag_scan/
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, '/NAS2/Data1/lbliao/Code-195/PrePATH')
MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
WR = '/NAS145/linboliao/Data/迈新生物'
F = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
POOLS = ['MIL训练数据', 'MIL测试数据', 'MIL外部测试']
OUT = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/mag_scan'


def mag_of(path):
    ext = path.rsplit('.', 1)[-1].lower()
    try:
        if ext in ('svs', 'tif', 'tiff', 'ndpi', 'mrxs'):
            import openslide
            s = openslide.OpenSlide(path)
            mag = s.properties.get('openslide.objective-power') or s.properties.get('aperio.AppMag')
            mpp = s.properties.get('openslide.mpp-x') or s.properties.get('aperio.MPP')
            s.close()
            return (float(mag) if mag else None, float(mpp) if mpp else None)
        if ext in ('kfb', 'tmap', 'sdpc'):
            from wsi_core.Aslide.aslide import Slide
            s = Slide(path)
            mag = getattr(s, 'objective_power', None)
            mpp = getattr(s, 'mpp', None)
            try:
                s.close()
            except Exception:
                pass
            return (float(mag) if mag else None, float(mpp) if mpp else None)
    except Exception as e:  # noqa
        return ('ERR:' + repr(e)[:60], None)
    return (None, None)


def split_map():
    m = {}
    for name, tag in [('dev', 'dev'), ('internal_test', 'internal_test'),
                      ('external_test_301', '301'), ('external_test_ynzl', 'ynzl')]:
        try:
            d = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/{name}.csv')
            for s in d['slide_id'].astype(str):
                m[s] = tag
        except Exception:
            pass
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    smap = split_map()
    slides = []
    for pool in POOLS:
        for f in glob.glob(os.path.join(WR, pool, '**'), recursive=True):
            e = f.rsplit('.', 1)[-1].lower()
            if e in ('svs', 'kfb', 'ndpi', 'tif', 'tiff', 'mrxs', 'sdpc', 'tmap'):
                sid = '.'.join(os.path.basename(f).split('.')[:-1])
                slides.append((pool, sid, f))
    print(f'{len(slides)} WSI files across {POOLS}', flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(mag_of, f): (pool, sid, f) for pool, sid, f in slides}
        for i, fut in enumerate(as_completed(futs)):
            pool, sid, f = futs[fut]
            mag, mpp = fut.result()
            has_patch = os.path.exists(f'{F}/{pool}/patches_0_224/patches/{sid}.h5')
            rows.append(dict(pool=pool, slide_id=sid, split=smap.get(sid, 'other'),
                             objective=mag, mpp=mpp, has_h5=has_patch, wsi=f))
            if i % 200 == 0:
                print(f'  {i}/{len(slides)}', flush=True)

    df = pd.DataFrame(rows)
    df['is_40x'] = df.objective.apply(lambda m: isinstance(m, (int, float)) and m and m > 30)
    df['is_20x'] = df.objective.apply(lambda m: isinstance(m, (int, float)) and m and 10 < m <= 30)
    df['mag_unknown'] = ~(df.is_40x | df.is_20x)
    df.to_csv(f'{OUT}/all_slides_mag.csv', index=False, encoding='utf-8-sig')

    L = ['FULL MAGNIFICATION SCAN', '=' * 60, '', f'{len(df)} WSI', '']
    L.append('--- by pool ---')
    L.append(df.groupby('pool')[['is_20x', 'is_40x', 'mag_unknown']].sum().to_string())
    L.append('')
    L.append('--- by split (the ones that matter for the models) ---')
    L.append(df.groupby('split')[['is_20x', 'is_40x', 'mag_unknown']].sum().to_string())
    L.append('')
    L.append('--- objective value counts ---')
    L.append(df.objective.astype(str).value_counts().to_string())
    L.append('')
    wrong = df[df.is_40x & df.has_h5]
    L.append(f'--- 40x slides WITH existing (wrong-scale) 224px patches: {len(wrong)} ---')
    L.append(wrong.groupby(['pool', 'split']).size().to_string())
    df[df.is_40x].to_csv(f'{OUT}/slides_40x.csv', index=False, encoding='utf-8-sig')
    df[df.mag_unknown].to_csv(f'{OUT}/slides_mag_unknown.csv', index=False, encoding='utf-8-sig')

    txt = '\n'.join(L)
    open(f'{OUT}/SUMMARY.txt', 'w').write(txt + '\n')
    json.dump({'n_total': len(df), 'n_40x': int(df.is_40x.sum()),
               'n_40x_with_patches': int(len(wrong)),
               'n_unknown': int(df.mag_unknown.sum()),
               'by_split_40x': df[df.is_40x].split.value_counts().to_dict()},
              open(f'{OUT}/summary.json', 'w'), indent=2)
    print('\n' + txt)
    print(f'\nsaved -> {OUT}/  (all_slides_mag.csv, slides_40x.csv, slides_mag_unknown.csv, SUMMARY.txt)')


if __name__ == '__main__':
    main()
