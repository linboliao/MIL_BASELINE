"""Consolidate every reason a slide should be kept out of the development pool,
and measure the combined impact on the current 5-fold splits.
"""
import json
import os

import pandas as pd

DS = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis'
FOLDS = f'{DS}/DataAnalysis/AB_MIL_uni2_5fold_3center'

man = pd.read_csv(f'{DS}/manifest.csv', dtype={'slide_id': str, 'patient_id': str})
print(f'manifest: {len(man)} slides, {man["patient_id"].nunique()} patients')
print(man.groupby('pool').size().to_string())
print()

sources = {}

# 1. serial-section panel (tissue the model must not have seen)
p = f'{DS}/serial_sections/exclude_from_training_slides.csv'
if os.path.exists(p):
    sources['serial_panel'] = set(pd.read_csv(p, dtype=str)['slide_id'])

# 2. human-vs-AI comparison set leakage
p = f'{DS}/aivs_leakage/exclude_slides_patientlevel.csv'
if os.path.exists(p):
    sources['aivs_leakage'] = set(pd.read_csv(p, dtype=str)['slide_id'])

# 3. duplicate slides recorded at build time
p = f'{DS}/duplicate_slides.csv'
if os.path.exists(p):
    d = pd.read_csv(p, dtype=str)
    print('duplicate_slides.csv columns:', list(d.columns), '| rows:', len(d))
    for c in ['slide_id', 'filename']:
        if c in d.columns:
            sources['duplicate_slides'] = set(d[c].dropna().astype(str))
            break
    print()

# 4. slides whose feature file is missing / unreadable (the known corrupt KFBs)
missing = []
for fold in range(1, 6):
    fp = f'{FOLDS}/fold_{fold}/prostate_dev_uni2_{fold}fold.csv'
    if not os.path.exists(fp):
        continue
    df = pd.read_csv(fp)
    for g in ['train', 'val', 'test']:
        for pth in df[f'{g}_slide_path'].dropna():
            if not os.path.exists(pth):
                missing.append(os.path.splitext(os.path.basename(pth))[0])
if missing:
    sources['missing_features'] = set(missing)

print('=== exclusion sources ===')
for k, v in sources.items():
    print(f'  {k:20s}: {len(v)} slides')
print()

allx = set().union(*sources.values()) if sources else set()
print(f'union of all exclusions: {len(allx)} slides')
print()

# overlap matrix
keys = list(sources)
print('=== overlap between sources ===')
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        ov = sources[a] & sources[b]
        if ov:
            print(f'  {a} ∩ {b}: {len(ov)} -> {sorted(ov)[:8]}')
print('  (no pairs listed = no overlap)')
print()

inman = man[man['slide_id'].isin(allx)]
print(f'=== of those, present in the manifest: {len(inman)} ===')
print(inman.groupby(['pool', 'center']).size().to_string())
print()
print('by specimen type / label:')
print(pd.crosstab(inman['type'], inman['label'], margins=True).to_string())
print()

pats = set(inman['patient_id'])
print(f'distinct patients affected: {len(pats)}')
extra = man[man['patient_id'].isin(pats) & ~man['slide_id'].isin(allx)]
print(f'further slides from those same patients not yet on any list: {len(extra)}')
print()

# ---- impact on the current folds ----
print('=== impact on the current 5 folds ===')
rows = []
for fold in range(1, 6):
    fp = f'{FOLDS}/fold_{fold}/prostate_dev_uni2_{fold}fold.csv'
    if not os.path.exists(fp):
        continue
    df = pd.read_csv(fp)
    row = {'fold': fold}
    for g in ['train', 'val', 'test']:
        stems = set(df[f'{g}_slide_path'].dropna().map(
            lambda x: os.path.splitext(os.path.basename(x))[0]))
        row[f'{g}_total'] = len(stems)
        row[f'{g}_excluded'] = len(stems & allx)
        row[f'{g}_pct'] = round(100 * len(stems & allx) / max(len(stems), 1), 1)
    rows.append(row)
imp = pd.DataFrame(rows)
print(imp.to_string(index=False))
print()

out = f'{DS}/exclusions_master.csv'
rec = []
for src, ids in sources.items():
    for sid in sorted(ids):
        rec.append({'slide_id': sid, 'reason': src})
mdf = pd.DataFrame(rec).groupby('slide_id')['reason'].apply(lambda x: ';'.join(sorted(set(x)))).reset_index()
mdf = mdf.merge(man[['slide_id', 'patient_id', 'label', 'type', 'center', 'pool']],
                on='slide_id', how='left')
mdf['in_manifest'] = mdf['pool'].notna()
mdf.to_csv(out, index=False, encoding='utf-8-sig')
print('written', out, f'({len(mdf)} rows)')

summary = {k: len(v) for k, v in sources.items()}
summary['union'] = len(allx)
summary['in_manifest'] = int(len(inman))
summary['patients_affected'] = len(pats)
with open(f'{DS}/exclusions_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
