"""Define the two serial-section panels cleanly and mark cross-tree duplicates.

Panel A (`panel_A_6center`)
    60 cases x 6 NAMED centres (广西瑞康 §2, 南京军区总院 §3, 301 §5,
    云南省肿瘤 §6, 900 §7, 河南安阳 §9). Fully paired; the centres include 301
    and 云南省肿瘤, which are also the two independent external cohorts, so this
    panel is the one that can be tied back to real deployment performance.

Panel B (`panel_B_20case`)
    20 different cases x 6 ANONYMOUS units (U1..U6). Units U2/U3/U6 are provably
    the same scans as 广西瑞康/南京军区总院/云南省肿瘤 (byte-identical files), and
    that provenance is retained in `known_identity`, but the panel is treated as
    six anonymous units for analysis. U1/U4/U5 could not be identified from any
    available file.

Because U2/U3/U6 files are byte-identical duplicates of slides that also sit
under the 6-centre directory tree, they are flagged with `duplicate_of_panelA_tree`
so the two panels are never pooled in a way that double-counts the same scan.
"""
import hashlib
import json
import os

import pandas as pd

OUT_DIR = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/serial_sections'
MAN = os.path.join(OUT_DIR, 'serial_slides_manifest.csv')

CONFIRMED = {2: '广西瑞康', 3: '南京军区总院', 6: '云南省肿瘤'}

df = pd.read_csv(MAN)

# ---- anonymous unit ids for the 20-case panel -------------------------------
is20 = df['set'] == '20case_panel'
df['unit_id'] = None
df.loc[is20, 'unit_id'] = df.loc[is20, 'section_idx'].map(lambda s: f'U{int(s)}' if pd.notna(s) else None)
df['known_identity'] = None
df.loc[is20, 'known_identity'] = df.loc[is20, 'section_idx'].map(CONFIRMED)

# For panel A the "unit" is simply the named centre.
df.loc[~is20, 'unit_id'] = df.loc[~is20, 'center']
df.loc[~is20, 'known_identity'] = df.loc[~is20, 'center']

# ---- flag scans that exist in both directory trees --------------------------
def head_md5(path, nbytes=2 * 1024 * 1024):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read(nbytes)).hexdigest()
    except OSError:
        return None


sizes = {}
for _, r in df.iterrows():
    try:
        sizes[r['slide_path']] = os.path.getsize(r['slide_path'])
    except OSError:
        sizes[r['slide_path']] = None
df['file_size'] = df['slide_path'].map(sizes)

dup_flags = []
panelA_by_name = {}
for _, r in df[~is20].iterrows():
    panelA_by_name.setdefault(r['filename'], []).append(r['slide_path'])

for _, r in df.iterrows():
    if r['set'] != '20case_panel':
        dup_flags.append(False)
        continue
    cands = panelA_by_name.get(r['filename'], [])
    hit = False
    for p in cands:
        if sizes.get(p) == r['file_size'] and head_md5(p) == head_md5(r['slide_path']):
            hit = True
            break
    dup_flags.append(hit)
df['duplicate_of_panelA_tree'] = dup_flags

# ---- panel assignment -------------------------------------------------------
p6 = df[~is20]
cov = p6.groupby('case_id')['center'].nunique()
panelA_cases = set(cov[cov == 6].index) - set(df[df['excluded']]['case_id'])
panelB_cases = set(df[is20 & ~df['excluded']]['case_id'])

df['panel'] = None
df.loc[(~is20) & df['case_id'].isin(panelA_cases) & (~df['excluded']), 'panel'] = 'A_6center'
df.loc[is20 & df['case_id'].isin(panelB_cases) & (~df['excluded']), 'panel'] = 'B_20case_anon'

df.to_csv(MAN, index=False, encoding='utf-8-sig')

# ---- per-panel tables -------------------------------------------------------
A = df[df['panel'] == 'A_6center']
B = df[df['panel'] == 'B_20case_anon']
A.to_csv(os.path.join(OUT_DIR, 'panel_A_6center.csv'), index=False, encoding='utf-8-sig')
B.to_csv(os.path.join(OUT_DIR, 'panel_B_20case_anon.csv'), index=False, encoding='utf-8-sig')

report = {
    'panel_A_6center': {
        'cases': int(A['case_id'].nunique()),
        'slides': int(len(A)),
        'centers': sorted(A['center'].dropna().unique().tolist()),
        'label_balance': {str(k): int(v) for k, v in
                          A.drop_duplicates('case_id')['label'].value_counts().items()},
        'note': 'fully paired; includes 301 and 云南省肿瘤 which are also external test cohorts',
    },
    'panel_B_20case_anon': {
        'cases': int(B['case_id'].nunique()),
        'slides': int(len(B)),
        'units': sorted(B['unit_id'].dropna().unique().tolist()),
        'units_with_known_identity': {
            u: B[B['unit_id'] == u]['known_identity'].dropna().unique().tolist()
            for u in sorted(B['unit_id'].dropna().unique())},
        'slides_duplicated_in_panelA_tree': int(B['duplicate_of_panelA_tree'].sum()),
        'label_balance': {str(k): int(v) for k, v in
                          B.drop_duplicates('case_id')['label'].value_counts().items()},
        'note': 'treated as six anonymous units; U2/U3/U6 are byte-identical to named centres',
    },
    'cases_disjoint_between_panels': len(panelA_cases & panelB_cases) == 0,
}
with open(os.path.join(OUT_DIR, 'panels_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print()
print('=== panel B: slides per anonymous unit ===')
print(B.groupby(['unit_id', 'known_identity'], dropna=False).size().to_string())
print()
print('=== panel A: slides per centre ===')
print(A.groupby('center').size().to_string())
