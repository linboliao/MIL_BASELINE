"""Consolidate the serial-section cross-centre staining panel into clean tables.

Layout
------
`6家医院各60张` is the 6-centre paired panel: one block per case was cut into
serial sections and each centre received ONE fixed section index, so centre and
section index are 1:1 (广西瑞康=2, 南京军区总院=3, 301=5, 云南省肿瘤=6, 900=7,
河南安阳=9). Centre comes from the directory name and is cross-checked against
the filename suffix.

`6家单位20例切片染色图像` is a flat 20-case set with section suffixes 1-6. No
file records which centre produced which suffix, so `center` stays empty there
and only `section_idx` is populated.

Ground truth vs predictions
---------------------------
GROUND TRUTH
  - `test.csv`                        : per-case labels (adjudicated)
  - `结果汇总`.病理医生                : pathologist read, 60 cases
  - `染色归一化结果`.Sheet1.病理医生    : pathologist read, 60 cases
  - `20例`.Sheet1.label               : per-case labels for the 20-case panel

MODEL PREDICTIONS - deliberately NOT used as labels, saved separately:
  - `60例`.Sheet1.结果                : per-section predictions (sections 2/6/7).
    Verified against test.csv at only 91% agreement, with errors clustering by
    case across all three sections - the signature of a model, not a reader.
  - `结果汇总` per-centre columns      : per-centre predictions, raw and
    stain-normalised (91.7-96.7% agreement with truth).
  - `20例`.Sheet1.prediction.*        : per-section predictions for the 20 cases.

There is therefore NO per-section pathologist read anywhere in this data, so the
biological section-to-section drift cannot be estimated from it - an open
limitation for any paired cross-centre analysis built on this panel.
"""
import json
import os
import re

import pandas as pd

ROOT = '/NAS145/linboliao/Data/迈新生物/连续切片跨中心染色'
SET6 = os.path.join(ROOT, '6家医院各60张')
SET20 = os.path.join(ROOT, '6家单位20例切片染色图像')
MANIFEST = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/manifest.csv'
OUT_DIR = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/serial_sections'

EXTS = ('.svs', '.kfb', '.ndpi', '.tif', '.tiff', '.mrxs', '.sdpc')
CENTER_SECTION = {'广西瑞康': 2, '南京军区总院': 3, '301': 5,
                  '云南省肿瘤': 6, '900': 7, '河南安阳': 9}
ID_FIXES = {'MX1638897.18': '1638897.18', 'MX1642001.4': '1642001.4', '211158N': '2111158N'}

# Cases withdrawn by the pathologist review (unusable images, not mislabels).
EXCLUDED_CASES = {'124714.1': 'out-of-focus scans at 4 centres (图片拼接质量问题)'}

os.makedirs(OUT_DIR, exist_ok=True)


def norm(s):
    s = str(s).strip().rstrip('-')
    return ID_FIXES.get(s, s)


def parse_stem(stem):
    m = re.match(r'^(.*?)-(\d+)$', stem)
    return (norm(m.group(1)), int(m.group(2))) if m else (norm(stem), None)


# ------------------------------------------------------------ slide inventory
rows = []
for center in sorted(os.listdir(SET6)):
    cdir = os.path.join(SET6, center)
    if not os.path.isdir(cdir):
        continue
    for dp, _, fns in os.walk(cdir):
        for fn in fns:
            if not fn.lower().endswith(EXTS):
                continue
            case_id, sec = parse_stem(os.path.splitext(fn)[0])
            exp = CENTER_SECTION.get(center)
            rows.append({'set': '6center_panel', 'slide_path': os.path.join(dp, fn),
                         'filename': fn, 'case_id': case_id,
                         'section_idx': sec if sec is not None else exp,
                         'section_from_filename': sec, 'center': center,
                         'section_matches_center': (sec == exp) if sec is not None else None})

for dp, _, fns in os.walk(SET20):
    for fn in fns:
        if not fn.lower().endswith(EXTS):
            continue
        case_id, sec = parse_stem(os.path.splitext(fn)[0])
        rows.append({'set': '20case_panel', 'slide_path': os.path.join(dp, fn),
                     'filename': fn, 'case_id': case_id, 'section_idx': sec,
                     'section_from_filename': sec, 'center': None,
                     'section_matches_center': None})

slides = pd.DataFrame(rows).sort_values(['set', 'case_id', 'section_idx']).reset_index(drop=True)
slides['excluded'] = slides['case_id'].isin(EXCLUDED_CASES)
slides['exclusion_reason'] = slides['case_id'].map(EXCLUDED_CASES)

# ------------------------------------------------------------ GROUND TRUTH
gt = []


def add_gt(df, id_col, label_col, source, strip_section=False):
    d = df[[id_col, label_col]].dropna()
    d.columns = ['case_id', 'label']
    d['case_id'] = d['case_id'].astype(str)
    if strip_section:
        d['case_id'] = d['case_id'].str.replace(r'-\d+$', '', regex=True)
    d['case_id'] = d['case_id'].map(norm)
    d['label'] = pd.to_numeric(d['label'], errors='coerce')
    d = d.dropna(subset=['label'])
    d['label'] = d['label'].astype(int)
    d['source'] = source
    gt.append(d.drop_duplicates())


wb60 = os.path.join(SET6, '60例不同医院染色).xlsx')
wb20 = os.path.join(SET20, '20例不同医院染色结果分析.xlsx')

add_gt(pd.read_csv(os.path.join(SET6, 'test.csv')), 'slide_id', 'label', 'test.csv')
add_gt(pd.read_excel(wb60, sheet_name='结果汇总'), 'slide_id', '病理医生', '结果汇总.病理医生')
add_gt(pd.read_excel(os.path.join(SET6, '染色归一化结果.xlsx'), sheet_name='Sheet1'),
       'slide_id', '病理医生', '染色归一化结果.病理医生')
add_gt(pd.read_excel(wb20, sheet_name='Sheet1'), 'slide_id', 'label', '20例.Sheet1.label',
       strip_section=True)

labels_long = pd.concat(gt, ignore_index=True).drop_duplicates()
labels_long = labels_long[~labels_long['case_id'].isin(EXCLUDED_CASES)]
labels_long.to_csv(os.path.join(OUT_DIR, 'serial_labels_by_source.csv'),
                   index=False, encoding='utf-8-sig')

agg = labels_long.groupby('case_id')['label'].agg(['nunique', 'first', 'count'])
case_labels = agg[['first', 'count']].rename(columns={'first': 'label', 'count': 'n_sources'})
case_labels['has_conflict'] = agg['nunique'] > 1
case_labels = case_labels.reset_index()
case_labels['label_sources'] = case_labels['case_id'].map(
    labels_long.groupby('case_id')['source'].apply(lambda x: ';'.join(sorted(set(x)))))

# ------------------------------------------------------------ MODEL PREDICTIONS
preds = []

s1 = pd.read_excel(wb60, sheet_name='Sheet1')[['切片号', '结果']].dropna()
s1['case_id'] = s1['切片号'].astype(str).str.replace(r'-\d+$', '', regex=True).map(norm)
s1['section_idx'] = pd.to_numeric(s1['切片号'].astype(str).str.extract(r'-(\d+)$')[0], errors='coerce')
s1['prediction'] = pd.to_numeric(s1['结果'], errors='coerce')
s1['source'] = '60例.Sheet1'
s1['center'] = s1['section_idx'].map({v: k for k, v in CENTER_SECTION.items()})
preds.append(s1.dropna(subset=['section_idx', 'prediction'])[
    ['case_id', 'section_idx', 'center', 'prediction', 'source']])

s20 = pd.read_excel(wb20, sheet_name='Sheet1')
for i in range(6):
    sid_col = 'slide_id' if i == 0 else f'slide_id.{i}'
    prd_col = 'prediction' if i == 0 else f'prediction.{i}'
    if sid_col not in s20.columns:
        continue
    d = s20[[sid_col, prd_col]].dropna()
    d.columns = ['slide_id', 'prediction']
    d = d[d['slide_id'].astype(str).str.contains('-')]
    d['case_id'] = d['slide_id'].astype(str).str.replace(r'-\d+$', '', regex=True).map(norm)
    d['section_idx'] = pd.to_numeric(
        d['slide_id'].astype(str).str.extract(r'-(\d+)$')[0], errors='coerce')
    d['prediction'] = pd.to_numeric(d['prediction'], errors='coerce')
    d['center'] = None
    d['source'] = '20例.Sheet1'
    preds.append(d.dropna(subset=['section_idx', 'prediction'])[
        ['case_id', 'section_idx', 'center', 'prediction', 'source']])

summ = pd.read_excel(wb60, sheet_name='结果汇总')
summ['case_id'] = summ['slide_id'].map(norm)
for col in summ.columns:
    if col in ('slide_id', 'case_id', '病理医生', '备注') or not isinstance(col, str):
        continue
    base = col.replace('-归一化', '')
    if base not in ('广西瑞康', '南京军总', '301', '云南肿瘤', '900', '迈新', '河南安阳', '陕西肿瘤', '河南'):
        continue
    d = summ[['case_id', col]].dropna()
    d.columns = ['case_id', 'prediction']
    d['prediction'] = pd.to_numeric(d['prediction'], errors='coerce')
    d['center'] = base
    d['section_idx'] = None
    d['source'] = f'结果汇总.{col}'
    preds.append(d.dropna(subset=['prediction'])[
        ['case_id', 'section_idx', 'center', 'prediction', 'source']])

predictions = pd.concat(preds, ignore_index=True)
predictions['prediction'] = predictions['prediction'].astype(int)
predictions.to_csv(os.path.join(OUT_DIR, 'prior_model_predictions.csv'),
                   index=False, encoding='utf-8-sig')

# ------------------------------------------------------------ main-cohort join
man = pd.read_csv(MANIFEST)
man['slide_id'] = man['slide_id'].astype(str)
lut = man.drop_duplicates('slide_id').set_index('slide_id')

case_labels['in_main_cohort'] = case_labels['case_id'].isin(lut.index)
for col in ['label', 'type', 'center', 'pool', 'patient_id']:
    case_labels[f'main_{col}'] = case_labels['case_id'].map(lut[col])
case_labels['label_matches_main'] = case_labels.apply(
    lambda r: bool(r['label'] == r['main_label']) if pd.notna(r['main_label']) else None, axis=1)
case_labels = case_labels.sort_values('case_id').reset_index(drop=True)
case_labels.to_csv(os.path.join(OUT_DIR, 'serial_case_labels.csv'),
                   index=False, encoding='utf-8-sig')

slides = slides.merge(
    case_labels[['case_id', 'label', 'has_conflict', 'in_main_cohort',
                 'main_type', 'main_center', 'main_pool', 'main_patient_id']],
    on='case_id', how='left')
slides.to_csv(os.path.join(OUT_DIR, 'serial_slides_manifest.csv'),
              index=False, encoding='utf-8-sig')

usable = slides[~slides['excluded']]
p6 = usable[usable['set'] == '6center_panel']
cov = p6.groupby('case_id')['center'].nunique()
complete = sorted(cov[cov == 6].index)
labeled = set(case_labels[case_labels['label'].notna()]['case_id'])
complete_labeled = sorted(set(complete) & labeled)
pd.DataFrame({'case_id': complete_labeled}).to_csv(
    os.path.join(OUT_DIR, 'paired_cases_all6centers.csv'), index=False, encoding='utf-8-sig')

panel_patients = set(case_labels['main_patient_id'].dropna().astype(str))
excl_slides = sorted(man[man['patient_id'].astype(str).isin(panel_patients)]['slide_id'])
pd.DataFrame({'slide_id': excl_slides}).to_csv(
    os.path.join(OUT_DIR, 'exclude_from_training_slides.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'patient_id': sorted(panel_patients)}).to_csv(
    os.path.join(OUT_DIR, 'exclude_from_training_patients.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame([{'case_id': k, 'reason': v} for k, v in EXCLUDED_CASES.items()]).to_csv(
    os.path.join(OUT_DIR, 'excluded_cases.csv'), index=False, encoding='utf-8-sig')

report = {
    'total_slide_files': int(len(slides)),
    'excluded_slide_files': int(slides['excluded'].sum()),
    'usable_slide_files': int(len(usable)),
    'by_set': {k: int(v) for k, v in usable.groupby('set').size().items()},
    'slides_per_center_6panel': {k: int(v) for k, v in p6.groupby('center').size().items()},
    'section_index_per_center': CENTER_SECTION,
    'section_filename_mismatches': int((p6['section_matches_center'] == False).sum()),
    'unique_cases_6panel': int(p6['case_id'].nunique()),
    'cases_in_all_6_centers_with_label': len(complete_labeled),
    'unique_cases_20panel': int(usable[usable['set'] == '20case_panel']['case_id'].nunique()),
    'cases_with_label': int(case_labels['label'].notna().sum()),
    'cases_with_label_conflict': int(case_labels['has_conflict'].sum()),
    'cases_in_main_cohort': int(case_labels['in_main_cohort'].sum()),
    'label_disagreements_vs_main': int((case_labels['label_matches_main'] == False).sum()),
    'excluded_cases': EXCLUDED_CASES,
    'patients_to_exclude_from_training': len(panel_patients),
    'slides_to_exclude_from_training': len(excl_slides),
    'label_balance': {str(k): int(v) for k, v in case_labels['label'].value_counts().items()},
    'prior_prediction_rows': int(len(predictions)),
    'per_section_pathologist_reads_available': False,
}
with open(os.path.join(OUT_DIR, 'serial_sections_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print()
print('=== label conflicts across ground-truth sources ===')
bad = case_labels[case_labels['has_conflict']]['case_id'].tolist()
print(labels_long[labels_long['case_id'].isin(bad)].sort_values(['case_id', 'source']).to_string(index=False)
      if bad else 'none')
print()
print('=== panel label vs main-cohort label ===')
dis = case_labels[case_labels['label_matches_main'] == False]
print(dis[['case_id', 'label', 'main_label', 'main_type', 'label_sources']].to_string(index=False)
      if len(dis) else 'none')
print()
print('=== specimen types (from main cohort) ===')
print(case_labels.groupby(['main_type', 'label']).size().to_string())
print()
print('=== slides with no label (excluding withdrawn cases) ===')
nl = usable[usable['label'].isna()]['case_id'].unique().tolist()
print(nl if nl else 'none')
print()
print('written to', OUT_DIR)
