"""Build the eval CSVs for the CONCH-baseline-vs-PSIR panel consistency
comparison: Panel A's fold-1 held-out cases (never seen by the fold-1
projection head) + all of Panel B (fully independent), once for raw conch
features and once for the fold-1 PSIR-projected features."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DS = str(ROOT / "datasets/ProstateDiagnosis")
PSIR_DIR = f"{DS}/psir"
FEAT_ROOT = os.environ.get("PROSTATE_FEAT_ROOT", "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")
OUT = os.environ.get("PSIR_PANEL_EVAL_DIR", f"{PSIR_DIR}/panel_eval")
os.makedirs(OUT, exist_ok=True)

folds_df = pd.read_csv(f'{PSIR_DIR}/panel_a_case_folds.csv', dtype={'case_id': str})
slides_df = pd.read_csv(f'{PSIR_DIR}/panel_a_usable_slides.csv', dtype={'case_id': str})
held_cases = set(folds_df.loc[folds_df['fold1'] == 'held_out', 'case_id'])
held_a = slides_df[slides_df['case_id'].isin(held_cases)].copy()
print(f'Panel A held-out (fold1): {len(held_a)} slides, {held_a["case_id"].nunique()} cases')

manifest = pd.read_csv(f'{DS}/serial_sections/panel_B_20case_anon.csv', dtype={'case_id': str})
manifest = manifest[manifest['excluded'] == False]  # noqa: E712
print(f'Panel B: {len(manifest)} slides, {manifest["case_id"].nunique()} cases')


def build(model, held_a_df, panel_b_df, out_suffix):
    rows = []
    for _, row in held_a_df.iterrows():
        stem = os.path.splitext(str(row['filename']))[0]
        path = f'{FEAT_ROOT}/SerialPanelA/feat_0_224/pt_files/{model}/{stem}.pt'
        if os.path.exists(path):
            rows.append({'panel': 'A', 'case_id': row['case_id'], 'center': row['center'],
                         'slide_id': stem, 'label': int(row['label']), 'test_slide_path': path,
                         'test_label': int(row['label'])})
    for _, row in panel_b_df.iterrows():
        stem = os.path.splitext(str(row['filename']))[0]
        path = f'{FEAT_ROOT}/SerialPanelB/feat_0_224/pt_files/{model}/{stem}.pt'
        if os.path.exists(path):
            rows.append({'panel': 'B', 'case_id': row['case_id'], 'center': row.get('unit_id', ''),
                         'slide_id': stem, 'label': int(row['label']), 'test_slide_path': path,
                         'test_label': int(row['label'])})
    df = pd.DataFrame(rows)
    df[['test_slide_path', 'test_label']].to_csv(f'{OUT}/eval_{out_suffix}.csv', index=False)
    df.to_csv(f'{OUT}/manifest_{out_suffix}.csv', index=False, encoding='utf-8-sig')
    print(f'{out_suffix}: {len(df)} rows -> {OUT}/eval_{out_suffix}.csv')
    return df


build('conch', held_a, manifest, 'conch_baseline')
build('conch_psir_fold1', held_a, manifest, 'conch_psir_fold1')
