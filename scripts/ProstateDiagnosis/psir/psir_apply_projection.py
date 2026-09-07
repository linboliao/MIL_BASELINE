"""PSIR step 3: apply a trained fold's projection head to CONCH patch features,
writing a parallel `conch_psir_fold{k}` feature tree that plugs into the
existing fold-build / train_mil.py pipeline unchanged (same directory
layout, just a different `--model` name).

Applies to:
  - the main cohort (dev_clean + internal_test_clean) -- for stage-2 classifier training
  - Panel A's HELD-OUT cases only (never the training-signal cases, to keep
    the later held-out evaluation honest)
  - all of Panel B (fully independent validation set)
"""
import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn

DS = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis'
PSIR_DIR = f'{DS}/psir'
FEAT_ROOT = '/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis'
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试'}
MODEL_SRC = 'conch'
MODEL_DST_TMPL = 'conch_psir_fold{fold}'
IN_DIM = 512
PROJ_DIM = 256


class ProjHead(nn.Module):
    def __init__(self, in_dim=IN_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def project_and_save(src_path, dst_path, proj, device):
    if os.path.exists(dst_path):
        return 'skip'
    feats = torch.load(src_path, map_location='cpu', weights_only=True).float()
    with torch.no_grad():
        out = proj(feats.to(device)).cpu()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(out, dst_path)
    return 'done'


def main(fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proj = ProjHead().to(device).eval()
    ckpt = f'{PSIR_DIR}/proj_heads/fold{fold}_proj.pt'
    proj.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model_dst = MODEL_DST_TMPL.format(fold=fold)
    print(f'loaded {ckpt}, writing projected features as model="{model_dst}"')

    counts = {'main_cohort': 0, 'panel_a_held_out': 0, 'panel_b': 0, 'skipped': 0}

    # --- main cohort ---
    dev = pd.read_csv(f'{DS}/dev_clean.csv', dtype={'slide_id': str})
    itest = pd.read_csv(f'{DS}/internal_test_clean.csv', dtype={'slide_id': str})
    for df in (dev, itest):
        for _, row in df.iterrows():
            d = POOL_DIR.get(row['pool'])
            if d is None:
                continue
            stem = os.path.splitext(str(row['filename']))[0]
            src = f'{FEAT_ROOT}/{d}/feat_0_224/pt_files/{MODEL_SRC}/{stem}.pt'
            dst = f'{FEAT_ROOT}/{d}/feat_0_224/pt_files/{model_dst}/{stem}.pt'
            if not os.path.exists(src):
                continue
            r = project_and_save(src, dst, proj, device)
            counts['main_cohort' if r == 'done' else 'skipped'] += 1

    # --- Panel A held-out cases only ---
    folds_df = pd.read_csv(f'{PSIR_DIR}/panel_a_case_folds.csv', dtype={'case_id': str})
    slides_df = pd.read_csv(f'{PSIR_DIR}/panel_a_usable_slides.csv', dtype={'case_id': str})
    held_cases = set(folds_df.loc[folds_df[f'fold{fold}'] == 'held_out', 'case_id'])
    held_slides = slides_df[slides_df['case_id'].isin(held_cases)]
    for _, row in held_slides.iterrows():
        stem = os.path.splitext(str(row['filename']))[0]
        src = f'{FEAT_ROOT}/SerialPanelA/feat_0_224/pt_files/{MODEL_SRC}/{stem}.pt'
        dst = f'{FEAT_ROOT}/SerialPanelA/feat_0_224/pt_files/{model_dst}/{stem}.pt'
        if not os.path.exists(src):
            continue
        r = project_and_save(src, dst, proj, device)
        counts['panel_a_held_out' if r == 'done' else 'skipped'] += 1

    # --- Panel B: fully independent, always projected regardless of fold ---
    panel_b_dir = f'{FEAT_ROOT}/SerialPanelB/feat_0_224/pt_files/{MODEL_SRC}'
    if os.path.isdir(panel_b_dir):
        for fname in os.listdir(panel_b_dir):
            if not fname.endswith('.pt'):
                continue
            src = f'{panel_b_dir}/{fname}'
            dst = f'{FEAT_ROOT}/SerialPanelB/feat_0_224/pt_files/{model_dst}/{fname}'
            r = project_and_save(src, dst, proj, device)
            counts['panel_b' if r == 'done' else 'skipped'] += 1

    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    main(args.fold)
