"""Project the 301/ynzl external test sets through a trained PSIR fold's
projection head, then build the matching external_test_*_conch_psir_fold{k}.csv
(same format as build_external_test_conch.py's output)."""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
DS = str(ROOT / "datasets/ProstateDiagnosis")
PSIR_DIR = f"{DS}/psir"
OUT = os.environ.get("PSIR_EXTERNAL_CSV_DIR", f"{DS}/DataAnalysis/external_test")
SRC_FEAT_ROOT = os.environ.get("PSIR_CACHE_ROOT", "/data5/lbliao_prostate_cache")
DST_FEAT_ROOT = os.environ.get("PROSTATE_FEAT_ROOT", "/NAS145/linboliao/Data/迈新生物_特征/ProstateDiagnosis")
POOL_DIR = {'dev': 'MIL训练数据', 'oldtest': 'MIL测试数据', 'ext_sl': 'MIL外部测试',
            'ext_301': 'MIL外部测试', 'ext_ynzl': 'MIL外部测试'}
MODEL_SRC = 'conch'
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


def main(fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proj = ProjHead().to(device).eval()
    ckpt = f'{PSIR_DIR}/proj_heads/fold{fold}_proj.pt'
    proj.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model_dst = f'conch_psir_fold{fold}'
    print(f'loaded {ckpt} -> projecting external test sets as model="{model_dst}"')

    for name in ['301', 'ynzl']:
        df = pd.read_csv(f'{DS}/external_test_{name}.csv', dtype=str)
        df['label'] = df['label'].astype(int)
        rows = []
        missing = 0
        for _, row in df.iterrows():
            d = POOL_DIR.get(row['pool'])
            stem = os.path.splitext(str(row['filename']))[0]
            src = f'{SRC_FEAT_ROOT}/{d}/feat_0_224/pt_files/{MODEL_SRC}/{stem}.pt'
            if not os.path.exists(src):
                # fall back to NAS if not present in the local mirror
                src = f'{DST_FEAT_ROOT}/{d}/feat_0_224/pt_files/{MODEL_SRC}/{stem}.pt'
            if not os.path.exists(src):
                missing += 1
                continue
            dst = f'{DST_FEAT_ROOT}/{d}/feat_0_224/pt_files/{model_dst}/{stem}.pt'
            if not os.path.exists(dst):
                feats = torch.load(src, map_location='cpu', weights_only=True).float()
                with torch.no_grad():
                    out = proj(feats.to(device)).cpu()
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                torch.save(out, dst)
            rows.append({'test_slide_path': dst, 'test_label': int(row['label'])})
        print(f'{name}: {len(rows)} projected, {missing} missing source features')
        out_df = pd.DataFrame(rows)
        out_path = f'{OUT}/external_test_{name}_conch_psir_fold{fold}.csv'
        out_df.to_csv(out_path, index=False)
        print(f'  pos_rate={out_df["test_label"].mean():.3f} -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    main(args.fold)
