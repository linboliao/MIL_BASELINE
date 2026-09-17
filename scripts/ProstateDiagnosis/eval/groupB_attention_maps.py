#!/usr/bin/env python
"""Group B (viz) - attention heatmaps for representative slides.

For a hand-picked set (301 false-positive benign RP/TURP, 301 true-positive
cancer, and internal benign RP correctly scored low for contrast), load the
[N,2560] features + patch coords, run fold-1's AB_MIL, and render the per-patch
attention over slide space next to the stitch thumbnail.

Outputs -> <run_dir>/groupB_analysis/attention_maps/<slide>.png  + index.json
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
sys.path.insert(0, REPO)
os.chdir(REPO)
RES_ROOT = os.path.join(REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
FEAT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis'
NAS_H5 = {
    'ext_301': f'{FEAT}/MIL外部测试/feat_0_224/h5_files/virchow2',
    'ext_ynzl': f'{FEAT}/MIL外部测试/feat_0_224/h5_files/virchow2',
    'internal': f'{FEAT}/MIL训练数据/feat_0_224/h5_files/virchow2',
}
STITCH = {
    'ext_301': f'{FEAT}/MIL外部测试/patches_0_224/stitches',
    'internal': f'{FEAT}/MIL训练数据/patches_0_224/stitches',
}
DEVICE = 'cuda:0'
N_PER_GROUP = 4


def load_model(run_dir):
    from modules.AB_MIL.ab_mil import AB_MIL
    from utils.process_utils import get_act
    from utils.yaml_utils import read_yaml
    ya = read_yaml(os.path.join(run_dir, 'fold_1', 'fold_1.yaml'))
    m = AB_MIL(L=ya.Model.L, D=ya.Model.D, num_classes=2, dropout=ya.Model.dropout,
              act=get_act(ya.Model.act), in_dim=ya.Model.in_dim)
    ck = sorted(glob.glob(os.path.join(run_dir, 'fold_1', 'Best_EPOCH_*.pth')))[-1]
    m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
    return m.to(DEVICE).eval()


def pick(run_dir):
    b = pd.read_csv(os.path.join(run_dir, 'groupB_analysis', 'slide_meta.csv'))
    b['slide_id'] = b['slide_id'].astype(str)
    groups = {}
    d = b[(b.cohort == 'ext_301') & (b.label == 0) & (b.type == 'RP')].sort_values('prob', ascending=False)
    groups['301_benign_RP_FALSEPOS'] = d.head(N_PER_GROUP)
    d = b[(b.cohort == 'ext_301') & (b.label == 0) & (b.type == 'TURP')].sort_values('prob', ascending=False)
    groups['301_benign_TURP_FALSEPOS'] = d.head(N_PER_GROUP)
    d = b[(b.cohort == 'ext_301') & (b.label == 1)].sort_values('prob', ascending=False)
    groups['301_cancer_truepos'] = d.head(2)
    d = b[(b.cohort == 'internal') & (b.label == 0) & (b.type.isin(['RP', 'TURP']))].sort_values('prob')
    groups['internal_benign_RPTURP_correct'] = d.head(N_PER_GROUP)
    return groups


@torch.no_grad()
def render(model, row, out_dir):
    sid, coh = row['slide_id'], row['cohort']
    feat_path = row['path'] if 'path' in row and isinstance(row['path'], str) else None
    if not feat_path or not os.path.exists(feat_path):
        return None
    h5p = os.path.join(NAS_H5.get(coh, ''), sid + '.h5')
    if not os.path.exists(h5p):
        return None
    x = torch.load(feat_path, map_location='cpu')
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.float()
    with h5py.File(h5p, 'r') as h:
        coords = h['coords'][:]
    if len(coords) != len(x):
        n = min(len(coords), len(x))
        coords, x = coords[:n], x[:n]
    out = model(x.to(DEVICE), return_WSI_attn=True)
    a = torch.softmax(out['WSI_attn'].squeeze(-1), dim=-1).cpu().numpy()
    ar = (np.argsort(np.argsort(a)) / (len(a) - 1))  # rank-normalized 0..1

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    stp = os.path.join(STITCH.get(coh, ''), sid + '.jpg')
    if os.path.exists(stp):
        try:
            img = plt.imread(stp)
            ax[0].imshow(img)
            ax[0].set_title(f'{sid}  stitch')
        except Exception:
            pass
    ax[0].axis('off')
    sc = ax[1].scatter(coords[:, 0], -coords[:, 1], c=ar, s=6, cmap='inferno')
    ax[1].set_aspect('equal'); ax[1].axis('off')
    ax[1].set_title(f"attention (rank-norm)  label={row['label']}  prob={row['prob']:.3f}\n"
                    f"type={row['type']} center={row['center']}  N={len(a)}  max_w={a.max():.3f}")
    plt.colorbar(sc, ax=ax[1], fraction=0.04)
    plt.tight_layout()
    p = os.path.join(out_dir, f'{sid}.png')
    plt.savefig(p, dpi=90, bbox_inches='tight')
    plt.close()
    return {'slide_id': sid, 'cohort': coh, 'type': row['type'], 'center': row['center'],
            'label': int(row['label']), 'prob': float(row['prob']), 'n_patch': int(len(a)),
            'max_attn': float(a.max()), 'png': os.path.basename(p)}


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1]
    meta = pd.read_csv(os.path.join(run_dir, 'groupB_analysis', 'slide_meta.csv'))
    meta['slide_id'] = meta['slide_id'].astype(str)
    path_by_id = dict(zip(meta.slide_id, meta.path))
    out_dir = os.path.join(run_dir, 'groupB_analysis', 'attention_maps')
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(run_dir)

    index = {}
    for gname, g in pick(run_dir).items():
        index[gname] = []
        for _, row in g.iterrows():
            row = row.to_dict()
            row['path'] = path_by_id.get(row['slide_id'])
            try:
                r = render(model, row, out_dir)
            except Exception as e:  # noqa
                print('  fail', row['slide_id'], repr(e))
                r = None
            if r:
                index[gname].append(r)
                print(f"  {gname}: {r['slide_id']}  prob {r['prob']:.3f}  max_attn {r['max_attn']:.3f}")
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'), indent=2, ensure_ascii=False)
    print(f'saved -> {out_dir}/')


if __name__ == '__main__':
    main()
