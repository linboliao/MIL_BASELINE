"""PSIR step 2 (Scheme A, stage 1): train a small per-patch projection head
with a supervised-contrastive loss so that the SAME case's bag representation
(mean-pooled projected patch features) ends up close together across the 6
real institutions that stained it, while different cases stay apart.

Only touches the training-signal cases of one fold -- the held-out cases and
all of Panel B are never read here.

This operates purely in feature space (pre-extracted CONCH .pt files), so it
is cheap: no WSI I/O, runs fine on a single small GPU or even CPU.
"""
import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SER = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/serial_sections'
PSIR_DIR = '/NAS3/lbliao/Code-138/MIL_BASELINE/datasets/ProstateDiagnosis/psir'
FEAT_ROOT = '/data5/lbliao_prostate_cache'  # local disk mirror, avoids NAS IO
POOL_DIR = 'SerialPanelA'
MODEL = 'conch'
IN_DIM = 512
PROJ_DIM = 256
TEMPERATURE = 0.1
LR = 1e-3
EPOCHS = 200
SEED = 42


class ProjHead(nn.Module):
    """Per-patch projection, shared weights, applied before mean-pooling."""
    def __init__(self, in_dim=IN_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def supcon_loss(z, group_ids, temperature=TEMPERATURE):
    """Supervised contrastive loss (Khosla et al. 2020) over bag embeddings.
    z: (N, D) L2-normalized bag embeddings. group_ids: (N,) case ids (as ints).
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    n = z.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    # use a large-but-finite value instead of -inf: -inf * 0 (for masked-out
    # positions in the pos_mask multiply below) is NaN, not 0, in IEEE float
    sim = sim.masked_fill(self_mask, -1e4)

    group_ids = group_ids.view(-1, 1)
    pos_mask = (group_ids == group_ids.t()) & ~self_mask

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1)[valid] / pos_counts[valid]
    return loss_per_anchor.mean()


def load_bag_features(slide_id, pool_dir=POOL_DIR, model=MODEL):
    path = f'{FEAT_ROOT}/{pool_dir}/feat_0_224/pt_files/{model}/{slide_id}.pt'
    return torch.load(path, map_location='cpu', weights_only=True)


def main(fold):
    torch.manual_seed(SEED)
    folds_df = pd.read_csv(f'{PSIR_DIR}/panel_a_case_folds.csv', dtype={'case_id': str})
    slides_df = pd.read_csv(f'{PSIR_DIR}/panel_a_usable_slides.csv', dtype={'case_id': str})

    train_cases = set(folds_df.loc[folds_df[f'fold{fold}'] == 'train_signal', 'case_id'])
    print(f'fold{fold}: {len(train_cases)} training-signal cases')

    train_slides = slides_df[slides_df['case_id'].isin(train_cases)].copy()
    stem = train_slides['filename'].map(lambda x: os.path.splitext(str(x))[0])
    train_slides = train_slides.assign(slide_stem=stem)

    missing = train_slides[~train_slides['slide_stem'].map(
        lambda s: os.path.exists(f'{FEAT_ROOT}/{POOL_DIR}/feat_0_224/pt_files/{MODEL}/{s}.pt'))]
    if len(missing):
        print(f'  missing conch features for {len(missing)} slides, dropping them')
    train_slides = train_slides[~train_slides.index.isin(missing.index)].reset_index(drop=True)

    case_to_idx = {c: i for i, c in enumerate(sorted(train_slides['case_id'].unique()))}
    print(f'  usable training-signal slides: {len(train_slides)}, '
          f'cases with features: {len(case_to_idx)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pre-load all patch features once (small dataset, fits fine)
    bag_features = []
    group_ids = []
    for _, row in train_slides.iterrows():
        feats = load_bag_features(row['slide_stem']).float()
        bag_features.append(feats)
        group_ids.append(case_to_idx[row['case_id']])
    group_ids = torch.tensor(group_ids, dtype=torch.long)

    proj = ProjHead().to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=LR, weight_decay=1e-5)

    for epoch in range(1, EPOCHS + 1):
        proj.train()
        bag_reps = []
        for feats in bag_features:
            feats = feats.to(device)
            projected = proj(feats)
            bag_reps.append(projected.mean(dim=0))
        z = torch.stack(bag_reps, dim=0)
        loss = supcon_loss(z, group_ids.to(device))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=5.0)
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f'  epoch {epoch:3d}/{EPOCHS}  supcon_loss={loss.item():.4f}')

    out_dir = f'{PSIR_DIR}/proj_heads'
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = f'{out_dir}/fold{fold}_proj.pt'
    torch.save(proj.state_dict(), ckpt_path)
    meta = {'fold': fold, 'n_train_cases': len(case_to_idx), 'n_train_slides': len(train_slides),
            'final_loss': loss.item(), 'in_dim': IN_DIM, 'proj_dim': PROJ_DIM,
            'temperature': TEMPERATURE, 'epochs': EPOCHS}
    with open(f'{out_dir}/fold{fold}_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'saved -> {ckpt_path}')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    main(args.fold)
