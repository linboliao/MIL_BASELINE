#!/usr/bin/env python
"""Does Macenko stain-norm help the uni2 5-fold model on 301?

Runs the frozen uni2 5-fold AB_MIL ensemble on the 301 (+ynzl control) external
cohorts twice - RAW uni2 features vs MACENKO-normalised uni2 features - and
compares. The question: does normalising 301 patches toward the training
staining recover benign RP/TURP specificity WITHOUT costing cancer sensitivity,
and without hurting ynzl?

out -> <MIL>/result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center/stainnorm_eval/
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
sys.path.insert(0, MIL)
os.chdir(MIL)
from modules.AB_MIL.ab_mil import AB_MIL          # noqa: E402
from utils.process_utils import get_act            # noqa: E402
from utils.yaml_utils import read_yaml             # noqa: E402

RUN = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center'
FEAT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis/MIL外部测试'
DEVICE = 'cuda:0'
OUT = f'{RUN}/stainnorm_eval'

_ap = argparse.ArgumentParser()
_ap.add_argument('--raw-dir', default=f'{FEAT}/feat_0_224/pt_files/uni2')
_ap.add_argument('--mac-dir', default=f'{FEAT}/feat_0_224_stains/Macenko/pt_files/uni2')
_ap.add_argument('--workers', type=int, default=16)
ARGS = _ap.parse_args()
RAW_DIR, MAC_DIR = ARGS.raw_dir, ARGS.mac_dir


def load_models():
    ms = []
    for k in range(1, 6):
        d = glob.glob(f'{RUN}/fold_{k}/*/AB_MIL/seed_*/fold_1')[0]
        ya = read_yaml(glob.glob(f'{d}/fold_{k}.yaml')[0])
        m = AB_MIL(L=ya.Model.L, D=ya.Model.D, num_classes=2, dropout=ya.Model.dropout,
                   act=get_act(ya.Model.act), in_dim=ya.Model.in_dim)
        ck = sorted(glob.glob(f'{d}/Best_EPOCH_*.pth'))[-1]
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
        ms.append(m.to(DEVICE).eval())
        print(f'fold {k}: {os.path.basename(ck)}  in_dim={ya.Model.in_dim}')
    return ms


@torch.no_grad()
def prob1(models, path):
    x = torch.load(path, map_location='cpu')
    if x.dim() == 3:
        x = x.squeeze(0)
    x = x.float().to(DEVICE)
    return [float(torch.softmax(m(x)['logits'].squeeze(0), -1)[1]) for m in models]


def M(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {'n': int(len(y)), 'prev': round(float(y.mean()), 3),
            'auc': round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            'auprc': round(float(__import__('sklearn.metrics', fromlist=['average_precision_score'])
                                 .average_precision_score(y, p)), 4) if len(set(y)) > 1 else None,
            'sens': round(float(tp / (tp + fn)), 4) if tp + fn else None,
            'spec': round(float(tn / (tn + fp)), 4) if tn + fp else None,
            'acc': round(float((tp + tn) / len(y)), 4),
            'cm': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def strata(d):  # d: columns type,label,prob
    r = {'ALL': M(d.label, d.prob)}
    for t in ['CNB', 'RP', 'TURP']:
        g = d[d.type == t]
        if len(g) >= 5:
            r[t] = M(g.label, g.prob)
    g = d[(d.label == 0) & d.type.isin(['RP', 'TURP'])]
    if len(g):
        r['benign_RPTURP_spec'] = round(float((g.prob < .5).mean()), 4)
    g = d[d.label == 1]
    if len(g):
        r['cancer_sens'] = round(float((g.prob >= .5).mean()), 4)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    models = load_models()

    rows = []
    for c in ('301', 'ynzl'):
        meta = pd.read_csv(f'{MIL}/datasets/ProstateDiagnosis/external_test_{c}.csv')
        for _, r in meta.iterrows():
            sid = str(r.slide_id)
            rp, mp = f'{RAW_DIR}/{sid}.pt', f'{MAC_DIR}/{sid}.pt'
            if not (os.path.exists(rp) and os.path.exists(mp)):
                continue
            pr = prob1(models, rp)
            pm = prob1(models, mp)
            row = dict(sid=sid, cohort=c, type=r['type'], label=int(r.label))
            for k in range(5):
                row[f'raw_f{k+1}'] = pr[k]
                row[f'mac_f{k+1}'] = pm[k]
            row['raw_ens'] = float(np.mean(pr))
            row['mac_ens'] = float(np.mean(pm))
            rows.append(row)
        print(f'{c}: {sum(1 for x in rows if x["cohort"]==c)} slides', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/slide_preds.csv', index=False, encoding='utf-8-sig')

    res = {}
    for c in ('301', 'ynzl'):
        sub = df[df.cohort == c]
        res[c] = {
            'raw_ensemble': strata(sub[['type', 'label']].assign(prob=sub.raw_ens)),
            'macenko_ensemble': strata(sub[['type', 'label']].assign(prob=sub.mac_ens)),
            'raw_per_fold': [M(sub.label, sub[f'raw_f{k+1}'])['auc'] for k in range(5)],
            'macenko_per_fold': [M(sub.label, sub[f'mac_f{k+1}'])['auc'] for k in range(5)],
            'per_fold_spec_raw': [M(sub.label, sub[f'raw_f{k+1}'])['spec'] for k in range(5)],
            'per_fold_spec_macenko': [M(sub.label, sub[f'mac_f{k+1}'])['spec'] for k in range(5)],
        }
    json.dump(res, open(f'{OUT}/results.json', 'w'), indent=2, default=float)

    L = ['UNI2  RAW vs MACENKO stain-norm  (frozen 5-fold ensemble, thr 0.5)', '=' * 74, '']
    for c in ('301', 'ynzl'):
        r = res[c]
        L.append(f'### {c}   (n={r["raw_ensemble"]["ALL"]["n"]}, cancer prev {r["raw_ensemble"]["ALL"]["prev"]})')
        for tag in ('raw_ensemble', 'macenko_ensemble'):
            a = r[tag]['ALL']
            L.append(f'  [{tag:16s}] AUC {a["auc"]}  AUPRC {a["auprc"]}  sens {a["sens"]}  spec {a["spec"]}  acc {a["acc"]}  cm {a["cm"]}')
            ex = []
            if 'benign_RPTURP_spec' in r[tag]:
                ex.append(f'benign RP+TURP spec {r[tag]["benign_RPTURP_spec"]}')
            if 'cancer_sens' in r[tag]:
                ex.append(f'cancer sens {r[tag]["cancer_sens"]}')
            if ex:
                L.append('                     ' + '  |  '.join(ex))
            for t in ['CNB', 'RP', 'TURP']:
                if t in r[tag]:
                    tt = r[tag][t]
                    L.append(f'                     {t:5s} n={tt["n"]:3d}  AUC {tt["auc"]}  sens {tt["sens"]}  spec {tt["spec"]}')
        L.append(f'  per-fold AUC   raw {r["raw_per_fold"]}   macenko {r["macenko_per_fold"]}')
        L.append(f'  per-fold spec  raw {r["per_fold_spec_raw"]}   macenko {r["per_fold_spec_macenko"]}')
        L.append('')
    txt = '\n'.join(L)
    open(f'{OUT}/SUMMARY.txt', 'w').write(txt + '\n')
    print('\n' + txt)
    print(f'saved -> {OUT}/')


if __name__ == '__main__':
    main()
