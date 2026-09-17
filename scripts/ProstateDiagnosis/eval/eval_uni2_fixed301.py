#!/usr/bin/env python
"""Existing uni2 5-fold ensemble on the MAGNIFICATION-FIXED 301 uni2 features.

No retrain - same checkpoints. Just: does re-patching 301 at the right physical
scale (508px @ mpp 0.22 -> ~0.5 um/enc-px, matching training) fix the benign
RP/TURP over-call?

Reference (301, 5-fold ensemble, thr 0.5, benign RP+TURP spec):
  uni2 WRONG scale (224px @ 40x)   spec 0.55   benign RP/TURP 0.28
  OLD h-opt + Reinhard-consistent   spec 0.99   benign RP/TURP 0.986

out -> <RUN>/stainnorm_eval/fixed301_SUMMARY.txt
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

MIL = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
sys.path.insert(0, MIL)
os.chdir(MIL)
from modules.AB_MIL.ab_mil import AB_MIL          # noqa: E402
from utils.process_utils import get_act            # noqa: E402
from utils.yaml_utils import read_yaml             # noqa: E402

RUN = f'{MIL}/result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_3center'
FEAT = '/NAS145/linboliao/Data/迈新生物_特征/Prostate_Diagnosis/MIL外部测试/feat_0_224/pt_files/uni2'
DEVICE = 'cuda:0'
OUT = f'{RUN}/stainnorm_eval'


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
    return ms


@torch.no_grad()
def probs(models, path):
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
            'auprc': round(float(average_precision_score(y, p)), 4) if len(set(y)) > 1 else None,
            'sens': round(float(tp / (tp + fn)), 4) if tp + fn else None,
            'spec': round(float(tn / (tn + fp)), 4) if tn + fp else None,
            'acc': round(float((tp + tn) / len(y)), 4),
            'cm': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def strata(d):
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
        miss = 0
        for _, r in meta.iterrows():
            fp = f'{FEAT}/{r.slide_id}.pt'
            if not os.path.exists(fp):
                miss += 1
                continue
            pp = probs(models, fp)
            rows.append(dict(sid=str(r.slide_id), cohort=c, type=r['type'], label=int(r.label),
                             **{f'f{k+1}': pp[k] for k in range(5)}, ens=float(np.mean(pp))))
        print(f'{c}: {sum(1 for x in rows if x["cohort"]==c)} evaluated, {miss} missing', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/fixed301_slide_preds.csv', index=False, encoding='utf-8-sig')

    res = {}
    L = ['UNI2 5-fold ensemble on MAGNIFICATION-FIXED 301 (508px)  thr 0.5', '=' * 68, '']
    for c in ('301', 'ynzl'):
        sub = df[df.cohort == c]
        if not len(sub):
            continue
        r = strata(sub[['type', 'label']].assign(prob=sub.ens))
        res[c] = {'ensemble': r,
                  'per_fold_spec': [M(sub.label, sub[f'f{k+1}'])['spec'] for k in range(5)],
                  'per_fold_auc': [M(sub.label, sub[f'f{k+1}'])['auc'] for k in range(5)]}
        a = r['ALL']
        L.append(f'### {c}  n={a["n"]}  cancer prev {a["prev"]}')
        L.append(f'  ENSEMBLE  AUC {a["auc"]}  AUPRC {a["auprc"]}  sens {a["sens"]}  spec {a["spec"]}  acc {a["acc"]}  cm {a["cm"]}')
        if 'benign_RPTURP_spec' in r:
            L.append(f'            benign RP+TURP spec {r["benign_RPTURP_spec"]}   |   cancer sens {r.get("cancer_sens")}')
        for t in ['CNB', 'RP', 'TURP']:
            if t in r:
                tt = r[t]
                L.append(f'            {t:5s} n={tt["n"]:3d}  AUC {tt["auc"]}  sens {tt["sens"]}  spec {tt["spec"]}')
        L.append(f'  per-fold spec  {res[c]["per_fold_spec"]}')
        L.append('')
    L += ['--- vs WRONG-scale uni2 (from earlier) ---',
          '  301: AUC 0.984  spec 0.55  benign RP+TURP 0.28  cm [[65,54],[0,28]]',
          '  ynzl: AUC 0.997 spec 0.97',
          '--- vs OLD h-opt+Reinhard-consistent ---',
          '  301: AUC 0.978  spec 0.99  benign RP+TURP 0.986  cm [[118,1],[6,22]]']
    txt = '\n'.join(L)
    open(f'{OUT}/fixed301_SUMMARY.txt', 'w').write(txt + '\n')
    json.dump(res, open(f'{OUT}/fixed301_results.json', 'w'), indent=2, default=float)
    print('\n' + txt)
    print(f'\nsaved -> {OUT}/fixed301_*')


if __name__ == '__main__':
    main()
