#!/usr/bin/env python
"""Group-A analyses for the virchow2 5-fold run (prediction-CSV only).

  1. Bootstrap 95% CIs (patient-clustered) for every per-stratum metric
  2. Calibration: ECE / Brier + reliability curves per center;
     temperature & Platt scaling fit on internal test, applied to 301 / ynzl;
     plus a 301-specific temperature fit on half of 301, tested on the other half
  3. Operating points: spec at sens>=0.95 / >=0.99, and Youden, per center
  4. AUPRC + PR-curve data per center
  5. Patient-level aggregation (internal): max / mean slide prob per patient

Outputs -> <run_dir>/groupA_analysis/
  bootstrap_ci.csv, calibration.json, calibration_curves.csv,
  operating_points.csv, pr_curves.csv, patient_level.csv, SUMMARY.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, confusion_matrix)

REPO = '/NAS2/Data1/lbliao/Code-195/MIL_BASELINE'
RES_ROOT = os.path.join(REPO, 'result/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center_fp16local')
INT_META = os.path.join(REPO, 'datasets/ProstateDiagnosis/internal_test.csv')
RNG = np.random.default_rng(42)
NBOOT = 2000


def point_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    yh = (p >= thr).astype(int)
    out = {'n': len(y), 'n_pos': int(y.sum()), 'prev': float(y.mean())}
    if len(np.unique(y)) == 2:
        out['auc'] = roc_auc_score(y, p)
        out['auprc'] = average_precision_score(y, p)
        out['brier'] = brier_score_loss(y, p)
    else:
        out['auc'] = out['auprc'] = out['brier'] = np.nan
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    out['sens'] = tp / (tp + fn) if tp + fn else np.nan
    out['spec'] = tn / (tn + fp) if tn + fp else np.nan
    out['ppv'] = tp / (tp + fp) if tp + fp else np.nan
    out['npv'] = tn / (tn + fn) if tn + fn else np.nan
    out['acc'] = (tp + tn) / len(y)
    out['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return out


def bootstrap_ci(df, thr=0.5):
    """df has columns: y, p, patient. Resample patients with replacement."""
    pts = df['patient'].unique()
    pt_to_idx = {q: df.index[df['patient'] == q].to_numpy() for q in pts}
    keys = ['auc', 'auprc', 'brier', 'sens', 'spec', 'ppv', 'npv', 'acc', 'f1']
    samples = {k: [] for k in keys}
    for _ in range(NBOOT):
        take = RNG.choice(pts, size=len(pts), replace=True)
        idx = np.concatenate([pt_to_idx[q] for q in take])
        d = df.loc[idx]
        if d['y'].nunique() < 2:
            continue
        m = point_metrics(d['y'].values, d['p'].values, thr)
        for k in keys:
            v = m[k]
            if v == v:  # not nan
                samples[k].append(v)
    ci = {}
    pt = point_metrics(df['y'].values, df['p'].values, thr)
    for k in keys:
        s = np.array(samples[k])
        ci[k] = {'point': float(pt[k]) if pt[k] == pt[k] else None,
                 'lo': float(np.percentile(s, 2.5)) if len(s) else None,
                 'hi': float(np.percentile(s, 97.5)) if len(s) else None,
                 'n_boot': len(s)}
    ci['_n'] = int(pt['n'])
    ci['_prev'] = float(pt['prev'])
    return ci


def fit_temperature(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    y = np.asarray(y, float)

    def nll(logT):
        T = np.exp(logT)
        q = expit(z / T)
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize_scalar(nll, bounds=(-3, 3), method='bounded')
    return float(np.exp(r.x))


def apply_temperature(p, T):
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / T)


def fit_platt(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs')
    lr.fit(z, np.asarray(y).astype(int))
    return lr


def apply_platt(p, lr):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def ece(y, p, nbins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    bins = np.linspace(0, 1, nbins + 1)
    e = 0.0
    rows = []
    for i in range(nbins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < nbins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            rows.append({'bin_lo': bins[i], 'bin_hi': bins[i + 1], 'n': 0,
                         'mean_pred': None, 'frac_pos': None})
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(y)) * abs(conf - acc)
        rows.append({'bin_lo': float(bins[i]), 'bin_hi': float(bins[i + 1]),
                     'n': int(m.sum()), 'mean_pred': float(conf), 'frac_pos': float(acc)})
    return float(e), rows


def operating_points(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {}
    fpr, tpr, thr = roc_curve(y, p)
    out = {}
    j = np.argmax(tpr - fpr)
    out['youden'] = {'thr': float(thr[j]), 'sens': float(tpr[j]), 'spec': float(1 - fpr[j])}
    for target in (0.95, 0.99):
        ok = np.where(tpr >= target)[0]
        if len(ok):
            k = ok[np.argmin(fpr[ok])]
            out[f'sens>={target}'] = {'thr': float(thr[k]), 'sens': float(tpr[k]), 'spec': float(1 - fpr[k])}
    return out


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.join(RES_ROOT, x) for x in os.listdir(RES_ROOT) if x.startswith('run_'))[-1]
    out = os.path.join(run_dir, 'groupA_analysis')
    os.makedirs(out, exist_ok=True)

    intp = pd.read_csv(os.path.join(run_dir, 'internal_results', 'slide_predictions.csv'))
    intp['slide_id'] = intp['slide_id'].astype(str)
    meta = pd.read_csv(INT_META)[['slide_id', 'patient_id']].astype(str)
    intp = intp.merge(meta, on='slide_id', how='left')
    intp['patient_id'] = intp['patient_id'].fillna(intp['slide_id'])
    intp['cohort'] = 'internal'

    extp = pd.read_csv(os.path.join(run_dir, 'external_results', 'slide_predictions.csv'))
    extp['slide_id'] = extp['slide_id'].astype(str)
    if 'patient_id' not in extp:
        extp['patient_id'] = extp['slide_id']
    extp['patient_id'] = extp['patient_id'].astype(str)

    strata = []  # (name, dataframe with y,p,patient)
    def add(name, d):
        strata.append((name, pd.DataFrame({'y': d['label'].values,
                                           'p': d['prob_ensemble'].values,
                                           'patient': d['patient_id'].values}).reset_index(drop=True)))
    add('internal_ALL', intp)
    for c, g in intp.groupby('center'):
        add(f'internal_{c}', g)
    for t, g in intp.groupby('type'):
        add(f'internal_type_{t}', g)
    for coh in ['301', 'ynzl']:
        g = extp[extp.cohort == coh]
        add(f'external_{coh}', g)
        for t, gg in g.groupby('type'):
            add(f'external_{coh}_type_{t}', gg)

    # ---- 1. bootstrap CIs ----
    ci_rows = []
    ci_json = {}
    for name, d in strata:
        ci = bootstrap_ci(d)
        ci_json[name] = ci
        for k in ['auc', 'auprc', 'brier', 'sens', 'spec', 'ppv', 'npv', 'acc', 'f1']:
            ci_rows.append({'stratum': name, 'n': ci['_n'], 'prev': round(ci['_prev'], 3),
                            'metric': k, 'point': ci[k]['point'],
                            'ci_lo': ci[k]['lo'], 'ci_hi': ci[k]['hi']})
    pd.DataFrame(ci_rows).to_csv(os.path.join(out, 'bootstrap_ci.csv'), index=False)

    # ---- 2. calibration ----
    yi, pi = intp['label'].values, intp['prob_ensemble'].values
    T = fit_temperature(yi, pi)
    platt = fit_platt(yi, pi)
    cal = {'temperature_fit_on_internal': T,
           'platt_fit_on_internal': {'coef': float(platt.coef_[0][0]), 'intercept': float(platt.intercept_[0])}}
    curve_rows = []
    for name, d in [('internal', intp)] + [(coh, extp[extp.cohort == coh]) for coh in ['301', 'ynzl']]:
        y = d['label'].values
        p_raw = d['prob_ensemble'].values
        p_temp = apply_temperature(p_raw, T)
        p_platt = apply_platt(p_raw, platt)
        entry = {}
        for tag, p in [('raw', p_raw), ('temp_scaled', p_temp), ('platt_scaled', p_platt)]:
            e, rows = ece(y, p, 10)
            m = point_metrics(y, p)
            entry[tag] = {'ece': e, 'brier': m['brier'], 'auc': m['auc'],
                          'sens@0.5': m['sens'], 'spec@0.5': m['spec'], 'acc@0.5': m['acc']}
            for r in rows:
                curve_rows.append({'cohort': name, 'calib': tag, **r})
        cal[name] = entry

    # 301-specific temperature: fit on half of 301, test on other half
    d301 = extp[extp.cohort == '301'].reset_index(drop=True)
    idx = RNG.permutation(len(d301))
    h = len(idx) // 2
    fitI, testI = idx[:h], idx[h:]
    T301 = fit_temperature(d301.loc[fitI, 'label'].values, d301.loc[fitI, 'prob_ensemble'].values)
    platt301 = fit_platt(d301.loc[fitI, 'label'].values, d301.loc[fitI, 'prob_ensemble'].values)
    dt = d301.loc[testI]
    m_raw = point_metrics(dt['label'].values, dt['prob_ensemble'].values)
    m_T = point_metrics(dt['label'].values, apply_temperature(dt['prob_ensemble'].values, T301))
    m_P = point_metrics(dt['label'].values, apply_platt(dt['prob_ensemble'].values, platt301))
    cal['301_specific_recalibration'] = {
        'note': 'fit on random 50% of 301, evaluated on the held-out 50%',
        'T_301': T301, 'n_test': len(testI),
        'raw':   {k: m_raw[k] for k in ['auc', 'sens', 'spec', 'acc']},
        'temp':  {k: m_T[k] for k in ['auc', 'sens', 'spec', 'acc']},
        'platt': {k: m_P[k] for k in ['auc', 'sens', 'spec', 'acc']},
    }
    json.dump(cal, open(os.path.join(out, 'calibration.json'), 'w'), indent=2, default=float)
    pd.DataFrame(curve_rows).to_csv(os.path.join(out, 'calibration_curves.csv'), index=False)

    # ---- 3. operating points ----
    op_rows = []
    for name, d in strata:
        if d['y'].nunique() < 2:
            continue
        for op, v in operating_points(d['y'].values, d['p'].values).items():
            op_rows.append({'stratum': name, 'operating_point': op, **v})
    pd.DataFrame(op_rows).to_csv(os.path.join(out, 'operating_points.csv'), index=False)

    # ---- 4. PR curves ----
    pr_rows = []
    for name, d in strata:
        if d['y'].nunique() < 2:
            continue
        prec, rec, thr = precision_recall_curve(d['y'].values, d['p'].values)
        for i in range(0, len(prec), max(1, len(prec) // 200)):
            pr_rows.append({'stratum': name, 'recall': float(rec[i]), 'precision': float(prec[i]),
                            'threshold': float(thr[i]) if i < len(thr) else None})
    pd.DataFrame(pr_rows).to_csv(os.path.join(out, 'pr_curves.csv'), index=False)

    # ---- 5. patient-level (internal) ----
    pl_rows = []
    pat = intp.groupby('patient_id').agg(
        label=('label', 'max'),
        prob_max=('prob_ensemble', 'max'),
        prob_mean=('prob_ensemble', 'mean'),
        n_slides=('slide_id', 'count'),
        center=('center', 'first')).reset_index()
    pat.to_csv(os.path.join(out, 'patient_level.csv'), index=False)
    pl = {}
    for agg in ['prob_max', 'prob_mean']:
        df = pd.DataFrame({'y': pat['label'], 'p': pat[agg], 'patient': pat['patient_id']})
        pl[agg] = bootstrap_ci(df)
    json.dump({'n_patients': len(pat), 'multi_slide': int((pat.n_slides > 1).sum()), 'ci': pl},
              open(os.path.join(out, 'patient_level_ci.json'), 'w'), indent=2, default=float)

    # ---- readable ----
    def fmt(c):
        if c.get('point') is None:
            return "n/a"
        if c.get('lo') is None or c.get('hi') is None:
            return f"{c['point']:.3f} (CI n/a)"
        return f"{c['point']:.3f} ({c['lo']:.3f}-{c['hi']:.3f})"
    L = [f"GROUP A - run {os.path.basename(run_dir)}", "=" * 78, "",
         "1. BOOTSTRAP 95% CI (patient-clustered, 2000 resamples)  [point (lo-hi)]", "-" * 78]
    hdr = f"{'stratum':26s} {'n':>4s} {'AUC':>20s} {'AUPRC':>20s} {'sens@.5':>20s} {'spec@.5':>20s}"
    L.append(hdr)
    for name, _ in strata:
        c = ci_json[name]
        L.append(f"{name:26s} {c['_n']:>4d} {fmt(c['auc']):>20s} {fmt(c['auprc']):>20s} "
                 f"{fmt(c['sens']):>20s} {fmt(c['spec']):>20s}")
    L += ["", "2. CALIBRATION  (ECE / Brier ; effect at thr 0.5)", "-" * 78,
          f"  global temperature fit on internal: T = {T:.3f}   (T>1 => over-confident, needs cooling)"]
    for coh in ['internal', '301', 'ynzl']:
        e = cal[coh]
        L.append(f"  {coh}:")
        for tag in ['raw', 'temp_scaled', 'platt_scaled']:
            v = e[tag]
            L.append(f"     {tag:13s} ECE {v['ece']:.3f}  Brier {v['brier']:.3f}  "
                     f"AUC {v['auc']:.3f}  sens {v['sens@0.5']:.3f}  spec {v['spec@0.5']:.3f}  acc {v['acc@0.5']:.3f}")
    r = cal['301_specific_recalibration']
    L += ["", f"  301-SPECIFIC recalibration ({r['note']}, n_test={r['n_test']}):",
          f"     raw    AUC {r['raw']['auc']:.3f}  sens {r['raw']['sens']:.3f}  spec {r['raw']['spec']:.3f}  acc {r['raw']['acc']:.3f}",
          f"     temp   AUC {r['temp']['auc']:.3f}  sens {r['temp']['sens']:.3f}  spec {r['temp']['spec']:.3f}  acc {r['temp']['acc']:.3f}   (T_301={r['T_301']:.3f})",
          f"     platt  AUC {r['platt']['auc']:.3f}  sens {r['platt']['sens']:.3f}  spec {r['platt']['spec']:.3f}  acc {r['platt']['acc']:.3f}"]
    L += ["", "3. OPERATING POINTS (threshold found on each stratum's own data - optimistic)", "-" * 78]
    for row in op_rows:
        L.append(f"  {row['stratum']:26s} {row['operating_point']:12s} thr {row['thr']:.3f}  "
                 f"sens {row['sens']:.3f}  spec {row['spec']:.3f}")
    L += ["", "4. AUPRC in table 1 above (baseline = prevalence). PR curves -> pr_curves.csv", ""]
    L += ["5. PATIENT-LEVEL (internal)", "-" * 78,
          f"  {len(pat)} patients ({int((pat.n_slides > 1).sum())} multi-slide, max {pat.n_slides.max()} slides)"]
    for agg in ['prob_max', 'prob_mean']:
        c = pl[agg]
        L.append(f"  aggregate={agg:9s}  AUC {fmt(c['auc'])}  sens {fmt(c['sens'])}  spec {fmt(c['spec'])}  acc {fmt(c['acc'])}")
    txt = "\n".join(L)
    open(os.path.join(out, 'SUMMARY.txt'), 'w').write(txt + "\n")
    print(txt)
    print(f"\nsaved -> {out}/")


if __name__ == '__main__':
    main()
