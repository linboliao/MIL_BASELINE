"""Lock source thresholds, then compare paired target decisions by domain."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .statistics import (
    evaluate_draw, get_thr, patient_weights, scores, sha, weighted_thr, writecsv,
)


def load(root, spec):
    path = root / spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError(f'Frozen input mismatch: {path}')
    frame = pd.read_csv(path, dtype={'slide_id': str, 'patient_group_id': str})
    if (frame[['slide_id', 'patient_group_id']].isna().any().any()
            or not frame.slide_id.is_unique or set(frame.label) != {0, 1}
            or not frame.prob_1.between(0, 1).all()):
        raise ValueError(f'Invalid score table: {path}')
    return frame.sort_values('slide_id').reset_index(drop=True)


def at_level(frame, level):
    if level == 'slide':
        return frame
    return frame.groupby('patient_group_id', as_index=False).agg(
        label=('label', 'max'), prob_1=('prob_1', 'max'))


def aligned(frames):
    keys = ['slide_id', 'patient_group_id', 'label']
    first = frames[0][keys]
    if any(not f[keys].equals(first) for f in frames[1:]):
        raise ValueError('Frozen membership differs between PFMs')


def analyze(config, output=None):
    root = config.resolve().parents[1]
    cfg = json.loads(config.read_text())
    if cfg['source_sensitivity'] != .95 or cfg['fixed'] != .5 or cfg['levels'] != ['slide', 'patient']:
        raise ValueError('Expected the frozen source95 versus 0.5 protocol')
    n = int(cfg['bootstrap_draws'])
    if n < 1:
        raise ValueError('Bootstrap requires at least one draw')
    cals = [load(root, c['calibration']) for c in cfg['cells']]
    locks = []
    for cell, cal in zip(cfg['cells'], cals):
        for level in cfg['levels']:
            table = at_level(cal, level)
            y, p = table.label.to_numpy(), table.prob_1.to_numpy()
            threshold = get_thr(y, p)
            locks.append(dict(model=cell['model'], domain=cell['domain'], level=level,
                              threshold=threshold, calibration_sha256=cell['calibration']['sha256'],
                              checkpoint_sha256=cell['checkpoint_sha256'],
                              calibration_positive_groups=cal.loc[cal.label == 1, 'patient_group_id'].nunique(),
                              achieved_source_sensitivity=float((p[y == 1] >= threshold).mean())))
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        writecsv(output/'threshold_lock.csv', locks)
    targets = [load(root, c['target']) for c in cfg['cells']]
    results = []
    for domain in sorted({c['domain'] for c in cfg['cells']}):
        ids = [i for i,c in enumerate(cfg['cells']) if c['domain'] == domain]
        aligned([cals[i] for i in ids])
        aligned([targets[i] for i in ids])
        if set(cals[ids[0]].patient_group_id) & set(targets[ids[0]].patient_group_id):
            raise ValueError('Calibration and target overlap')
        if output is None:
            continue
        for level in cfg['levels']:
            cal_ref, test_ref = (at_level(f[ids[0]], level) for f in (cals, targets))
            # Identical group draws across models and measurement levels.
            seeds = np.random.SeedSequence([cfg['seed'], sum(map(ord, domain))]).spawn(2)
            cn, cg, cw, _ = patient_weights(cal_ref, np.random.default_rng(seeds[0]), n)
            tn, tg, tw, _ = patient_weights(test_ref, np.random.default_rng(seeds[1]), n)
            with (output/f'{domain}_{level}_groups.npz').open('xb') as f:
                np.savez_compressed(f, calibration_ids=cn.astype(str), calibration_weights=cw,
                                    target_ids=tn.astype(str), target_weights=tw)
            for i in ids:
                cell = cfg['cells'][i]
                cal, target = (at_level(f[i], level) for f in (cals, targets))
                y, p = target.label.to_numpy(), target.prob_1.to_numpy()
                t = next(r['threshold'] for r in locks if
                         (r['model'],r['domain'],r['level']) == (cell['model'],domain,level))
                fixed, locked = scores(y,p,.5), scores(y,p,t)
                row = dict(model=cell['model'], domain=domain, level=level, threshold=t,
                           auroc=roc_auc_score(y,p),
                           new_fn=int(((y==1)&(p>=.5)&(p<t)).sum()),
                           recovered_fn=int(((y==1)&(p<.5)&(p>=t)).sum()))
                for name, point in [('fixed',fixed),('source95',locked)]:
                    row.update({f'{name}_{k}':v for k,v in point.items()})
                td = weighted_thr(cal,cal.prob_1.to_numpy(),cw,cg)
                fd = evaluate_draw(y,p,np.full(n,.5),tw,tg)
                ld = evaluate_draw(y,p,td,tw,tg)
                row.update(threshold_ci_low=float(np.quantile(td,.025)),
                           threshold_ci_high=float(np.quantile(td,.975)), valid_draws=len(td))
                for metric in ('sensitivity','specificity','bacc'):
                    lo,hi = np.quantile(ld[metric]-fd[metric],[.025,.975])
                    abs_lo,abs_hi = np.quantile(ld[metric],[.025,.975])
                    row.update({f'delta_{metric}_ci_low':lo,f'delta_{metric}_ci_high':hi,
                                f'source95_{metric}_ci_low':abs_lo,f'source95_{metric}_ci_high':abs_hi})
                with (output/f'{domain}_{level}_{cell["model"]}_draws.npz').open('xb') as f:
                    np.savez_compressed(f, thresholds=td, **{f'fixed_{k}':v for k,v in fd.items()},
                                        **{f'source95_{k}':v for k,v in ld.items()})
                results.append(row)
    if output is not None:
        for domain in sorted({r['domain'] for r in results}):
            writecsv(output/f'{domain}_transport.csv',[r for r in results if r['domain']==domain])
        with (output/'completion.json').open('x') as f:
            json.dump(dict(status='completed',config_sha256=sha(config),n_results=len(results)),f)
    print(f'Validated {len(cfg["cells"])} cells, {len(locks)} thresholds, execution={output is not None}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
        provenance = args.config.resolve().parents[2]/'provenance'
        if provenance not in args.output.parents:
            parser.error('Output must be under project provenance')
    analyze(args.config,args.output)


if __name__ == '__main__':
    main()
