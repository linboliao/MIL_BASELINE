"""Patient-level LOCO decision reliability from frozen scores."""
import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.threshold_transport.analyze import at_level, load
from experiments.threshold_transport.statistics import get_thr
from .metrics import review_metrics, review_weights


def analyze_cell(root, cell, coverage, draws, seed):
    calibration = at_level(load(root, cell['calibration']), 'patient')
    threshold = get_thr(calibration.label.to_numpy(), calibration.prob_1.to_numpy())
    target = at_level(load(root, cell['target']), 'patient')
    if set(calibration.patient_group_id) & set(target.patient_group_id):
        raise ValueError('Calibration and target overlap')
    y, p = target.label.to_numpy(), target.prob_1.to_numpy()
    decision = p >= threshold
    margin = -np.abs(p - threshold)
    budget = len(y) - int(np.ceil(coverage * len(y)))
    error = decision != y
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == k) for k in (0, 1)]
    estimates = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
        m = review_metrics(y[ix], decision[ix], review_weights(margin[ix], budget))
        estimates.append(m['kept_error'] - error[ix].mean())
    fixed_budget = review_metrics(y, decision, review_weights(margin, budget))
    source_margin = np.abs(calibration.prob_1.to_numpy() - threshold)
    source_cutoff = float(np.quantile(source_margin, 1 - coverage))
    source_locked = review_metrics(y, decision, (np.abs(p-threshold) < source_cutoff).astype(float)) if (np.abs(p-threshold) >= source_cutoff).any() else dict(coverage=0.,kept_error=None,reviewed_fn=int((error&(y==1)).sum()),reviewed_fp=int((error&(y==0)).sum()))
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(model=cell['model'], domain=cell['domain'], threshold=threshold,
                patients=len(y), baseline_error=float(error.mean()),
                error_auroc=float(roc_auc_score(error, margin)) if len(set(error)) == 2 else None,
                random_expected_kept_error=float(error.mean()),
                random_expected_reviewed_fn=float((error&(y==1)).sum()*budget/len(y)),
                random_expected_reviewed_fp=float((error&(y==0)).sum()*budget/len(y)),
                delta_vs_random_ci_low=float(lo), delta_vs_random_ci_high=float(hi),
                source_review_margin=source_cutoff, **fixed_budget,
                **{f'source_locked_{k}':v for k,v in source_locked.items()})
