"""Selective review with label-free fractional allocation at score ties."""
import numpy as np


def review_weights(priority, budget):
    if budget == 0:
        return np.zeros(len(priority))
    cutoff = np.sort(priority)[-budget]
    above = priority > cutoff
    tied = priority == cutoff
    return above.astype(float) + tied * ((budget - above.sum()) / tied.sum())


def review_metrics(label, decision, weights):
    error = label != decision
    kept = 1 - weights
    return dict(coverage=float(kept.mean()), kept_error=float(np.average(error, weights=kept)),
                reviewed_fn=float(weights[error & (label == 1)].sum()),
                reviewed_fp=float(weights[error & (label == 0)].sum()))
