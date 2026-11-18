"""Freeze C0/C1 source95 thresholds and evaluate existing slide scores."""
import argparse
import math
from pathlib import Path

from .artifacts import json_x, read, sha


def scores(path):
    frame = read(path)
    frame['prob_1'] = frame.prob_1.astype(float)
    if frame.empty or set(frame.label) - {'0', '1'} or not frame.prob_1.between(0, 1).all():
        raise ValueError('Expected finite binary slide scores')
    if not frame.leakage_group.str.strip().all():
        raise ValueError('Missing case group')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c0', required=True)
    parser.add_argument('--c1', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    calibrations = {key: scores(path) for key, path in [('C0', args.c0), ('C1', args.c1)]}
    thresholds = {}
    for key, data in calibrations.items():
        positive = sorted(data.loc[data.label == '1', 'prob_1'], reverse=True)
        thresholds[key] = positive[math.ceil(.95 * len(positive)) - 1] if positive else None
    args.output.mkdir(parents=True, exist_ok=False)
    json_x(args.output / 'threshold_lock.json',
           dict(thresholds=thresholds, source95=.95, c0_sha256=sha(args.c0), c1_sha256=sha(args.c1)))
    test = scores(args.test)
    if any(set(frame.leakage_group) & set(test.leakage_group) for frame in calibrations.values()):
        raise ValueError('Calibration and test overlap by case group')
    results = {}
    for key, threshold in thresholds.items():
        if threshold is None:
            results[key] = None
            continue
        positive = test.label == '1'
        calls = test.prob_1 >= threshold
        sens = float(calls[positive].mean()) if positive.any() else None
        spec = float((~calls[~positive]).mean()) if (~positive).any() else None
        results[key] = dict(sensitivity=sens, specificity=spec,
                            balanced_accuracy=(sens + spec) / 2 if None not in (sens, spec) else None)
    json_x(args.output / 'metrics.json', dict(test_sha256=sha(args.test), metrics=results))


if __name__ == '__main__':
    main()
