"""Frozen MVE policy primitives. No entry point for training or evaluation truth.
The stage2 runner must use the frozen config and keep a read audit per fold.
"""
from pathlib import Path
import csv, hashlib, math, io

PRIORITY = ('A', 'B', 'C', 'D')


def stable_seed(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, args)).encode()).digest()[:8], 'little') % (2 ** 32 - 1)


def io_row_id(center, stem):
    return stable_seed('mve-wsi', center, stem)


def indices(pool, k, seed):
    import numpy as np
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) <= k: return pool.copy()
    return pool[np.sort(np.random.default_rng(seed).choice(len(pool), k, replace=False))]


def threshold(scores, labels, sensitivity=1.):
    if len(scores) != len(labels): raise ValueError('length mismatch')
    if not 0 < sensitivity <= 1: raise ValueError('invalid sensitivity')
    if any(not math.isfinite(float(z)) for z in scores): raise ValueError('nonfinite score')
    if any(int(y) not in (0, 1) for y in labels): raise ValueError('invalid label')
    positives = sorted(float(z) for z, y in zip(scores, labels) if int(y) == 1)
    if not positives: return None
    # ceil using integer arithmetic for the two frozen target sensitivities.
    required = (95 * len(positives) + 99) // 100 if sensitivity == .95 else len(positives)
    if sensitivity not in (.95, 1.): raise ValueError('unfrozen sensitivity')
    return positives[len(positives) - required]


def working_point(scores, labels, t):
    if len(scores) != len(labels): raise ValueError('length mismatch')
    if t is None or not math.isfinite(t): raise ValueError('invalid threshold')
    if any(not math.isfinite(float(z)) for z in scores): raise ValueError('nonfinite score')
    if any(int(y) not in (0, 1) for y in labels): raise ValueError('invalid label')
    p = sum(int(y) == 1 for y in labels);
    n = len(labels) - p
    fn = sum(int(y) == 1 and float(z) < t for z, y in zip(scores, labels))
    fp = sum(int(y) == 0 and float(z) >= t for z, y in zip(scores, labels))
    return dict(positive=p, negative=n, FN=fn, FP=fp, sensitivity=(p - fn) / p if p else None, specificity=(n - fp) / n if n else None)


def select(candidates, labels):
    if not labels or set(map(int, labels)) != {0, 1}:
        return dict(action='A', fallback=True, reason='insufficient_class_support')
    acceptable = []
    for a in PRIORITY:
        x = candidates[a]
        if x['threshold'] is None: continue
        w = working_point(x['scores'], labels, x['threshold'])
        if 100 * (w['positive'] - w['FN']) >= 95 * w['positive']:
            acceptable.append((w['FP'], PRIORITY.index(a), a))
    if not acceptable: return dict(action='A', fallback=True, reason='no_candidate_meets_select95')
    return dict(action=min(acceptable)[2], fallback=False, reason='max_specificity_then_A_B_C_D')


def guarded_read(config, slug, kind, *, path=None, repeat=0, audit=None):
    """Exact hashed bytes + exact schema; only primary repeat until stage4.
    This is application-level enforcement, not OS isolation. Phase restrictions
    and selection-lock validation remain the stage2 runner's responsibility.
    """
    if repeat != 0: raise ValueError('repeat1-4 require stage4 authorization')
    if slug not in config['inputs'] or kind not in config['inputs'][slug]:
        raise ValueError('fold/kind not allowed')
    spec = config['inputs'][slug][kind];
    expected = Path(spec['path']).resolve()
    if path is not None and Path(path).resolve() != expected:
        raise ValueError('cross-fold/non-allowlisted path rejected before open')
    raw = expected.read_bytes();
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec['sha256']: raise ValueError('input bytes changed')
    rd = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if rd.fieldnames != spec['columns']: raise ValueError('schema mismatch')
    rows = list(rd);
    center = config['centers'][slug]
    for r in rows:
        if r['outer_target'] != center: raise ValueError('wrong outer fold')
        if kind == 'source':
            if r['center'] == center or r['role'] not in ('source_design', 'source_calibration'):
                raise ValueError('target in source')
        else:
            if int(r['repeat']) != repeat: raise ValueError('wrong repeat')
            roles = ('target_evaluation',) if kind == 'evaluation_wsi' else ('target_fit', 'target_select')
            if r['role'] not in roles: raise ValueError('wrong role')
    if audit is not None: audit.append(dict(path=str(expected), sha256=digest, slug=slug, kind=kind, repeat=repeat, rows=len(rows)))
    return rows
