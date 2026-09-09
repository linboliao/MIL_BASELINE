"""Flatten the redundant per-fold result nesting that the OLD loco / psir_recheck
config layout produced, and re-merge k-fold metrics.

OLD (one train_mil.py per fold, each sees a 1-CSV dir -> its own fold_1 + own timestamp):
  AB_MIL_<model>_loco_<mode>/
    fold_1/ProstateDiagnosis_<model>_loco_<mode>_fold1/AB_MIL/seed_42_<ts>/fold_1/<files>
    fold_2/ProstateDiagnosis_<model>_loco_<mode>_fold2/AB_MIL/seed_42_<ts>/fold_1/<files>
    ...

NEW (what the fixed loco_gen_configs.py + one train_mil.py call give natively):
  AB_MIL_<model>_loco_<mode>/
    AB_MIL/seed_42_<ts0>/fold_1/<files>
    AB_MIL/seed_42_<ts0>/fold_2/<files>
    AB_MIL/seed_42_<ts0>/merge_<N>_fold_metrics.json

Run AFTER your current experiments finish.  Dry-run by default; --apply to move.
Idempotent: an exp dir that's already flat is skipped.

  python relayout.py                       # dry-run, both servers' local result root
  python relayout.py --apply
  python relayout.py --apply --rm-empty    # also delete the emptied fold_<k>/ shells
  python relayout.py --root <dir> --glob 'AB_MIL_*_loco_*' --apply
"""
import argparse
import glob
import json
import os
import re
import shutil
import statistics as st
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = str(_REPO / "result" / "ProstateDiagnosis" / "DataAnalysis")
FOLD_RE = re.compile(r"^fold_(\d+)$")
SEEDDIR_RE = re.compile(r"seed_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})$")

TEST_METRICS = ['acc', 'bacc', 'macro_auc', 'micro_auc', 'weighted_auc',
                'macro_f1', 'micro_f1', 'weighted_f1',
                'macro_recall', 'micro_recall', 'weighted_recall',
                'macro_pre', 'micro_pre', 'weighted_pre',
                'quadratic_kappa', 'linear_kappa']


def find_leaf(fold_shell):
    """fold_<k>/<DATASET_NAME>/AB_MIL/seed_<s>_<ts>/fold_1  -> (leafdir, seed, ts, seeddir)."""
    hits = glob.glob(f"{fold_shell}/*/AB_MIL/seed_*/fold_1")
    hits = [h for h in hits if os.path.isdir(h)]
    if not hits:
        return None
    leaf = max(hits, key=os.path.getmtime)                 # newest if re-run
    seeddir = os.path.dirname(leaf)                        # .../seed_<s>_<ts>
    m = SEEDDIR_RE.search(os.path.basename(seeddir))
    seed, ts = (m.group(1), m.group(2)) if m else ("42", "0000-00-00-00-00")
    return leaf, seed, ts, seeddir


def merge_k_fold(seed_root):
    """Re-implement merge_k_fold_logs, tolerant of folds with no Best*.csv."""
    import pandas as pd
    agg = {k: [] for k in TEST_METRICS}
    folds = sorted(d for d in os.listdir(seed_root)
                   if FOLD_RE.match(d) and os.path.isdir(os.path.join(seed_root, d)))
    used = 0
    for fd in folds:
        best = glob.glob(os.path.join(seed_root, fd, "Best*.csv"))
        if not best:
            continue
        row = pd.read_csv(best[0]).iloc[0]
        cols = {c.replace("test_", ""): c for c in row.index if c.startswith("test_")}
        if not all(k in cols for k in TEST_METRICS):
            continue
        for k in TEST_METRICS:
            agg[k].append(float(row[cols[k]]))
        used += 1
    if not used:
        return None
    out = {k: {"mean": st.mean(v), "std": st.pstdev(v)} for k, v in agg.items() if v}
    p = os.path.join(seed_root, f"merge_{used}_fold_metrics.json")
    json.dump(out, open(p, "w"))
    return p


def process_expdir(expdir, apply, rm_empty):
    name = os.path.basename(expdir)
    shells = sorted((d for d in os.listdir(expdir) if FOLD_RE.match(d)),
                    key=lambda d: int(FOLD_RE.match(d).group(1)))
    if not shells:
        return f"  {name}: already flat / no fold_<k> shells — skip"

    plan = []      # (k, leafdir, seed, ts)
    for sh in shells:
        r = find_leaf(os.path.join(expdir, sh))
        if r is None:
            plan.append((int(FOLD_RE.match(sh).group(1)), None, None, None))
            continue
        leaf, seed, ts, _ = r
        plan.append((int(FOLD_RE.match(sh).group(1)), leaf, seed, ts))

    have = [p for p in plan if p[1]]
    if not have:
        return f"  {name}: {len(shells)} shells but no seed_/fold_1 leaves inside — skip"
    seed = have[0][2]
    ts0 = min(p[3] for p in have)                          # shared timestamp = earliest fold
    dst_seed = os.path.join(expdir, "AB_MIL", f"seed_{seed}_{ts0}")

    lines = [f"  {name}:  {len(have)}/{len(shells)} folds -> {os.path.relpath(dst_seed, expdir)}/fold_<k>"]
    for k, leaf, _, ts in plan:
        if leaf is None:
            lines.append(f"      fold_{k}: (no leaf — skipped)")
            continue
        dst = os.path.join(dst_seed, f"fold_{k}")
        lines.append(f"      fold_{k}: {os.path.relpath(leaf, expdir)}"
                     f"{'  [ts '+ts+']' if ts != ts0 else ''}  ->  {os.path.relpath(dst, expdir)}")
        if apply:
            if os.path.exists(dst):
                lines[-1] += "   (dst exists, skip)"
                continue
            os.makedirs(dst_seed, exist_ok=True)
            shutil.move(leaf, dst)
    if apply:
        mp = merge_k_fold(dst_seed)
        lines.append(f"      merged -> {os.path.relpath(mp, expdir) if mp else '(no complete fold)'}")
        if rm_empty:
            for sh in shells:
                shp = os.path.join(expdir, sh)
                try:
                    shutil.rmtree(shp)
                    lines.append(f"      rm {sh}/")
                except OSError as e:
                    lines.append(f"      rm {sh}/ FAILED: {e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--glob", default="AB_MIL_*_loco_*",
                    help="exp-dir glob under --root (also try 'AB_MIL_*_recheck_*')")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rm-empty", action="store_true", help="delete emptied fold_<k>/ shells (implies --apply intent)")
    a = ap.parse_args()

    globs = a.glob.split(",") if "," in a.glob else [a.glob]
    exps = sorted({d for g in globs for d in glob.glob(os.path.join(a.root, g)) if os.path.isdir(d)})
    if not exps:
        raise SystemExit(f"no exp dirs match {globs} under {a.root}")
    print(f"{'APPLY' if a.apply else 'DRY-RUN'}  root={a.root}  ({len(exps)} exp dirs)\n")
    for e in exps:
        print(process_expdir(e, a.apply, a.rm_empty))
    if not a.apply:
        print("\n(dry-run — re-run with --apply to move; add --rm-empty to also delete the old fold_<k>/ shells)")


if __name__ == "__main__":
    main()
