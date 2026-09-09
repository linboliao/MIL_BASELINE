"""Flatten the redundant per-fold result nesting under
result/ProstateDiagnosis/DataAnalysis/ and re-merge k-fold metrics.

Two OLD shell patterns are handled (one train_mil.py invocation per fold/split,
each getting its own DATASET_NAME dir + its own inner fold_1 + its own timestamp):

  <exp>/fold_<k>/<DATASET_NAME>/AB_MIL/seed_<s>_<ts>/fold_1/<files>   (k-fold: loco, 5fold_3center, recheck, psir_fold*)
  <exp>/held_<X>/<DATASET_NAME>/AB_MIL/seed_<s>_<ts>/<files>          (leave-one-out: held_<center/type>)

NEW (also what the fixed loco_gen_configs.py + one train_mil.py call give natively):

  <exp>/AB_MIL/seed_<s>_<ts0>/fold_<k>/<files>   (+ merge_<N>_fold_metrics.json)
  <exp>/AB_MIL/seed_<s>_<ts0>/held_<X>/<files>

<ts0> = the earliest timestamp among the exp's shells (shared).

Dirs with neither shell (mag_scan, patch_mpp_audit, *_fp16local's run_<ts>/ ...) are skipped.
Already-flat exp dirs are skipped.

  python relayout.py                              # DRY-RUN, glob AB_MIL_* under the repo's result root
  python relayout.py --apply
  python relayout.py --apply --rm-empty           # also remove emptied shell dirs (keeps non-empty ones, e.g. fold_<k>/external_*)
  python relayout.py --apply --rm-empty --move-siblings   # + relocate fold_<k>/external_* etc into the flat fold dir
  python relayout.py --root <dir> --glob '*'
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
HELD_RE = re.compile(r"^held_(.+)$")
SEEDDIR_RE = re.compile(r"^seed_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})$")

TEST_METRICS = ['acc', 'bacc', 'macro_auc', 'micro_auc', 'weighted_auc',
                'macro_f1', 'micro_f1', 'weighted_f1',
                'macro_recall', 'micro_recall', 'weighted_recall',
                'macro_pre', 'micro_pre', 'weighted_pre',
                'quadratic_kappa', 'linear_kappa']


def _seed_ts(seeddir):
    m = SEEDDIR_RE.match(os.path.basename(seeddir))
    return (m.group(1), m.group(2)) if m else ("42", "0000-00-00-00-00")


def find_leaf(shell, kind):
    """Return (src_to_move, seed, ts).  kind='fold' -> the inner fold_<m> dir;
    kind='held' -> the seed_<s>_<ts> dir itself."""
    if kind == "fold":
        hits = [h for h in glob.glob(f"{shell}/*/AB_MIL/seed_*/fold_*") if os.path.isdir(h)]
        if not hits:
            return None
        src = max(hits, key=os.path.getmtime)
        seed, ts = _seed_ts(os.path.dirname(src))
        return src, seed, ts
    hits = [h for h in glob.glob(f"{shell}/*/AB_MIL/seed_*") if os.path.isdir(h)]
    if not hits:
        return None
    src = max(hits, key=os.path.getmtime)
    seed, ts = _seed_ts(src)
    return src, seed, ts


def prune_empty(path, stop_at):
    """Remove `path` and its now-empty ancestors, walking up, stopping before stop_at
    or at the first non-empty dir. Deletes stray per-fold merge_*.json (regenerated).
    Returns list of removed dirs."""
    removed = []
    p = path
    while os.path.abspath(p) != os.path.abspath(stop_at):
        for j in glob.glob(os.path.join(p, "merge_*_fold_metrics.json")):
            os.remove(j)
        try:
            os.rmdir(p)          # only succeeds if empty
            removed.append(p)
            p = os.path.dirname(p)
        except OSError:
            break
    return removed


def merge_folds(seed_root):
    import pandas as pd
    agg = {k: [] for k in TEST_METRICS}
    subs = sorted(d for d in os.listdir(seed_root)
                  if (FOLD_RE.match(d) or HELD_RE.match(d))
                  and os.path.isdir(os.path.join(seed_root, d)))
    used = 0
    for sd in subs:
        best = glob.glob(os.path.join(seed_root, sd, "Best*.csv"))
        if not best:
            continue
        row = pd.read_csv(best[0]).iloc[0]
        cols = {c[len("test_"):]: c for c in row.index if c.startswith("test_")}
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


def process_exp(exp, apply, rm_empty, move_siblings):
    name = os.path.basename(exp)
    subs = os.listdir(exp)
    shells = []  # (shell_name, kind, tag)   tag = "fold_<k>" or "held_<X>"
    for s in subs:
        if not os.path.isdir(os.path.join(exp, s)):
            continue
        mf, mh = FOLD_RE.match(s), HELD_RE.match(s)
        if mf:
            shells.append((s, "fold", s))
        elif mh:
            shells.append((s, "held", s))
    if not shells:
        flat = glob.glob(f"{exp}/AB_MIL/seed_*/fold_*") + glob.glob(f"{exp}/AB_MIL/seed_*/held_*")
        return f"  {name}: {'already flat' if flat else 'no fold_<k>/held_<X> shells'} — skip"

    plan = []
    for sh, kind, tag in shells:
        r = find_leaf(os.path.join(exp, sh), kind)
        plan.append((sh, kind, tag, *(r if r else (None, None, None))))

    have = [p for p in plan if p[3]]
    if not have:
        return f"  {name}: {len(shells)} shells, no seed_ leaves inside — skip"
    seed = have[0][4]
    ts0 = min(p[5] for p in have)
    dst_seed = os.path.join(exp, "AB_MIL", f"seed_{seed}_{ts0}")

    lines = [f"  {name}:  {len(have)}/{len(shells)} splits -> AB_MIL/seed_{seed}_{ts0}/<{'fold'if have[0][1]=='fold' else 'held'}_*>"]
    for sh, kind, tag, src, sd, ts in plan:
        if src is None:
            lines.append(f"      {sh}: (no leaf) skip")
            continue
        dst = os.path.join(dst_seed, tag)
        note = f"  [ts {ts}]" if ts != ts0 else ""
        lines.append(f"      {os.path.relpath(src, exp)}{note}  ->  {os.path.relpath(dst, exp)}")
        if not apply:
            continue
        if os.path.exists(dst):
            lines[-1] += "  (dst exists, skip)"
        else:
            os.makedirs(dst_seed, exist_ok=True)
            shutil.move(src, dst)
        shell_path = os.path.join(exp, sh)
        src_top = os.path.relpath(src, shell_path).split(os.sep)[0]  # the DATASET_NAME dir we drained
        if move_siblings:                      # relocate fold_<k>/external_* etc.
            for sib in list(os.listdir(shell_path)) if os.path.isdir(shell_path) else []:
                if sib == src_top:
                    continue
                sp = os.path.join(shell_path, sib)
                if os.path.isdir(sp):
                    d2 = os.path.join(dst, sib)
                    if not os.path.exists(d2):
                        shutil.move(sp, d2)
                        lines.append(f"        + sibling {sh}/{sib} -> {tag}/{sib}")
        if rm_empty:
            rm = prune_empty(os.path.dirname(src), exp)  # start at the emptied seed_ dir, walk up
            if rm:
                lines.append(f"        rm {', '.join(os.path.relpath(r, exp) for r in rm)}")
            if os.path.isdir(shell_path):
                lines.append(f"        kept {sh}/ (still has {', '.join(os.listdir(shell_path))})")
    if apply:
        mp = merge_folds(dst_seed)
        lines.append(f"      merged -> {os.path.relpath(mp, exp) if mp else '(no complete split)'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--glob", default="AB_MIL_*")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rm-empty", action="store_true", help="prune emptied shell dirs (keeps non-empty, e.g. fold_<k>/external_*)")
    ap.add_argument("--move-siblings", action="store_true", help="also relocate fold_<k>/external_* into the flat fold dir")
    a = ap.parse_args()
    exps = sorted(d for g in a.glob.split(",") for d in glob.glob(os.path.join(a.root, g)) if os.path.isdir(d))
    if not exps:
        raise SystemExit(f"no exp dirs match {a.glob!r} under {a.root}")
    print(f"{'APPLY' if a.apply else 'DRY-RUN'}  root={a.root}  ({len(exps)} dirs)\n")
    for e in exps:
        print(process_exp(e, a.apply, a.rm_empty, a.move_siblings))
    if not a.apply:
        print("\n(dry-run — re-run with --apply; add --rm-empty to prune old shells, --move-siblings to relocate external_* too)")


if __name__ == "__main__":
    main()
