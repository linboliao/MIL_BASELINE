#!/usr/bin/env python3
"""Scan a std_benchmark tag for missing/failed folds and print a status table
+ a ready-to-use rerun plan. Does NOT rerun anything itself.

  python std_rescan_failed.py --repo <MIL_BASELINE> --tag std-20260911 \
     --models conch uni uni2 virchow2 h-optimus-1 mstar gigapath gpfm

A fold counts as:
  OK       Best_Log_*.csv present
  CRASHED  fold dir exists, Best_EPOCH_*.pth exists, but no Best_Log (died mid-training)
  MISSING  fold dir doesn't exist at all (never started / PFM not reached yet)
"""
import argparse, glob, os, json

FOLDS = {"internal": {1: "省立", 2: "新昌"}, "type": {1: "CNB", 2: "RP", 3: "TURP"}}
DIM = {"conch": 512, "uni": 1024, "uni2": 1536, "virchow2": 2560,
       "h-optimus-1": 1536, "gigapath": 1536, "gpfm": 1024, "mstar": 1024}


def status(repo, model, mode, k, tag):
    fd = os.path.join(repo, "result/ProstateDiagnosis/DataAnalysis",
                      f"AB_MIL_{model}_loco_{mode}", "AB_MIL", f"seed_42_{tag}", f"fold_{k}")
    if not os.path.isdir(fd):
        return "MISSING", fd
    if glob.glob(os.path.join(fd, "Best_Log_*.csv")):
        return "OK", fd
    if glob.glob(os.path.join(fd, "Best_EPOCH_*.pth")):
        return "CRASHED", fd
    return "MISSING", fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()

    rows, todo = [], []
    for m in a.models:
        for mode, fdict in FOLDS.items():
            for k, ho in fdict.items():
                st, fd = status(a.repo, m, mode, k, a.tag)
                rows.append((m, mode, k, ho, st))
                if st != "OK":
                    todo.append({"model": m, "mode": mode, "fold": k, "held_out": ho,
                                "status": st, "in_dim": DIM[m]})

    print(f"{'PFM':13} {'mode':9} {'fold':5} {'held_out':9} status")
    print("-" * 50)
    for m, mode, k, ho, st in rows:
        flag = "" if st == "OK" else "  <-- needs rerun"
        print(f"{m:13} {mode:9} {k:<5} {ho:9} {st}{flag}")

    n_ok = sum(1 for r in rows if r[4] == "OK")
    print(f"\n{n_ok}/{len(rows)} folds OK. {len(todo)} need rerun.")
    if todo:
        out = os.path.join(a.repo, f"std_rerun_plan_{a.tag}.json")
        json.dump(todo, open(out, "w"), indent=1)
        print(f"wrote rerun plan -> {out}")
        print(f"next:  bash scripts/ProstateDiagnosis/loco/std_benchmark/std_rerun_failed.sh "
              f"--tag {a.tag} --plan {out}")


if __name__ == "__main__":
    main()
