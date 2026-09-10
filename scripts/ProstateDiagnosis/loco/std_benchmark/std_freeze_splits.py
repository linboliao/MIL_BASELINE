#!/usr/bin/env python3
"""Phase-1 standardized benchmark — freeze the patient splits ONCE.

Builds every (model, mode) fold CSV via the existing loco_build_folds.py logic,
then records a manifest of *stem-level* split hashes (feature-path-prefix
independent) so every later step can assert "same frozen split". The patient
split is model-independent for a given mode; this script verifies that and
fails loudly if any model's split diverges.

  python std_freeze_splits.py --repo <MIL_BASELINE> \
      --models conch uni uni2 virchow2 h-optimus-1 mstar gigapath gpfm \
      --modes internal type \
      --out consolidated_frozen_splits/     # manifest dir (in-repo, committable)

Writes:
  <out>/frozen_splits_manifest.json   # per (mode,fold): held_out, n, sha256 of
                                       #   sorted train/val/test slide stems + labels
  (fold CSVs themselves land in datasets/.../AB_MIL_<model>_loco_<mode>/ as before)
"""
import argparse, hashlib, json, os, subprocess, sys, csv


def stem_split_hash(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    h = {}
    for pre in ("train", "val", "test"):
        items = sorted(
            (os.path.splitext(os.path.basename(r[f"{pre}_slide_path"]))[0],
             str(r[f"{pre}_label"]))
            for r in rows if r.get(f"{pre}_slide_path"))
        blob = "\n".join(f"{s}\t{l}" for s, l in items)
        h[pre] = {"n": len(items),
                  "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["internal", "type"])
    ap.add_argument("--out", default="consolidated_frozen_splits")
    a = ap.parse_args()
    bf = os.path.join(a.repo, "scripts/ProstateDiagnosis/loco/loco_build_folds.py")
    py = sys.executable
    os.makedirs(os.path.join(a.repo, a.out), exist_ok=True)

    ref = {}          # (mode, fold) -> stem hash  (from first model)
    manifest = {"modes": {}, "models_checked": a.models}
    for mi, model in enumerate(a.models):
        for mode in a.modes:
            r = subprocess.run([py, bf, "--model", model, "--mode", mode],
                               cwd=os.path.dirname(bf), capture_output=True, text=True)
            if r.returncode:
                print(r.stdout, r.stderr); sys.exit(f"build_folds failed: {model}/{mode}")
            ds = os.path.join(a.repo, "datasets/ProstateDiagnosis/DataAnalysis",
                              f"AB_MIL_{model}_loco_{mode}")
            cmap = json.load(open(os.path.join(ds, "fold_center_map.json")))
            for fk in sorted(cmap):
                fn = fk.split("_")[1]
                cp = os.path.join(ds, f"prostate_loco_{model}_{mode}_{fn}fold.csv")
                sh = stem_split_hash(cp)
                key = f"{mode}/{fk}"
                if mi == 0:
                    ref[key] = sh
                    manifest["modes"][key] = {"held_out": cmap[fk]["held_out"], "split": sh}
                else:
                    if sh != ref[key]:
                        sys.exit(f"SPLIT DRIFT: {model} {key} != reference\n"
                                 f"  ref  {ref[key]}\n  this {sh}")
            print(f"ok  {model}/{mode}")
    manifest["note"] = ("stem-level split is identical across all listed models "
                        "for each mode/fold; feature-path prefix varies by cache "
                        "location only and is NOT hashed.")
    outp = os.path.join(a.repo, a.out, "frozen_splits_manifest.json")
    json.dump(manifest, open(outp, "w"), ensure_ascii=False, indent=1)
    print("\nwrote", outp)
    print("verify later:  python std_freeze_splits.py --models <one-model> "
          "(re-run and diff the manifest)")


if __name__ == "__main__":
    main()
