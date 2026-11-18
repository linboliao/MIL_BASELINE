# RP support A/D/B

Run `bash experiments/rp_support/run.sh` on 195 for a read-only check
of the mSTAR fold 1 A/D example. With a fresh run ID and physical GPU
index, the script runs those two arms serially. B is an optional line.

The local entry point takes an explicit frozen JSON config. It resolves
training and the shared RP-free Selection from that file. Each execution
creates a new provenance output and records the resolved config, input
hashes, per-600-update validation and complete checkpoint artifacts.
The fixed 30,000-update loop is adapted from the archived
run_fullreview_matrix_v3.py. It uses the byte-identical repo AB-MIL model
and preloads the two roles directly from NAS with the original RAM check.

`evaluate_scores.py` accepts existing C0, C1 and test slide-score CSVs.
It writes the source95 thresholds before opening test scores. This is
a score-level evaluation utility. Checkpoint-to-score inference and the
historical joint bootstrap are outside this launcher.
