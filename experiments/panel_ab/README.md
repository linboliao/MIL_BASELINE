# Panel A/B

Run `bash experiments/panel_ab/run.sh` on host 195 to verify the NAS input
package and the six frozen checkpoints. Pass a fresh run name to analyze the
existing frozen predictions. Analysis creates a new provenance output.
`analyze.py` also accepts an explicit predictions CSV in the frozen schema.

The primary table covers FullSource UNI2 seed42 and mSTAR seed42, with A/B
reported separately. Remaining UNI2 seeds go to the sensitivity table.
The report keeps the seed44 versus seed42 paired contrast separate.
The analysis uses 5,000 shared patient-group bootstrap draws within each
Panel and retains the exceptional pair and per-unit correctness tables.

Inputs come from the completed six-model inference release. This recipe
does not run GPU inference. A new prediction pass should use the same
frozen weights, reviewed whitelist and whole-bag feature rule.
