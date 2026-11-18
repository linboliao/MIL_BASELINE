# Case decision reliability

On 195, run `bash experiments/case_reliability/run.sh` for a read-only input
check. Pass a new run ID to write analysis under provenance. No GPU is needed.

LOCO uses patient maxima and source Calibration source95. The fixed review
budget allocates expected review uniformly at tied margins and compares against
the exact random-review expectation. Target-patient bootstrap intervals are
conditional on the frozen threshold. The source-locked margin cutoff is
reported separately. LOCO archives have probabilities only, so the logit
margin comparison is deferred until original logits are available.

Panel reports one-anchor flip, slide error and consistent-but-wrong detection
for UNI2 and mSTAR, separately by Panel. It uses probability and original logit
margins, case/group weighting and patient-group bootstrap.
