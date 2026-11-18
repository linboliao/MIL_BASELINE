# Source-locked threshold transport

On 195, run `bash experiments/threshold_transport/run.sh` for read-only
input validation. Supply a unique run ID to compute the RP and SL tables
under project provenance. The analysis reads frozen scores and requires
no GPU. `analyze.py` accepts an explicit configuration and output path.

The threshold is independently fit at slide and patient level on source
Calibration. Patient label and score use group maxima. The lock is written
before reading target scores in execution mode. Target evaluation compares
fixed 0.5 with source95 and counts new and recovered false negatives.
Bootstrap resamples source and target groups independently and shares
draws across PFMs within each domain. Domains are reported separately.

The dataset package preserves input membership and checkpoint lineage.
Its scope is the October RP/SL rebuild. Patient-level analysis uses the
same frozen scores and is a separate measurement level.
