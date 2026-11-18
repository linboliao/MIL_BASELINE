# Contrast PFM benchmark

- `run.sh RUN_ID GPU` is the node 138 serial staging and training recipe.
- `run_195.sh` is the node 195 direct NAS training recipe.

The node 138 recipe retains its original staging module paths under
`scripts/experiments/pfm_loco` so the active serial queue can finish across
the checkout update. Shared tools for new recipes live under `experiments`.
