"""Emit ONE yaml per (model, mode). train_mil.py's built-in k-fold loop then
consumes all N fold CSVs from dataset_root_dir in a single invocation, so the
results land in ONE shared timestamped dir:

  result/ProstateDiagnosis/DataAnalysis/AB_MIL_<model>_loco_<mode>/AB_MIL/seed_42_<ts>/fold_{1..N}/
                                                                  + merge_<N>_fold_metrics.json

(DATASET_NAME is the folder name; train_mil.py inserts <DATASET_NAME>/<MODEL_NAME>
under log_root_dir, so log_root_dir is just .../DataAnalysis.)
"""
import argparse
import os
from pathlib import Path

TEMPLATE = """General:
  MODEL_NAME: AB_MIL
  seed: 42
  num_classes: 2
  num_epochs: 50
  device: {gpu}
  num_workers: 4
  best_model_metric: macro_f1
  earlystop:
    use: true
    patience: 15
    metric: macro_f1

Dataset:
  DATASET_NAME: AB_MIL_{model}_loco_{mode}
  dataset_csv_path: null
  dataset_root_dir: datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_{model}_loco_{mode}
  balanced_sampler:
    use: false
    replacement: true

Logs:
  log_root_dir: result/ProstateDiagnosis/DataAnalysis

Model:
  in_dim: {in_dim}
  L: 512
  D: 128
  dropout: 0.1
  act: relu
  optimizer:
    which: adam
    adam_config:
      lr: 0.0002
      weight_decay: 1.0e-05
    adamw_config:
      lr: 0.0002
      weight_decay: 1.0e-05
  criterion:
    loss: ce
  scheduler:
    warmup: 2
    which: step
    step_config:
      step_size: 3
      gamma: 0.9
    multi_step_config:
      milestones: [20, 30, 40]
      gamma: 0.9
    exponential_config:
      gamma: 0.9
    cosine_config:
      T_max: 10
      eta_min: 0.0001
"""

ROOT = str(Path(__file__).resolve().parents[3])  # repo root, inferred from this file's location

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", required=True, choices=["internal", "fivesite", "type"])
ap.add_argument("--in_dim", type=int, required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--nfold", type=int, default=0, help="ignored (kept for back-compat)")
a = ap.parse_args()

out = f"{ROOT}/configs/ProstateDiagnosis/DataAnalysis"
os.makedirs(out, exist_ok=True)
p = f"{out}/AB_MIL_{a.model}_loco_{a.mode}.yaml"
open(p, "w").write(TEMPLATE.format(gpu=a.gpu, model=a.model, mode=a.mode, in_dim=a.in_dim))
print("wrote", p, "in_dim", a.in_dim, "gpu", a.gpu)
