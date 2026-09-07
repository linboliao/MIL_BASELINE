import argparse
import os

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
  DATASET_NAME: ProstateDiagnosis_{model}_3center_fold{fold}
  dataset_csv_path: null
  dataset_root_dir: datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_{model}_5fold_3center/fold_{fold}
  balanced_sampler:
    use: false
    replacement: true

Logs:
  log_root_dir: result/ProstateDiagnosis/DataAnalysis/AB_MIL_{model}_5fold_3center/fold_{fold}

Model:
  in_dim: 256
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

ROOT = '/NAS3/lbliao/Code-138/MIL_BASELINE'  # adjust to your MIL_BASELINE checkout

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
parser.add_argument('--gpu', type=int, default=0, help='GPU for this single fold\'s classifier training')
args = parser.parse_args()

model = f'conch_psir_fold{args.fold}'
out_dir = f'{ROOT}/configs/ProstateDiagnosis/DataAnalysis/AB_MIL_{model}_5fold_3center'
os.makedirs(out_dir, exist_ok=True)

for cv_fold in range(1, 6):
    content = TEMPLATE.format(fold=cv_fold, gpu=args.gpu, model=model)
    path = os.path.join(out_dir, f'fold_{cv_fold}.yaml')
    with open(path, 'w') as f:
        f.write(content)
    print('wrote', path, 'gpu', args.gpu)
