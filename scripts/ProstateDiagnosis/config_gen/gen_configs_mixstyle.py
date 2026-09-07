import os

TEMPLATE = """General:
  MODEL_NAME: MIXSTYLE_AB_MIL
  seed: 42
  num_classes: 2
  num_epochs: 50
  device: 0
  num_workers: 4
  best_model_metric: macro_f1
  earlystop:
    use: true
    patience: 15
    metric: macro_f1

Dataset:
  DATASET_NAME: ProstateDiagnosis_uni2_mixstyle_fold{fold}
  dataset_csv_path: null
  dataset_root_dir: datasets/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_centeradv/fold_{fold}
  balanced_sampler:
    use: false
    replacement: true

Logs:
  log_root_dir: result/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_mixstyle/fold_{fold}

Model:
  in_dim: 1536
  L: 512
  D: 128
  dropout: 0.1
  act: relu
  mixstyle:
    num_domains: 3
    mix_prob: 0.5
    mix_alpha: 0.1
    momentum: 0.9
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

OUT_DIR = '/NAS3/lbliao/Code-138/MIL_BASELINE/configs/ProstateDiagnosis/DataAnalysis/AB_MIL_uni2_5fold_mixstyle'
os.makedirs(OUT_DIR, exist_ok=True)

for fold in range(1, 6):
    content = TEMPLATE.format(fold=fold)
    path = os.path.join(OUT_DIR, f'fold_{fold}.yaml')
    with open(path, 'w') as f:
        f.write(content)
    print('wrote', path)
