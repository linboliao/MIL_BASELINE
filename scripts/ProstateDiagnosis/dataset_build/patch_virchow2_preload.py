import os

OUT_DIR = '/NAS3/lbliao/Code-138/MIL_BASELINE/configs/ProstateDiagnosis/DataAnalysis/AB_MIL_virchow2_5fold_3center'

INSERT = """  preload: true
  preload_mem_gb:
    train: 52
    val: 65
    test: 60
  balanced_sampler:"""

for fold in range(1, 6):
    path = os.path.join(OUT_DIR, f'fold_{fold}.yaml')
    with open(path) as f:
        content = f.read()
    if 'preload:' in content:
        print(f'fold{fold}: already patched, skipping')
        continue
    content = content.replace('  balanced_sampler:', INSERT)
    with open(path, 'w') as f:
        f.write(content)
    print(f'fold{fold}: patched {path}')
