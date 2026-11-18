"""Verify the frozen Panel A/B inputs without running inference or statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    cfg = json.loads(args.config.read_text())
    for name, record in json.loads((root/'manifest.json').read_text()).items():
        if sha(root/name) != record['sha256']:
            raise ValueError(f'Changed frozen file: {name}')
    with (root/cfg['models']).open(encoding='utf-8-sig') as f:
        models = list(csv.DictReader(f))
    for row in models:
        if sha(row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError(f'Changed checkpoint: {row["model_id"]}')
    print(f'Validated frozen inputs and {len(models)} checkpoints')


if __name__ == '__main__':
    main()
