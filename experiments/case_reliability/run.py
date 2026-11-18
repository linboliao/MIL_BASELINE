"""Validate frozen inputs or run the LOCO and Panel reliability analyses."""
import argparse
import json
from pathlib import Path

from experiments.threshold_transport.analyze import at_level, load
from experiments.threshold_transport.statistics import sha, writecsv
from .loco import analyze_cell
from .panel import read_panel, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.config.resolve().parents[1]
    cfg = json.loads(args.config.read_text())
    if not 0 < cfg['coverage'] < 1 or cfg['bootstrap_draws'] < 1:
        parser.error('Invalid review coverage or bootstrap count')
    for name, item in json.loads((root/'manifest.json').read_text()).items():
        if sha(root/name) != item['sha256']:
            raise ValueError(f'Frozen file changed: {name}')
    for cell in cfg['cells']:
        a,b = (at_level(load(root, cell[k]),'patient') for k in ('calibration','target'))
        if set(a.patient_group_id) & set(b.patient_group_id):
            raise ValueError('Overlapping patients')
    predictions, pairs = read_panel(root,cfg)
    if args.output is None:
        print(f'Validated {len(cfg["cells"])} LOCO cells and {len(predictions)} Panel prediction rows')
        return
    output = args.output.resolve()
    if root.parent/'provenance' not in output.parents:
        parser.error('Output must be below project provenance')
    output.mkdir(parents=True,exist_ok=False)
    rows = [analyze_cell(root,c,cfg['coverage'],cfg['bootstrap_draws'],cfg['seed']) for c in cfg['cells']]
    writecsv(output/'loco_review.csv',rows)
    writecsv(output/'panel_anchor.csv',summarize(predictions,pairs,cfg['bootstrap_draws'],cfg['seed']))
    with (output/'completion.json').open('x') as f:
        json.dump(dict(status='completed',config_sha256=sha(args.config),
                       input_manifest_sha256=sha(root/'manifest.json')),f)


if __name__ == '__main__':
    main()
