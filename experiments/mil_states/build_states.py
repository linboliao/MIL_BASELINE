"""Build four weight states from one completed standard AB-MIL run."""
import argparse
import csv
import json
import re
from pathlib import Path

import torch


def build(run, output):
    paths = sorted((run / 'epochs').glob('epoch_*_model.pth'))
    epochs = [int(re.fullmatch(r'epoch_(\d+)_model\.pth', p.name)[1]) for p in paths]
    last_path, = run.glob('Last_EPOCH_*.pth')
    last = int(last_path.stem.rsplit('_', 1)[1])
    if epochs != list(range(1, last + 1)):
        raise ValueError('Expected a complete epoch trajectory through Last')
    best_log, = run.glob('Best_Log_*.csv')
    with best_log.open() as f:
        record, = csv.DictReader(f)
    best = int(record['epoch'])
    if best not in epochs:
        raise ValueError('Best epoch missing from trajectory')

    best_state = None
    ema, swa = {}, {}
    window = min(5, last)
    for epoch, path in zip(epochs, paths):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if epoch == best:
            best_state = state
        for key, value in state.items():
            if not value.is_floating_point():
                ema[key] = value.clone()
                if epoch == last:
                    swa[key] = value.clone()
                continue
            value = value.float()
            if epoch == 1:
                ema[key] = value.clone()
            else:
                ema[key].mul_(0.9).add_(value, alpha=0.1)
            if epoch > last - window:
                if key not in swa:
                    swa[key] = value.clone()
                else:
                    swa[key].add_(value)
                if epoch == last:
                    swa[key].div_(window)

    output.mkdir(parents=True)
    for name, weights in [('best', best_state), ('last', state),
                          ('swa_last5', swa), ('ema_epoch_0p9', ema)]:
        with (output / f'{name}.pth').open('xb') as f:
            torch.save(weights, f)
    with (output / 'manifest.json').open('x') as f:
        json.dump(dict(source=str(run.resolve()), inputs=[str(p.resolve()) for p in paths],
                       best_epoch=best, last_epoch=last, swa_epochs=epochs[-window:],
                       ema_epochs=epochs, ema_decay=0.9), f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.run, args.output)
