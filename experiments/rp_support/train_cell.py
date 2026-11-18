"""Plan or train one frozen RP support model/fold/arm."""
import argparse
import json
from pathlib import Path

from .artifacts import json_x
from .inputs import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    training = cfg['training']
    expected = dict(seed=42, optimizer_updates=30000, validate_every=600,
                    early_stopping=False, batch_size=1, gradient_accumulation=1,
                    warmup_updates=1000, lr=.0002, weight_decay=.00001,
                    minimum_lr=.00002, L=512, D=128, dropout=.1)
    if any(training[k] != v for k, v in expected.items()) or cfg['arm'] not in ('A', 'D', 'B'):
        parser.error('Config does not match the native fixed-update protocol')
    train, selection = validate(cfg)
    if args.plan:
        print(json.dumps(dict(job=cfg['job_id'], train=len(train), selection=len(selection),
                              updates=30000, validation_checks=50, execution=False)))
        return
    if args.output is None or args.gpu is None or args.gpu < 0:
        parser.error('--output and a nonnegative --gpu are required for training')
    output = args.output.resolve()
    provenance = Path('/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance').resolve()
    if provenance not in output.parents:
        parser.error('Output must be below project provenance')
    from .training import train_job
    from .inputs import preload
    output.mkdir(parents=True, exist_ok=False)
    cfg['run_root'] = str(output)
    cfg['output_base'] = str(output / 'training')
    json_x(output / 'config.json', cfg)
    features = preload((train, selection), cfg['dimension'], output, cfg['model'], 'training')
    train_job(output, output / 'config.json', features, args.gpu)


if __name__ == '__main__':
    main()
