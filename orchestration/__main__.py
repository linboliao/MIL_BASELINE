"""CLI composition root; scientific logic belongs in the study/input components."""
import argparse
import json
from pathlib import Path
from .study import Study
from .inputs import FrozenInputs
from .audit import CellAudit


def main():
    parser = argparse.ArgumentParser(description='NAS-backed PFM LOCO orchestration')
    commands = parser.add_subparsers(dest='command', required=True)
    profile = str(Path(__file__).parent/'profiles/pfm_loco.json')
    for name in ('plan','run'):
        p = commands.add_parser(name)
        p.add_argument('--profile', default=profile)
        p.add_argument('--pfms', nargs='+')
        p.add_argument('--domains', nargs='+')
        if name == 'plan':
            p.add_argument('--check-features', action='store_true')
        else:
            p.add_argument('--run-id', required=True)
            p.add_argument('--gpu', required=True)
            p.add_argument('--preload-gb', type=float, default=0,
                           help='Tensor payload budget PER train/selection split; default off')
    audit = commands.add_parser('audit')
    audit.add_argument('--cell', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(CellAudit.inspect(args.cell), indent=2))
        return
    study = Study.load(args.profile)
    cells = study.cells(args.pfms, args.domains)
    if args.command == 'plan':
        print(json.dumps({'study':study.name,'execution':'none','cells':[
            FrozenInputs(c,args.check_features).summary() for c in cells]}, indent=2))
    else:
        from .execution import LocalExecutor
        print(LocalExecutor(args.gpu).run(study,cells,args.run_id,args.preload_gb))


if __name__ == '__main__':
    main()
