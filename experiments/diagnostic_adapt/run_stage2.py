"""Stage2 repeat0 runner. No evaluation truth reader or stage3 metrics.
Run from the 195 checkout with the project's absolute clam interpreter.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from .artifacts import put, require, sha, utc
from .inputs import load_policy
from .run_fold import BudgetStop, run_fold

REPO = Path(__file__).resolve().parents[2]
PROJECT = Path('/NAS145/linboliao/Projects/ProstateDiagnosis')
BASE = Path('/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance/generalization/diagnostic-adaptation-mve')
HERE = Path(__file__).resolve().parent
POLICY = HERE / 'policy.py'
IO_AMENDMENT = HERE / 'io_amendment.json'


def identity():
    require(socket.gethostname() == 'ubuntu', 'runner currently validated on195 only')
    require(sys.executable == '/home/lbliao/anaconda3/envs/clam/bin/python', 'wrong Python')
    require(Path.cwd().resolve() == REPO, 'explicit MIL checkout working directory required')
    require(Path(os.environ.get('TMPDIR', '')).resolve() == PROJECT / '.agent/nodes/195/tmp', 'wrong TMPDIR')


def source_manifest():
    paths = [Path(__file__), POLICY, IO_AMENDMENT, HERE / 'run.sh', HERE / 'inputs.py', HERE / 'artifacts.py',
             HERE / 'scoring.py', HERE / 'run_fold.py',
             REPO / 'modules/AB_MIL/ab_mil.py']
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def unique_output(path):
    p = Path(path).absolute()
    require(p.parent.resolve() == BASE.resolve(), 'output must be a new direct child of governed MVE provenance')
    require(not p.exists(), 'output exists; use a new attempt ID')
    return p


def run_serial(args):
    identity()
    c, _ = load_policy(CONFIG, CONFIG_SHA, DATASETS, POLICY)
    require(os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit(), 'set one numeric CUDA_VISIBLE_DEVICES')
    root = unique_output(args.output)
    root.mkdir(exist_ok=False)
    put(root / 'run_meta.json', dict(study='diagnostic-adaptation-mve', run_tag=root.name, status='running',
            scientific_status='candidate_pending_human_review', parent_run=str(CONFIG.parent), config_sha256=CONFIG_SHA,
            hostname=socket.gethostname(), host_alias='195', python=sys.executable, repository=str(REPO),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)),
            seed=42, model='UNI2+ABMIL ordinary single-view ERM', checkpoint_rule='step300',
            GPU=os.environ['CUDA_VISIBLE_DEVICES'], started_at=utc(), code_manifest=source_manifest(), output_path=str(root)))
    start = time.monotonic()
    durations = []
    try:
        for slug in c['fold_order']:
            elapsed = time.monotonic() - start
            # Allocate no further fold if a previous full fold predicts the remaining budget is insufficient.
            predicted = max(durations) if durations else 0
            if elapsed + predicted > 24 * 3600 or predicted > 12 * 3600:
                raise BudgetStop('next fold would exceed projected cumulative24h or single-job12h budget')
            t = time.monotonic()
            cmd = [sys.executable, '-B', '-m', 'experiments.diagnostic_adapt.run_stage2',
                   '--config', str(CONFIG), '--datasets', str(DATASETS),
                   '--prepath', str(PREPATH), '--config-sha256', CONFIG_SHA, '_fold', '--fold', slug,
                   '--output', str(root / slug), '--used-seconds', str(elapsed)]
            with (root / (slug + '_console.log')).open('x') as log:
                subprocess.run(cmd, cwd=REPO, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
            require((root / slug / 'FOLD_COMPLETE.json').is_file(), 'fold exited without completion artifact')
            durations.append(time.monotonic() - t)
            put(root / (slug + '_timing.json'), dict(allocated_GPU_wall_seconds=durations[-1], cumulative_seconds=time.monotonic()-start))
        put(root / 'STAGE_2_COMPLETE.json', dict(at=utc(), gate='STOP_GATE_2', folds=c['fold_order'],
                    allocated_GPU_wall_seconds=time.monotonic()-start, evaluation_truth_read=False, stage3_executed=False))
        with (root / 'STAGE_2_REPORT.md').open('x') as f:
            f.write('# Stage2 execution completed — STOP GATE2\n\nThree serial outer folds completed. See fold checkpoint lineage, action locks, read audits, unlabelled predictions and timing. Evaluation truth was not opened; no stage3 metrics. Human confirmation required before stage3.\n')
    except BaseException as exc:
        put(root / 'TERMINAL_STATUS.json', dict(status='budget_stopped' if isinstance(exc, BudgetStop) else 'failed_or_partial',
                    at=utc(), message=str(exc), traceback=traceback.format_exc(), preserve_partial_outputs=True))
        raise
    else:
        put(root / 'TERMINAL_STATUS.json', dict(status='completed', gate='STOP_GATE_2', at=utc()))
    print(root, flush=True)


def main():
    global CONFIG, CONFIG_SHA, DATASETS, PREPATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--datasets', required=True, type=Path)
    parser.add_argument('--prepath', required=True, type=Path)
    parser.add_argument('--config-sha256')
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='configuration-only check; no cohort feature/model run')
    run = sub.add_parser('run', help='real stage2 execution, only after user authorization')
    run.add_argument('--output', required=True)
    fold = sub.add_parser('_fold', help=argparse.SUPPRESS)
    fold.add_argument('--fold', choices=['SL', 'XC', 'YN'], required=True)
    fold.add_argument('--output', required=True)
    fold.add_argument('--used-seconds', type=float, required=True)
    args = parser.parse_args()
    CONFIG = args.config.resolve(strict=True)
    CONFIG_SHA = args.config_sha256 or sha(CONFIG)
    DATASETS = args.datasets.resolve(strict=True)
    PREPATH = args.prepath.resolve(strict=True)
    if args.command == 'plan':
        identity()
        c, _ = load_policy(CONFIG, CONFIG_SHA, DATASETS, POLICY)
        print(json.dumps(dict(config_sha256=CONFIG_SHA, folds=c['fold_order'], training_updates=900,
                              scoring_upper_bound=6380, real_execution=False), indent=2))
    elif args.command == 'run':
        run_serial(args)
    else:
        require(Path(args.output).absolute().parent.parent.resolve() == BASE.resolve(), 'invalid fold output parent')
        require(Path(args.output).name == args.fold, 'fold output name mismatch')
        require(Path(args.output).parent.joinpath('run_meta.json').is_file(), 'use serial run entry point')
        identity()
        c, policy = load_policy(CONFIG, CONFIG_SHA, DATASETS, POLICY)
        run_fold(args, c=c, policy=policy, config_sha=CONFIG_SHA, prepath=PREPATH,
                 amendment=IO_AMENDMENT, code_manifest=source_manifest())


if __name__ == '__main__':
    main()
