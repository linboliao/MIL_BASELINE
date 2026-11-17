"""Execute ONE supplied config/dataset pair, independently of any experiment matrix."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
import yaml


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def validate_dataset(path, require_train=True, forbid_test=False):
    data = rows(path)
    if not data:
        raise ValueError('Empty dataset: '+str(path))
    for group in (('train','val') if require_train else ('test',)):
        if group+'_slide_path' not in data[0] or group+'_label' not in data[0]:
            raise ValueError('Missing paired columns for '+group)
        count = 0
        for row in data:
            feature, label = row[group+'_slide_path'], row[group+'_label']
            if bool(feature) != bool(label):
                raise ValueError('Unpaired path/label for '+group)
            if feature:
                if not Path(feature).is_absolute() or not Path(feature).is_file():
                    raise ValueError('Missing or nonabsolute feature: '+feature)
                if not float(label).is_integer() or float(label) < 0:
                    raise ValueError('Invalid class label: '+label)
                count += 1
        if not count:
            raise ValueError('Empty '+group+' split')
    if forbid_test and any(r.get('test_slide_path') or r.get('test_label') for r in data):
        raise ValueError('Independent post-training test requires a train/val-only dataset')
    return len(data)


def selected_checkpoint(output):
    logs = list((Path(output)/'training').rglob('Best_Log_*.csv'))
    if len(logs) != 1 or len(rows(logs[0])) != 1:
        raise ValueError('Expected one selected checkpoint log')
    epoch = int(rows(logs[0])[0]['epoch'])
    path = logs[0].parent/('Best_EPOCH_'+str(epoch)+'.pth')
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Selected checkpoint is missing')
    return path


def prepare(config, dataset, output, test_dataset=None, tie_break=None):
    """Write runtime snapshots; no model-specific or study-specific path knowledge."""
    config, dataset, output = Path(config), Path(dataset), Path(output).resolve()
    cfg = yaml.safe_load(config.read_text())
    validate_dataset(dataset, forbid_test=test_dataset is not None)
    if test_dataset is not None:
        validate_dataset(test_dataset, require_train=False)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for name, source in [('source_config.yaml',config),('dataset.csv',dataset)]+(
            [('test_dataset.csv',Path(test_dataset))] if test_dataset else []):
        target = output/name
        with target.open('xb') as f:
            f.write(source.read_bytes())
        snapshots[name] = {'source':str(source.resolve()),'sha256':sha256(target)}
    cfg = copy.deepcopy(cfg)
    cfg['Dataset'].update(dataset_csv_path=str(output/'dataset.csv'),dataset_root_dir={})
    cfg['General'].update(device=0,retain_all_checkpoints=True)
    if tie_break is not None:
        cfg['General']['checkpoint_tie_break'] = tie_break
    cfg['Logs'] = {'log_root_dir':str(output/'training')}
    with (output/'runtime.yaml').open('x') as f:
        yaml.safe_dump(cfg,f,sort_keys=False)
    return snapshots


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    host = socket.gethostname()
    interpreters = {'server94':'/data12/jing/anaconda3/envs/PrePATH/bin/python',
                    'ubuntu':'/home/lbliao/anaconda3/envs/clam/bin/python'}
    if host not in interpreters or Path(sys.executable) != Path(interpreters[host]):
        raise RuntimeError('Use the project-mapped interpreter on 138 or 195')
    if not args.gpu.isdigit():
        raise ValueError('gpu must be one physical GPU index')
    output = Path(args.output).resolve()
    provenance = Path('/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance')
    if provenance not in output.parents:
        raise ValueError('Execution output must be under project provenance')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('Commit source changes before execution')
    snapshots = prepare(args.config,args.dataset,output,args.test_dataset,args.tie_break)
    write_json(output/'run_meta.json', {'purpose':'reproduction','status_at_creation':'planned',
               'scientific_status':'candidate_pending_human_review','hostname':host,'python':sys.executable,
               'gpu':args.gpu,'repository':str(repo),'git_commit':commit,'inputs':snapshots,
               'started_at':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
               'output_path':str(output),'reproduction_mode':'current_implementation'})
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    if host == 'ubuntu':
        env['LD_LIBRARY_PATH'] = str(Path(sys.executable).parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    def invoke(stage, command):
        write_json(output/(stage+'_command.json'), {'command':command,'cwd':str(repo),'gpu':args.gpu})
        with (output/(stage+'.log')).open('x') as log:
            subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        invoke('train',[sys.executable,str(repo/'train_mil.py'),'--yaml_path',str(output/'runtime.yaml')])
        checkpoint = selected_checkpoint(output)
        if args.test_dataset:
            invoke('test',[sys.executable,str(repo/'test_mil.py'),'--yaml_path',str(output/'runtime.yaml'),
                          '--test_dataset_csv',str(output/'test_dataset.csv'),'--model_weight_path',str(checkpoint),
                          '--test_log_dir',str(output/'evaluation'),'--device','cuda:0'])
        write_json(output/'completed.json', {'status':'completed','checkpoint':str(checkpoint),
                   'checkpoint_sha256':sha256(checkpoint),'runtime_sha256':sha256(output/'runtime.yaml'),
                   'heldout_test_performed':bool(args.test_dataset)})
    except Exception as error:
        write_json(output/'failed.json', {'status':'failed','error':repr(error)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--dataset',required=True,help='One wide-format train/val CSV, not a fold directory')
    p.add_argument('--test-dataset',help='Optional independent wide-format test CSV')
    p.add_argument('--output',required=True,help='New unique attempt directory')
    p.add_argument('--gpu',required=True)
    p.add_argument('--tie-break',choices=('none','val_loss'),default=None,help='Omit to preserve source config')
    p.add_argument('--validate-only',action='store_true',help='Read inputs, do not create outputs or train')
    args=p.parse_args()
    if args.validate_only:
        yaml.safe_load(Path(args.config).read_text())
        validate_dataset(args.dataset,forbid_test=args.test_dataset is not None)
        if args.test_dataset:
            validate_dataset(args.test_dataset,require_train=False)
        print('Config and dataset inputs valid; no execution')
    else:
        execute(args)


if __name__=='__main__':
    main()
