"""Host-local execution backend. No SSH, resource allocation, or cache deletion."""
from pathlib import Path
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from .audit import CellAudit
from .inputs import FrozenInputs, digest


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class LocalExecutor:
    interpreters = {'server94':'/data12/jing/anaconda3/envs/PrePATH/bin/python',
                    'ubuntu':'/home/lbliao/anaconda3/envs/clam/bin/python'}

    def __init__(self, gpu):
        self.repo = Path(__file__).resolve().parents[1]
        self.host = socket.gethostname()
        if self.host not in self.interpreters or Path(sys.executable) != Path(self.interpreters[self.host]):
            raise RuntimeError('Use the project-mapped interpreter on host 138 or 195')
        if not str(gpu).isdigit():
            raise ValueError('Specify one physical GPU index')
        self.gpu = str(gpu)
        self.commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.repo,text=True).strip()
        if subprocess.check_output(['git','status','--porcelain'],cwd=self.repo,text=True).strip():
            raise RuntimeError('Execution requires a committed, clean checkout')

    def _invoke(self, args, logfile):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.gpu
        env['PYTHONPATH'] = str(self.repo) + os.pathsep + env.get('PYTHONPATH','')
        if self.host == 'ubuntu':
            env['LD_LIBRARY_PATH'] = str(Path(sys.executable).parent.parent/'lib') + os.pathsep + env.get('LD_LIBRARY_PATH','')
        write_json(Path(str(logfile)+'.command.json'), {'command':[sys.executable]+[str(a) for a in args],
                   'cwd':str(self.repo),'CUDA_VISIBLE_DEVICES':self.gpu})
        with Path(logfile).open('x') as f:
            subprocess.run([sys.executable]+[str(a) for a in args],cwd=self.repo,env=env,
                           stdout=f,stderr=subprocess.STDOUT,check=True)

    def run(self, study, cells, run_id, preload_gb=0):
        if not run_id or Path(run_id).name != run_id or run_id in ('.','..'):
            raise ValueError('run_id must be one path component')
        if not math.isfinite(preload_gb) or preload_gb < 0:
            raise ValueError('preload_gb must be nonnegative')
        # Validate the full selection before allocating the attempt directory.
        prepared = [FrozenInputs(cell, check_features=True) for cell in cells]
        root = study.output_parent/run_id
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'run_meta.json', {'study':study.name,'run_tag':run_id,'status':'planned',
                   'scientific_status':'candidate_pending_human_review','git_commit':self.commit,
                   'repository':str(self.repo),'hostname':self.host,'python':sys.executable,
                   'gpu':self.gpu,'source':str(study.source),'output_path':str(root),
                   'started_at':datetime.now(timezone.utc).isoformat(),
                   'reproduction_mode':'current_implementation','cells':[c.key for c in cells]})
        for inputs in prepared:
            cell = root/inputs.cell.pfm/inputs.cell.domain
            cell.mkdir(parents=True, exist_ok=False)
            write_json(cell/'inputs_meta.json', inputs.summary())
            try:
                config = inputs.materialize(cell, preload_gb)
                self._invoke(['train_mil.py','--yaml_path',config],cell/'train.log')
                checkpoint = CellAudit.checkpoint(cell)
                self._invoke(['test_mil.py','--yaml_path',config,'--test_dataset_csv',cell/'inputs/test.csv',
                              '--model_weight_path',checkpoint,'--test_log_dir',cell/'evaluation',
                              '--device','cuda:0'],cell/'test.log')
                result = CellAudit.inspect(cell)
                result['runtime_config_sha256'] = digest(config)
                result['checkpoint_sha256'] = digest(checkpoint)
                result['predictions_sha256'] = digest(cell/'evaluation/Infer_Result.csv')
                write_json(cell/'completed.json', result)
            except Exception as error:
                write_json(cell/'failed.json', {'status':'failed','error':repr(error)})
                raise
        write_json(root/'completed.json', {'status':'completed','cells':len(cells)})
        return root
