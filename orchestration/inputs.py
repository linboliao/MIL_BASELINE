"""Adapt frozen role CSVs to the training API without modifying NAS sources."""
import copy
import csv
import hashlib
from pathlib import Path
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fields):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FrozenInputs:
    roles = ('train', 'selection', 'calibration', 'test')

    def __init__(self, cell, check_features=False):
        self.cell = cell
        self.config = yaml.safe_load(cell.config.read_text())
        if self.config['General']['MODEL_NAME'] != 'AB_MIL':
            raise ValueError('PFM LOCO adapter currently supports AB_MIL only')
        self.rows = {r: read_rows(cell.splits/(r+'.csv')) for r in self.roles}
        self.hashes = {str(cell.config): digest(cell.config)}
        patients, slides = set(), set()
        for role, rows in self.rows.items():
            if not rows:
                raise ValueError('Empty role: ' + role)
            current_patients, current_slides = set(), set()
            for row in rows:
                if row['label'] not in ('0', '1') or not row['patient_group_id']:
                    raise ValueError('Invalid label or patient identity')
                path = Path(row['slide_path'])
                if not path.is_absolute() or str(path) in current_slides:
                    raise ValueError('Nonabsolute or duplicate feature path')
                if row.get('partition', role) != role or row.get('model', cell.pfm) != cell.pfm:
                    raise ValueError('Role or PFM mismatch')
                if check_features:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if row.get('feature_bytes') and path.stat().st_size != int(row['feature_bytes']):
                        raise ValueError('Feature size changed: ' + str(path))
                current_slides.add(str(path))
                current_patients.add(row['patient_group_id'])
            if patients & current_patients or slides & current_slides:
                raise ValueError('Cross-role patient or slide overlap: ' + role)
            patients.update(current_patients)
            slides.update(current_slides)
            self.hashes[str(cell.splits/(role+'.csv'))] = digest(cell.splits/(role+'.csv'))

    def summary(self):
        return {'cell': self.cell.key, 'counts': {r: len(v) for r,v in self.rows.items()},
                'input_sha256': self.hashes}

    def materialize(self, root, preload_gb=0):
        root = Path(root)
        inputs = root/'inputs'
        inputs.mkdir()
        # Snapshot the frozen inputs, including the reserved calibration role.
        for role in self.roles:
            source = self.cell.splits/(role+'.csv')
            with (inputs/('frozen_'+role+'.csv')).open('xb') as f:
                f.write(source.read_bytes())
            if digest(inputs/('frozen_'+role+'.csv')) != self.hashes[str(source)]:
                raise RuntimeError('Input changed during preparation')
        with (inputs/'source_model.yaml').open('xb') as f:
            f.write(self.cell.config.read_bytes())
        if digest(inputs/'source_model.yaml') != self.hashes[str(self.cell.config)]:
            raise RuntimeError('Config changed during preparation')
        fields = ['train_slide_path','train_label','val_slide_path','val_label','test_slide_path','test_label']
        rows = []
        for i in range(max(len(self.rows['train']), len(self.rows['selection']))):
            row = dict.fromkeys(fields, '')
            for role, prefix in (('train','train'),('selection','val')):
                if i < len(self.rows[role]):
                    source = self.rows[role][i]
                    row[prefix+'_slide_path'] = source['slide_path']
                    row[prefix+'_label'] = source['label']
            rows.append(row)
        write_rows(inputs/'train_val.csv', rows, fields)
        write_rows(inputs/'test.csv', [{'test_slide_path':r['slide_path'], 'test_label':r['label']}
                                     for r in self.rows['test']], ['test_slide_path','test_label'])
        config = copy.deepcopy(self.config)
        config['General'].update(device=0, num_workers=0, best_model_metric='macro_f1',
                                 checkpoint_tie_break='val_loss', checkpoint_tie_epsilon=1e-9,
                                 retain_all_checkpoints=True)
        config['Dataset'].update(dataset_csv_path=str(inputs/'train_val.csv'), dataset_root_dir={},
                                 preload={'train':preload_gb>0, 'val':preload_gb>0, 'test':False},
                                 preload_mem_gb={'train':preload_gb,'val':preload_gb,'test':0},
                                 preload_workers=1)
        config['Logs'] = {'log_root_dir': str(root/'training')}
        with (root/'model.yaml').open('x') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return root/'model.yaml'
